import firebase_admin
from firebase_admin import credentials, firestore, storage
import functions_framework
from google.cloud import texttospeech, exceptions as google_cloud_exceptions
import os
import replicate
import requests 
from datetime import datetime

# --- 전역 초기화 ---
try:
    firebase_admin.get_app()
except ValueError:
    firebase_admin.initialize_app()

bucket = storage.bucket()  # 기본 Firebase Storage 버킷
db = firestore.client()    # Firestore 클라이언트, 전역 또는 함수별로 초기화 가능

# --- 헬퍼 함수 정의 ---

class OperationFailure(Exception):
    """HTTP 오류를 발생시켜야 하는 작업 실패 시 사용하는 사용자 정의 예외입니다."""
    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code

def download_avatar_image(bucket_client, avatar_storage_path, temp_avatar_path):
    """Firebase Storage에서 아바타 이미지를 다운로드합니다."""
    try:
        blob = bucket_client.blob(avatar_storage_path)
        blob.download_to_filename(temp_avatar_path)
        print(f"아바타 이미지 '{avatar_storage_path}'를 '{temp_avatar_path}'로 다운로드했습니다.")
        return True
    except google_cloud_exceptions.NotFound:
        print(f"오류: '{avatar_storage_path}'에서 아바타 이미지를 찾을 수 없습니다.")
        raise OperationFailure(f"'{avatar_storage_path}'에서 아바타 이미지를 찾을 수 없습니다.", 404)
    except Exception as e:
        print(f"오류: 아바타 '{avatar_storage_path}' 다운로드에 실패했습니다. 세부 정보: {e}")
        raise OperationFailure(f"아바타 이미지 다운로드에 실패했습니다. 서버 오류: {e}", 500)

def generate_tts_audio(script_text, tts_client, temp_audio_path):
    """TTS 오디오를 생성하고 임시 경로에 저장합니다."""
    try:
        synthesis_input = texttospeech.SynthesisInput(text=script_text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name="en-US-Standard-C",
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )
        response = tts_client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        with open(temp_audio_path, "wb") as out:
            out.write(response.audio_content)
        print(f'오디오 콘텐츠를 파일 "{temp_audio_path}"에 작성했습니다.')
        return True
    except Exception as e:
        print(f"TTS 생성 중 오류 발생: {e}")
        raise OperationFailure(f"TTS 생성 실패: {e}", 500)

def perform_lip_sync(replicate_client, temp_avatar_path, temp_audio_path):
    """Replicate API를 호출하여 립싱크를 수행하고 비디오 URL을 반환합니다."""
    print(f"아바타 '{temp_avatar_path}'와 오디오 '{temp_audio_path}'로 립싱크 프로세스를 시작합니다.")
    try:
        output = replicate_client.run(
            "cjwb/sadtalker:3aa2daf61579702c6ba2411452269943457be29cc01be511252541925a0c090d",
            input={
                "source_image": open(temp_avatar_path, "rb"),
                "driven_audio": open(temp_audio_path, "rb"),
                "preprocess": "full",
                "still_mode": True,
                "enhancer": "gfpgan"
            }
        )
        replicate_video_url = output
        if not replicate_video_url: # API 호출이 성공하면 발생하지 않아야 합니다.
             raise OperationFailure("립싱크 프로세스에서 Replicate로부터 비디오 URL을 반환하지 않았습니다.", 500)
        print(f"Replicate에서 생성된 비디오 URL: {replicate_video_url}")
        return replicate_video_url
    except replicate.exceptions.ReplicateError as e:
        print(f"Replicate API 오류: {e}")
        raise OperationFailure(f"Replicate API 오류로 인해 립싱크 생성에 실패했습니다: {e}", 500)
    except Exception as e: # 파일 I/O와 같은 다른 잠재적 오류를 포착합니다.
        print(f"립싱크 Replicate 호출 중 오류 발생: {e}")
        raise OperationFailure(f"립싱크 프로세스 실패: {e}", 500)

def download_replicate_video(video_url, temp_video_path):
    """Replicate URL에서 비디오를 임시 경로로 다운로드합니다."""
    try:
        video_response = requests.get(video_url, stream=True)
        video_response.raise_for_status()  # 잘못된 응답(4XX 또는 5XX)에 대해 HTTPError를 발생시킵니다.
        with open(temp_video_path, "wb") as f:
            for chunk in video_response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Replicate에서 생성된 비디오를 {temp_video_path}(으)로 다운로드했습니다.")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Replicate URL '{video_url}'에서 비디오 다운로드에 실패했습니다: {e}")
        raise OperationFailure(f"Replicate에서 생성된 비디오 다운로드에 실패했습니다: {e}", 500)

def upload_to_firebase_storage(bucket_client, temp_video_path, storage_path):
    """비디오를 Firebase Storage에 업로드하고 공개 URL을 반환합니다."""
    try:
        blob = bucket_client.blob(storage_path)
        blob.upload_from_filename(temp_video_path)
        blob.make_public()
        video_storage_url = blob.public_url
        print(f"비디오를 Firebase Storage에 업로드했습니다: {storage_path}")
        print(f"공개 URL: {video_storage_url}")
        return video_storage_url
    except Exception as e:
        print(f"Firebase Storage의 '{storage_path}'에 비디오 업로드 실패: {e}")
        raise OperationFailure(f"Firebase Storage에 비디오 업로드 실패: {e}", 500)

def save_metadata_to_firestore(db_client, user_id, video_url, script_text, avatar_id):
    """비디오 메타데이터를 Firestore에 저장합니다."""
    try:
        doc_ref_tuple = db_client.collection("video_creations").add({
            "userId": user_id,
            "videoUrl": video_url,
            "scriptText": script_text,
            "avatarId": avatar_id,
            "createdAt": firestore.SERVER_TIMESTAMP,
            "status": "completed"
        })
        print(f"메타데이터를 Firestore에 문서 ID {doc_ref_tuple[1].id}(으)로 저장했습니다.")
    except Exception as e:
        # MVP의 경우 경고를 기록하지만 전체 작업을 실패시키지는 않습니다.
        print(f"경고: Firestore에 메타데이터 저장 실패: {e}")

# --- 메인 클라우드 함수 ---
@functions_framework.http
def generateAvatarVideo(request):
    """아바타 비디오 생성 프로세스를 조정합니다."""
    
    # --- 초기 설정 및 매개변수 유효성 검사 ---
    REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN")
    if not REPLICATE_API_TOKEN:
        print("오류: REPLICATE_API_TOKEN 환경 변수가 설정되지 않았습니다.")
        return "구성 오류: Replicate API 토큰을 찾을 수 없습니다.", 500

    request_json = request.get_json(silent=True)
    if not request_json:
        return "잘못된 요청: JSON 페이로드가 없습니다.", 400
    
    script_text = request_json.get('script_text')
    avatar_id = request_json.get('avatar_id')
    user_id = request_json.get('user_id', 'test_user') # MVP 기본값

    if not script_text:
        return "잘못된 요청: 'script_text'가 비어 있거나 누락되었습니다.", 400
    if not avatar_id:
        return "잘못된 요청: 'avatar_id'가 비어 있거나 누락되었습니다.", 400

    # --- API 클라이언트 초기화 ---
    try:
        tts_client = texttospeech.TextToSpeechClient()
        replicate_client = replicate.Client(api_token=REPLICATE_API_TOKEN)
    except Exception as e:
        print(f"API 클라이언트 초기화 중 오류 발생: {e}")
        return "필수 API 클라이언트를 초기화하지 못했습니다.", 500

    # --- 임시 파일 경로 정의 ---
    # 고유한 이름은 여러 함수 인스턴스가 실행될 경우 충돌을 방지하는 데 도움이 됩니다.
    execution_id = request.headers.get('Function-Execution-Id', datetime.now().strftime('%Y%m%d%H%M%S%f'))
    temp_avatar_path = f"/tmp/avatar_{execution_id}.png"
    temp_audio_path = f"/tmp/output_{execution_id}.mp3"
    temp_video_path = f"/tmp/final_video_{execution_id}.mp4"
    
    final_video_storage_url = None

    try:
        # --- 1단계: 아바타 이미지 다운로드 ---
        avatar_storage_path = f"avatars/default/{avatar_id}" # .png로 가정, 필요한 경우 조정
        download_avatar_image(bucket, avatar_storage_path, temp_avatar_path)

        # --- 2단계: TTS 오디오 생성 ---
        generate_tts_audio(script_text, tts_client, temp_audio_path)

        # --- 3단계: 립싱크 수행 (Replicate 경유) ---
        replicate_video_url = perform_lip_sync(replicate_client, temp_avatar_path, temp_audio_path)
        
        # --- 4단계: Replicate에서 비디오 다운로드 ---
        download_replicate_video(replicate_video_url, temp_video_path)

        # --- 5단계: 최종 비디오를 Firebase Storage에 업로드 ---
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_avatar_id = avatar_id.split('.')[0] # 확장자가 있는 경우 제거
        storage_video_filename = f"{timestamp}_{base_avatar_id}.mp4"
        final_storage_path = f"generated_videos/{user_id}/{storage_video_filename}"
        
        final_video_storage_url = upload_to_firebase_storage(bucket, temp_video_path, final_storage_path)
        if not final_video_storage_url: # 예외로 처리되어야 하지만 안전 장치로 사용합니다.
            raise OperationFailure("업로드 후 비디오 URL을 가져오는 데 실패했지만 예외는 발생하지 않았습니다.", 500)

        # --- 6단계: Firestore에 메타데이터 저장 ---
        save_metadata_to_firestore(db, user_id, final_video_storage_url, script_text, avatar_id)

        print(f"비디오를 성공적으로 생성했습니다: {final_video_storage_url}")
        return {"message": "비디오가 성공적으로 생성되었습니다!", "video_url": final_video_storage_url}, 200

    except OperationFailure as e:
        print(f"작업 실패: {e}")
        return str(e), e.status_code
    except Exception as e:
        # 프로세스 중 예기치 않은 오류를 모두 포착합니다.
        print(f"예기치 않은 오류 발생: {e}")
        return "예기치 않은 서버 오류가 발생했습니다.", 500
    finally:
        # --- 임시 파일 정리 ---
        for temp_file_path in [temp_avatar_path, temp_audio_path, temp_video_path]:
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                    print(f"임시 파일 정리: {temp_file_path}")
                except Exception as e:
                    print(f"임시 파일 {temp_file_path} 정리 중 오류 발생: {e}")
