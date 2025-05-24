import firebase_admin
from firebase_admin import credentials, firestore, storage
import functions_framework
from google.cloud import texttospeech, exceptions as google_cloud_exceptions
import os
import replicate
import requests 
from datetime import datetime

# --- Global Initializations ---
try:
    firebase_admin.get_app()
except ValueError:
    firebase_admin.initialize_app()

bucket = storage.bucket()  # Default Firebase Storage bucket
db = firestore.client()    # Firestore client, can be initialized globally or per function

# --- Helper Function Definitions ---

class OperationFailure(Exception):
    """Custom exception for operations that fail and should result in an HTTP error."""
    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code

def download_avatar_image(bucket_client, avatar_storage_path, temp_avatar_path):
    """Downloads avatar image from Firebase Storage."""
    try:
        blob = bucket_client.blob(avatar_storage_path)
        blob.download_to_filename(temp_avatar_path)
        print(f"Avatar image '{avatar_storage_path}' downloaded to '{temp_avatar_path}'")
        return True
    except google_cloud_exceptions.NotFound:
        print(f"Error: Avatar image not found at '{avatar_storage_path}'.")
        raise OperationFailure(f"Avatar image not found at '{avatar_storage_path}'.", 404)
    except Exception as e:
        print(f"Error: Failed to download avatar '{avatar_storage_path}'. Details: {e}")
        raise OperationFailure(f"Failed to download avatar image. Server error: {e}", 500)

def generate_tts_audio(script_text, tts_client, temp_audio_path):
    """Generates TTS audio and saves it to a temporary path."""
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
        print(f'Audio content written to file "{temp_audio_path}"')
        return True
    except Exception as e:
        print(f"Error during TTS generation: {e}")
        raise OperationFailure(f"TTS generation failed: {e}", 500)

def perform_lip_sync(replicate_client, temp_avatar_path, temp_audio_path):
    """Calls Replicate API for lip-sync and returns the video URL."""
    print(f"Starting lip-sync process with avatar '{temp_avatar_path}' and audio '{temp_audio_path}'")
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
        if not replicate_video_url: # Should not happen if API call is successful
             raise OperationFailure("Lip-sync process did not return a video URL from Replicate.", 500)
        print(f"Replicate generated video URL: {replicate_video_url}")
        return replicate_video_url
    except replicate.exceptions.ReplicateError as e:
        print(f"Replicate API error: {e}")
        raise OperationFailure(f"Lip-sync generation failed due to Replicate API error: {e}", 500)
    except Exception as e: # Catch other potential errors like file I/O
        print(f"Error during lip-sync Replicate call: {e}")
        raise OperationFailure(f"Lip-sync process failed: {e}", 500)

def download_replicate_video(video_url, temp_video_path):
    """Downloads video from Replicate URL to a temporary path."""
    try:
        video_response = requests.get(video_url, stream=True)
        video_response.raise_for_status()  # Raises HTTPError for bad responses (4XX or 5XX)
        with open(temp_video_path, "wb") as f:
            for chunk in video_response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Generated video downloaded from Replicate to {temp_video_path}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Failed to download video from Replicate URL '{video_url}': {e}")
        raise OperationFailure(f"Failed to download generated video from Replicate: {e}", 500)

def upload_to_firebase_storage(bucket_client, temp_video_path, storage_path):
    """Uploads video to Firebase Storage and returns its public URL."""
    try:
        blob = bucket_client.blob(storage_path)
        blob.upload_from_filename(temp_video_path)
        blob.make_public()
        video_storage_url = blob.public_url
        print(f"Video uploaded to Firebase Storage: {storage_path}")
        print(f"Public URL: {video_storage_url}")
        return video_storage_url
    except Exception as e:
        print(f"Failed to upload video to Firebase Storage at '{storage_path}': {e}")
        raise OperationFailure(f"Failed to upload video to Firebase Storage: {e}", 500)

def save_metadata_to_firestore(db_client, user_id, video_url, script_text, avatar_id):
    """Saves video metadata to Firestore."""
    try:
        doc_ref_tuple = db_client.collection("video_creations").add({
            "userId": user_id,
            "videoUrl": video_url,
            "scriptText": script_text,
            "avatarId": avatar_id,
            "createdAt": firestore.SERVER_TIMESTAMP,
            "status": "completed"
        })
        print(f"Metadata saved to Firestore with document ID: {doc_ref_tuple[1].id}")
    except Exception as e:
        # For MVP, log warning but don't fail the entire operation
        print(f"Warning: Failed to save metadata to Firestore: {e}")

# --- Main Cloud Function ---
@functions_framework.http
def generateAvatarVideo(request):
    """Orchestrates the avatar video generation process."""
    
    # --- Initial Setup & Parameter Validation ---
    REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN")
    if not REPLICATE_API_TOKEN:
        print("Error: REPLICATE_API_TOKEN environment variable not set.")
        return "Configuration error: Replicate API token not found.", 500

    request_json = request.get_json(silent=True)
    if not request_json:
        return "Invalid request: No JSON payload.", 400
    
    script_text = request_json.get('script_text')
    avatar_id = request_json.get('avatar_id')
    user_id = request_json.get('user_id', 'test_user') # Default for MVP

    if not script_text:
        return "Invalid request: 'script_text' cannot be empty or is missing.", 400
    if not avatar_id:
        return "Invalid request: 'avatar_id' cannot be empty or is missing.", 400

    # --- Initialize API Clients ---
    try:
        tts_client = texttospeech.TextToSpeechClient()
        replicate_client = replicate.Client(api_token=REPLICATE_API_TOKEN)
    except Exception as e:
        print(f"Error initializing API clients: {e}")
        return "Failed to initialize necessary API clients.", 500

    # --- Define Temporary File Paths ---
    # Unique names help avoid collisions if multiple function instances run
    execution_id = request.headers.get('Function-Execution-Id', datetime.now().strftime('%Y%m%d%H%M%S%f'))
    temp_avatar_path = f"/tmp/avatar_{execution_id}.png"
    temp_audio_path = f"/tmp/output_{execution_id}.mp3"
    temp_video_path = f"/tmp/final_video_{execution_id}.mp4"
    
    final_video_storage_url = None

    try:
        # --- Step 1: Download Avatar Image ---
        avatar_storage_path = f"avatars/default/{avatar_id}" # Assuming .png, adjust if needed
        download_avatar_image(bucket, avatar_storage_path, temp_avatar_path)

        # --- Step 2: Generate TTS Audio ---
        generate_tts_audio(script_text, tts_client, temp_audio_path)

        # --- Step 3: Perform Lip Sync (via Replicate) ---
        replicate_video_url = perform_lip_sync(replicate_client, temp_avatar_path, temp_audio_path)
        
        # --- Step 4: Download Video from Replicate ---
        download_replicate_video(replicate_video_url, temp_video_path)

        # --- Step 5: Upload Final Video to Firebase Storage ---
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_avatar_id = avatar_id.split('.')[0] # Remove extension if any
        storage_video_filename = f"{timestamp}_{base_avatar_id}.mp4"
        final_storage_path = f"generated_videos/{user_id}/{storage_video_filename}"
        
        final_video_storage_url = upload_to_firebase_storage(bucket, temp_video_path, final_storage_path)
        if not final_video_storage_url: # Should be caught by exception, but as a safeguard
            raise OperationFailure("Failed to get video URL after upload, though no exception was raised.", 500)

        # --- Step 6: Save Metadata to Firestore ---
        save_metadata_to_firestore(db, user_id, final_video_storage_url, script_text, avatar_id)

        print(f"Successfully generated video: {final_video_storage_url}")
        return {"message": "Video generated successfully!", "video_url": final_video_storage_url}, 200

    except OperationFailure as e:
        print(f"Operation failed: {e}")
        return str(e), e.status_code
    except Exception as e:
        # Catch-all for unexpected errors during the process
        print(f"An unexpected error occurred: {e}")
        return "An unexpected server error occurred.", 500
    finally:
        # --- Cleanup Temporary Files ---
        for temp_file_path in [temp_avatar_path, temp_audio_path, temp_video_path]:
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                    print(f"Cleaned up temporary file: {temp_file_path}")
                except Exception as e:
                    print(f"Error cleaning up temporary file {temp_file_path}: {e}")
