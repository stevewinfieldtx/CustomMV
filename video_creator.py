import os
import tempfile
import requests
import logging
import json
import librosa
import itertools
from google.cloud import storage
from PIL import Image
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
# Note: No more dotenv for cloud functions, use environment variables directly

# --- Configure logging for Cloud Functions ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Get environment variables ---
RUNWARE_API_KEY = os.getenv('RUNWARE_API_KEY')
RUNWARE_API_URL = os.getenv('RUNWARE_API_URL', 'https://api.runware.ai/v1/generate')
GCS_BUCKET_NAME = os.getenv('GCS_BUCKET_NAME')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# --- Helper Functions (moved from music_creator) ---

def get_image_prompts_from_gemini(vision: str, mood: str, age: str, num_prompts: int) -> list[str]:
    if not GEMINI_API_KEY: raise Exception('Missing GEMINI_API_KEY in Cloud Function environment.')
    prompt_lines = [
        "You are an expert AI art prompt engineer.", "Your task is to generate a list of unique, vivid, and creative prompts for an AI image generator.",
        "Each prompt should be a self-contained, detailed scene.", f"The overarching theme is: '{vision}'.", f"The desired mood is: '{mood}'.",
        f"The target audience is: '{age}'.", f"Generate exactly {num_prompts} unique prompts.",
        "Return the prompts as a JSON-formatted list of strings. Do not include any other text or markdown.",
        "Example format: [\"prompt 1 here\", \"prompt 2 here\"]"
    ]
    prompt = "\n".join(prompt_lines)
    url = f'https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent?key={GEMINI_API_KEY}'
    resp = requests.post(url, json={'contents': [{'parts': [{'text': prompt}]}]}, headers={'Content-Type': 'application/json'})
    resp.raise_for_status()
    cleaned_text = resp.json()['candidates'][0]['content']['parts'][0]['text'].strip().replace('```json', '').replace('```', '').strip()
    return json.loads(cleaned_text)

# --- Video Creation Functions ---

def download_from_gcs(source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    logger.info(f"Downloaded {source_blob_name} to {destination_file_name}.")
    return destination_file_name

def upload_to_gcs(local_path, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_path)
    blob.make_public()
    logger.info(f"Uploaded {local_path} to {destination_blob_name}. Public URL: {blob.public_url}")
    return blob.public_url

def generate_images(prompts: list[str], num_images: int):
    # This function remains largely the same
    logger.info(f"Generating {num_images} images from {len(prompts)} unique prompts.")
    if not RUNWARE_API_KEY: raise Exception("RUNWARE_API_KEY is not set.")
    headers = {'Authorization': f'Bearer {RUNWARE_API_KEY}', 'Content-Type': 'application/json'}
    image_urls = []
    prompt_cycle = itertools.cycle(prompts)
    for i in range(num_images):
        current_prompt = next(prompt_cycle)
        logger.info(f"Requesting image {i+1}/{num_images}...")
        payload = [{"taskType": "imageInference", "model": "runware:100@1", "positivePrompt": current_prompt, "numberResults": 1, "outputFormat": "JPEG", "width": 896, "height": 1152, "steps": 12, "CFGScale": 1, "scheduler": "DPM++ 3M", "checkNSFW": True, "lora": [{"model": "civitai:982309@1100321", "weight": 1}]}]
        resp = requests.post(RUNWARE_API_URL, headers=headers, json=payload)
        resp.raise_for_status()
        if resp.json().get('images'): image_urls.extend(resp.json()['images'])
    
    paths = []
    for i, url in enumerate(image_urls):
        r = requests.get(url)
        r.raise_for_status()
        fd, img_path = tempfile.mkstemp(suffix='.jpeg')
        os.close(fd)
        with open(img_path, 'wb') as f: f.write(r.content)
        paths.append(img_path)
    return paths

def assemble_video(audio_path, image_paths):
    # This function remains the same
    duration = librosa.get_duration(path=audio_path)
    fps = len(image_paths) / duration if duration > 0 else 24
    clip = ImageSequenceClip(image_paths, fps=fps)
    audio_clip = AudioFileClip(audio_path)
    final = clip.set_audio(audio_clip).set_duration(audio_clip.duration)
    fd, tmp_video = tempfile.mkstemp(suffix='.mp4')
    os.close(fd)
    final.write_videofile(tmp_video, codec='libx264', audio_codec='aac', logger='bar')
    return tmp_video

# --- Main Cloud Function Entry Point ---

def main(event, context):
    """
    Triggered by a file upload to a GCS bucket.
    Args:
         event (dict): Event payload.
         context (google.cloud.functions.Context): Metadata for the event.
    """
    file_name = event['name']
    if not file_name.endswith('.json') or not file_name.startswith('pending/'):
        logger.info(f"Ignoring file {file_name} as it is not a new job file.")
        return

    logger.info(f"--- Processing new job file: {file_name} ---")
    
    local_job_path = os.path.join(tempfile.gettempdir(), os.path.basename(file_name))
    local_audio_path = None
    images = []
    final_video_path = None

    try:
        # 1. Download and parse the job file
        download_from_gcs(file_name, local_job_path)
        with open(local_job_path) as f:
            job_data = json.load(f)
        
        req = job_data['original_request']
        task_id = job_data['task_id']
        audio_gcs_path = job_data['audio_gcs_path']
        local_audio_path = os.path.join(tempfile.gettempdir(), f"{task_id}.mp3")
        download_from_gcs(audio_gcs_path, local_audio_path)

        # 2. Analyze audio to determine image/prompt count
        y, sr = librosa.load(local_audio_path)
        _, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        num_images = max(8, len(beat_frames) // 2)
        num_prompts = num_images // 2
        logger.info(f"Job requires {num_prompts} prompts for {num_images} images.")

        # 3. Generate prompts from Gemini
        image_prompts = get_image_prompts_from_gemini(
            vision=req.get('vision', ''), mood=req.get('mood', ''),
            age=req.get('age', ''), num_prompts=num_prompts
        )

        # 4. Generate images
        images = generate_images(image_prompts, num_images)
        
        # 5. Assemble video
        final_video_path = assemble_video(local_audio_path, images)
        
        # 6. Upload final video to 'complete' folder
        upload_to_gcs(final_video_path, f"complete/{task_id}.mp4")

        # 7. Clean up source files from 'pending' folder
        storage_client = storage.Client()
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        bucket.blob(file_name).delete()
        bucket.blob(audio_gcs_path).delete()
        logger.info(f"--- Successfully completed job {task_id} ---")

    except Exception as e:
        logger.error(f"!!! FAILED to process job {file_name}: {e} !!!", exc_info=True)
        # Optionally, move the files to a 'failed' folder instead of deleting
    finally:
        # Clean up all temporary local files
        cleanup_files = [local_job_path, local_audio_path, final_video_path] + images
        for p in cleanup_files:
            if p and os.path.exists(p):
                os.remove(p)
