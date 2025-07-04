import os
import tempfile
import requests
import logging
import json
import librosa
import itertools
import uuid
from celery import Celery
from google.cloud import storage
from PIL import Image
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from moviepy.audio.io.AudioFileClip import AudioFileClip # KEEP THIS ONE

# --- Configure logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Get environment variables ---
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
RUNWARE_API_KEY = os.getenv("RUNWARE_API_KEY")
RUNWARE_API_URL = os.getenv(
    "RUNWARE_API_URL", "https://api.runware.ai/v1"
)
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Initialize Celery ---
celery_app = Celery("tasks", broker=REDIS_URL)

# --- Helper Functions ---

def get_image_prompts_from_gemini(vision, mood, age, num_prompts):
    if not GEMINI_API_KEY:
        raise Exception("Missing GEMINI_API_KEY")
    prompt = f"Generate {num_prompts} unique, vivid AI art prompts based on the theme '{vision}', with a '{mood}' mood for a '{age}' audience. Return as a JSON list of strings."
    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    resp = requests.post(
        url,
        json={"contents": [{"parts": [{"text": prompt}]}]},
        headers={"Content-Type": "application/json"},
    )
    resp.raise_for_status()
    cleaned_text = (
        resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        .strip()
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )
    return json.loads(cleaned_text)

def download_audio(url):
    resp = requests.get(url)
    resp.raise_for_status()
    fd, path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    with open(path, "wb") as f:
        f.write(resp.content)
    return path

def generate_images(prompts, num_images):
    if not RUNWARE_API_KEY:
        raise Exception("RUNWARE_API_KEY is not set.")
    headers = {
        "Authorization": f"Bearer {RUNWARE_API_KEY}",
        "Content-Type": "application/json",
    }
    image_urls = []
    prompt_cycle = itertools.cycle(prompts)
    for i in range(num_images):
        current_prompt = next(prompt_cycle)
        logger.info(f"Requesting image {i+1}/{num_images} from Runware AI...")
        payload = [{
            "taskType":      "imageInference",
            "taskUUID":      str(uuid.uuid4()),
            "positivePrompt": current_prompt,
            "model":         "runware:100@1",
            "width":         896,
            "height":        1152,
            "steps": 12,
            "scheduler": "DPM++ 3M",        
            "numberResults": 1,
            "outputType":    "URL",
            "checkNSFW": True
        }]
        
        resp = requests.post(RUNWARE_API_URL, headers=headers, json=payload)
        logger.info(f"Runware API Raw Response for image {i+1}: {resp.text}")
        resp.raise_for_status()
        if resp.json().get("images"):
            image_urls.extend(resp.json()["images"])

    paths = []
    for i, url in enumerate(image_urls):
        try:
            logger.info(f"Downloading image {i+1}/{len(image_urls)} from URL: {url}")
            r = requests.get(url, stream=True)
            r.raise_for_status()
            
            fd, path = tempfile.mkstemp(suffix=".jpeg")
            os.close(fd)
            with open(path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            paths.append(path)
            logger.info(f"Successfully downloaded image {i+1} to {path}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading image {i+1} from {url}: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error during image {i+1} processing from {url}: {e}", exc_info=True)
    
    logger.info(f"Finished image downloading. Total successfully downloaded images: {len(paths)}")
    return paths

def assemble_video(audio_path, image_paths):
    duration = librosa.get_duration(path=audio_path)
    fps = len(image_paths) / duration if duration > 0 and len(image_paths) > 0 else 24
    
    if not image_paths:
        logger.error("No images available for video assembly. Creating a placeholder image.")
        img_placeholder = Image.new('RGB', (896, 1152), color = 'black')
        fd, placeholder_path = tempfile.mkstemp(suffix=".jpeg")
        os.close(fd)
        img_placeholder.save(placeholder_path)
        image_paths = [placeholder_path]
        fps = 1
        
    clip = ImageSequenceClip(image_paths, fps=fps)
    audio_clip = AudioFileClip(audio_path) # Changed back to AudioFileClip
    final = clip.set_audio(audio_clip).set_duration(audio_clip.duration)
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    final.write_videofile(path, codec="libx264", audio_codec="aac")
    return path

def upload_to_gcs(local_path, destination_blob_name):
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_path)
    blob.make_public()
    return blob.public_url

@celery_app.task
def create_video_task(task_id, audio_url, original_request):
    logger.info(f"--- [CELERY WORKER] Starting video task {task_id} ---")
    local_audio_path = None
    images = []
    final_video_path = None
    truncated_audio_path = None
    try:
        local_audio_path = download_audio(audio_url)
        y, sr = librosa.load(local_audio_path)
        
        desired_length_tag = original_request.get('length', 'medium').lower()
        desired_duration_sec = 60
        if 'short' in desired_length_tag:
            desired_duration_sec = 30
        elif 'long' in desired_length_tag:
            desired_duration_sec = 90
        
        num_images = max(8, int(desired_duration_sec * 0.5))
        num_prompts = num_images // 2

        logger.info(f"Task {task_id}: Desired video length: {desired_duration_sec}s, Num images to generate: {num_images}")

        full_audio_clip = AudioFileClip(local_audio_path) # Changed back to AudioFileClip
        final_audio_duration = min(desired_duration_sec, full_audio_clip.duration)
        truncated_audio_clip = full_audio_clip.subclip(0, final_audio_duration) # This line uses subclip
        
        fd_trunc, truncated_audio_path = tempfile.mkstemp(suffix=".mp3")
        os.close(fd_trunc)
        truncated_audio_clip.write_audiofile(truncated_audio_path, codec="aac")
        truncated_audio_clip.close()

        image_prompts = get_image_prompts_from_gemini(
            vision=original_request.get('vision', ''),
            mood=original_request.get('mood', ''),
            age=original_request.get('age', ''),
            num_prompts=num_prompts
        )

        images = generate_images(image_prompts, num_images)
        final_video_path = assemble_video(truncated_audio_path, images)
        
        upload_to_gcs(final_video_path, f"complete/{task_id}.mp4")
        logger.info(f"--- [CELERY WORKER] Successfully completed task {task_id} ---")

    except Exception as e:
        logger.error(f"!!! [CELERY WORKER] FAILED task {task_id}: {e} !!!", exc_info=True)
    finally:
        if local_audio_path and os.path.exists(local_audio_path):
            new_audio_path = local_audio_path.replace('.mp3', '_DONE.mp3')
            os.rename(local_audio_path, new_audio_path)
            logger.info(f"Original audio file renamed to: {new_audio_path}")

        cleanup_files = [truncated_audio_path, final_video_path] + images
        for p in cleanup_files:
            if p and os.path.exists(p): os.remove(p)
