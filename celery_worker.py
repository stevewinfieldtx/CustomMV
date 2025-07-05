#celery_worker.py
import os
import tempfile
import requests
import logging
import json
import librosa
import itertools
import uuid
from celery import Celery
from celery.exceptions import Retry
from PIL import Image
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from moviepy.audio.io.AudioFileClip import AudioFileClip

# --- Configure logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Environment variables ---
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
RUNWARE_API_KEY = os.getenv("RUNWARE_API_KEY")
RUNWARE_API_URL = os.getenv("RUNWARE_API_URL", "https://api.runware.ai/v1")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
APIBOX_KEY = os.getenv("APIBOX_KEY")

# --- Initialize Celery ---
celery_app = Celery("tasks", broker=REDIS_URL)

# --- Helpers for duration buckets ---
def categorize_length(seconds: float):
    choices = [30, 60, 120, 180, 240]
    target = next((c for c in choices if seconds <= c), 240)
    label_map = {30: "XS", 60: "S", 120: "M", 180: "L", 240: "XL"}
    return label_map[target], target


def trim_to_bucket(input_path: str, target_secs: int) -> str:
    try:
        from pydub import AudioSegment
    except ImportError:
        raise Exception("Audio trimming requires pydub; please install it.")
    song = AudioSegment.from_file(input_path)
    trimmed = song[: target_secs * 1000]
    out_path = input_path.replace(".mp3", f"_{target_secs}s.mp3")
    trimmed.export(out_path, format="mp3")
    return out_path


def download_audio(url: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    resp = requests.get(url)
    resp.raise_for_status()
    with open(path, "wb") as f:
        f.write(resp.content)
    return path


def upload_to_gcs(local_path: str, destination_blob_name: str) -> str:
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_path)
    blob.make_public()
    return blob.public_url


def get_image_prompts_from_gemini(vision, mood, age, num_prompts, duration_label=None, duration_secs=None):
    if not GEMINI_API_KEY:
        raise Exception("Missing GEMINI_API_KEY")
    lines = [
        f"Generate {num_prompts} unique, vivid AI art prompts based on the theme '{vision}',",
        f"with a '{mood}' mood for a '{age}' audience."
    ]
    if duration_label and duration_secs:
        lines.append(f"Duration category: {duration_label} ({duration_secs} sec)")
    lines.append("Return as a JSON list of strings.")
    prompt = ' '.join(lines)
    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    resp = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, headers={"Content-Type": "application/json"})
    resp.raise_for_status()
    cleaned = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    return json.loads(cleaned)


def generate_images(prompts, num_images):
    if not RUNWARE_API_KEY:
        raise Exception("Missing RUNWARE_API_KEY")
    headers = {"Authorization": f"Bearer {RUNWARE_API_KEY}", "Content-Type": "application/json"}
    image_urls = []
    for idx in range(num_images):
        current = prompts[idx % len(prompts)]
        logger.info(f"Requesting image {idx+1}/{num_images}...")
        payload = [{
            "taskType": "imageInference",
            "taskUUID": str(uuid.uuid4()),
            "positivePrompt": current,
            "model": "runware:100@1",
            "width": 896,
            "height": 1152,
            "steps": 12,
            "scheduler": "DPM++ 3M",
            "numberResults": 1,
            "outputType": "URL",
            "checkNSFW": True
        }]
        resp = requests.post(RUNWARE_API_URL, headers=headers, json=payload)
        resp.raise_for_status()
        for entry in resp.json().get("data", []):
            url = entry.get("imageURL") or entry.get("imageUrl")
            if url:
                image_urls.append(url)
    if not image_urls:
        raise Exception("No images returned from Runware API.")
    paths = []
    for url in image_urls:
        r = requests.get(url)
        r.raise_for_status()
        fd, p = tempfile.mkstemp(suffix=".jpeg")
        os.close(fd)
        with open(p, "wb") as f:
            f.write(r.content)
        paths.append(p)
    return paths


def assemble_video(audio_path, image_paths):
    duration = librosa.get_duration(path=audio_path)
    fps = len(image_paths) / duration if duration > 0 else 24
    clip = ImageSequenceClip(image_paths, fps=fps)
    audio_clip = AudioFileClip(audio_path)
    final = clip.set_audio(audio_clip).set_duration(audio_clip.duration)
    fd, out = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    final.write_videofile(out, codec="libx264", audio_codec="aac")
    return out


@celery_app.task(bind=True)
def poll_music_status(self, task_id, original_request):
    status_url = "https://apibox.erweima.ai/api/v1/status"
    try:
        resp = requests.get(status_url, params={"taskId": task_id}, headers={"Authorization": f"Bearer {APIBOX_KEY}"})
        resp.raise_for_status()
        body = resp.json()
        if body.get("status") != "complete":
            raise self.retry(countdown=15, max_retries=20)
        audio_list = body.get("data", [])
        audio_url = next((d.get("audio_url") for d in audio_list if d.get("audio_url")), None)
        if not audio_url:
            raise Exception(f"No audio_url in status response: {body}")
        # Enqueue video creation
        create_video_task.delay(task_id, audio_url, original_request)
    except Retry:
        raise
    except Exception as e:
        logger.error(f"Polling error for {task_id}: {e}")
        raise self.retry(countdown=15, max_retries=20)


@celery_app.task
def create_video_task(task_id, audio_url, original_request):
    logger.info(f"--- [CELERY WORKER] Starting video task {task_id} ---")
    local_audio = None
    images = []
    final_video = None
    try:
        local_audio = download_audio(audio_url)
        actual = librosa.get_duration(path=local_audio)
        label, secs = categorize_length(actual)
        local_audio = trim_to_bucket(local_audio, secs)
        logger.info(f"Trimmed to {label} ({secs}s)")
        # Upload trimmed audio
        blob_audio = f"audio/{task_id}_{secs}s.mp3"
        url_audio = upload_to_gcs(local_audio, blob_audio)
        logger.info(f"Uploaded trimmed audio: {url_audio}")
        # Image prompts
        y, sr = librosa.load(local_audio)
        beats = librosa.beat.beat_track(y=y, sr=sr)[1]
        num = max(8, len(beats)//2)
        prompts = get_image_prompts_from_gemini(
            original_request.get('vision',''),
            original_request.get('mood',''),
            original_request.get('age',''),
            num//2,
            duration_label=label,
            duration_secs=secs
        )
        images = generate_images(prompts, num)
        final_video = assemble_video(local_audio, images)
        blob_vid = f"complete/{task_id}.mp4"
        url_vid = upload_to_gcs(final_video, blob_vid)
        logger.info(f"Uploaded video: {url_vid}")
        logger.info(f"--- [CELERY WORKER] Completed {task_id} ---")
    except Exception as e:
        logger.error(f"create_video_task error {task_id}: {e}", exc_info=True)
    finally:
        if local_audio and os.path.exists(local_audio): os.remove(local_audio)
        for p in images + ([final_video] if final_video else []):
            if p and os.path.exists(p): os.remove(p)
