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

# --- Initialize Celery ---
celery_app = Celery("tasks", broker=REDIS_URL)

# --- Duration Bucket Helpers ---
def categorize_length(seconds: float):
    """Map any duration to XS/S/M/L/XL buckets."""
    choices = [30, 60, 120, 180, 240]
    target = next((c for c in choices if seconds <= c), 240)
    label_map = {30: "XS", 60: "S", 120: "M", 180: "L", 240: "XL"}
    return label_map[target], target


def trim_to_bucket(input_path: str, target_secs: int) -> str:
    """Trim an audio file to the first target_secs seconds."""
    try:
        from pydub import AudioSegment
    except ImportError:
        raise Exception("Audio trimming requires pydub; please install it.")
    song = AudioSegment.from_file(input_path)
    trimmed = song[: target_secs * 1000]
    out = input_path.replace(".mp3", f"_{target_secs}s.mp3")
    trimmed.export(out, format="mp3")
    return out


def upload_to_gcs(local_path: str, destination_blob_name: str) -> str:
    """Upload a local file to GCS and return its public URL."""
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_path)
    blob.make_public()
    return blob.public_url


# --- Helper Functions ---
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
    cleaned_text = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    return json.loads(cleaned_text)


def download_audio(url):
    fd, path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    resp = requests.get(url)
    resp.raise_for_status()
    with open(path, "wb") as f:
        f.write(resp.content)
    return path


def generate_images(prompts, num_images):
    if not RUNWARE_API_KEY:
        raise Exception("RUNWARE_API_KEY is not set.")
    headers = {"Authorization": f"Bearer {RUNWARE_API_KEY}", "Content-Type": "application/json"}
    image_urls = []
    prompt_cycle = itertools.cycle(prompts)
    for i in range(num_images):
        current_prompt = next(prompt_cycle)
        logger.info(f"Requesting image {i+1}/{num_images}...")
        payload = [{
            "taskType": "imageInference",
            "taskUUID": str(uuid.uuid4()),
            "positivePrompt": current_prompt,
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
        logger.info(f"Runware API Raw Response: {resp.text}")
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
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    final.write_videofile(path, codec="libx264", audio_codec="aac")
    return path


@celery_app.task
def create_video_task(task_id, audio_url, original_request):
    logger.info(f"--- [CELERY WORKER] Starting video task {task_id} ---")
    local_audio_path = None
    images = []
    final_video_path = None
    try:
        # Download and trim audio to bucket
        local_audio_path = download_audio(audio_url)
        actual_duration = librosa.get_duration(path=local_audio_path)
        bucket_label, bucket_secs = categorize_length(actual_duration)
        local_audio_path = trim_to_bucket(local_audio_path, bucket_secs)
        logger.info(f"Trimmed audio to bucket {bucket_label} ({bucket_secs}s)")

        # Upload trimmed audio to GCS
        gcs_audio_path = f"audio/{task_id}_{bucket_secs}s.mp3"
        audio_gcs_url = upload_to_gcs(local_audio_path, gcs_audio_path)
        logger.info(f"Uploaded trimmed audio to GCS: {audio_gcs_url}")

        # Beat-based image count
        y, sr = librosa.load(local_audio_path)
        _, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        num_images = max(8, len(beat_frames) // 2)
        num_prompts = num_images // 2

        # Generate Gemini-driven prompts
        image_prompts = get_image_prompts_from_gemini(
            vision=original_request.get("vision", ""),
            mood=original_request.get("mood", ""),
            age=original_request.get("age", ""),
            num_prompts=num_prompts,
            duration_label=bucket_label,
            duration_secs=bucket_secs,
        )

        # Image generation + video assembly
        images = generate_images(image_prompts, num_images)
        final_video_path = assemble_video(local_audio_path, images)

        # Upload final video to GCS
        video_gcs_path = f"complete/{task_id}.mp4"
        video_gcs_url = upload_to_gcs(final_video_path, video_gcs_path)
        logger.info(f"Uploaded final video to GCS: {video_gcs_url}")
        logger.info(f"--- [CELERY WORKER] Successfully completed task {task_id} ---")

    except Exception as e:
        logger.error(f"!!! [CELERY WORKER] FAILED task {task_id}: {e} !!!", exc_info=True)
    finally:
        # Rename and cleanup local files
        if local_audio_path and os.path.exists(local_audio_path):
            done_path = local_audio_path.replace(".mp3", "_DONE.mp3")
            os.rename(local_audio_path, done_path)
            logger.info(f"Audio file renamed to: {done_path}")
        for p in ([final_video_path] if final_video_path else []) + images:
            if p and os.path.exists(p):
                os.remove(p)
