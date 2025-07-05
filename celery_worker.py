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
from google.cloud import storage
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
REDIS_URL        = os.getenv("REDIS_URL", "redis://localhost:6379/0")
RUNWARE_API_KEY  = os.getenv("RUNWARE_API_KEY")
RUNWARE_API_URL  = os.getenv("RUNWARE_API_URL", "https://api.runware.ai/v1")
GCS_BUCKET_NAME  = os.getenv("GCS_BUCKET_NAME")
GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY")
APIBOX_KEY       = os.getenv("APIBOX_KEY")

# --- Initialize Celery ---
celery_app = Celery("tasks", broker=REDIS_URL)

def categorize_length(seconds: float):
    choices = [30, 60, 120, 180, 240]
    target = next((c for c in choices if seconds <= c), 240)
    labels = {30: "XS", 60: "S", 120: "M", 180: "L", 240: "XL"}
    return labels[target], target

def trim_to_bucket(input_path: str, target_secs: int) -> str:
    try:
        from pydub import AudioSegment
    except ImportError:
        raise Exception("pydub not installed; cannot trim audio")
    song = AudioSegment.from_file(input_path)
    trimmed = song[: target_secs * 1000]
    out = input_path.replace(".mp3", f"_{target_secs}s.mp3")
    trimmed.export(out, format="mp3")
    return out

def download_audio(url: str) -> str:
    resp = requests.get(url)
    resp.raise_for_status()
    fd, path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    with open(path, "wb") as f:
        f.write(resp.content)
    return path

def upload_to_gcs(local_path, destination_blob_name):
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_path)
    blob.make_public()
    return blob.public_url

def get_image_prompts_from_gemini(vision, mood, age, num_prompts):
    if not GEMINI_API_KEY:
        raise Exception("Missing GEMINI_API_KEY")
    prompt = (
        f"Generate {num_prompts} unique, vivid AI art prompts based on the theme "
        f"'{vision}', with a '{mood}' mood for a '{age}' audience. Return as a JSON list."
    )
    url = (
        f"https://generativelanguage.googleapis.com/v1/models/"
        f"gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    )
    resp = requests.post(
        url,
        json={"contents": [{"parts": [{"text": prompt}]}]},
        headers={"Content-Type": "application/json"},
    )
    resp.raise_for_status()
    text = (
        resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        .strip()
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )
    return json.loads(text)

def generate_images(prompts, num_images):
    if not RUNWARE_API_KEY:
        raise Exception("Missing RUNWARE_API_KEY")
    headers = {
        "Authorization": f"Bearer {RUNWARE_API_KEY}",
        "Content-Type": "application/json",
    }
    image_urls = []
    cycle = itertools.cycle(prompts)
    for i in range(num_images):
        p = next(cycle)
        logger.info(f"Requesting image {i+1}/{num_images}...")
        payload = [{
            "taskType": "imageInference",
            "taskUUID": str(uuid.uuid4()),
            "positivePrompt": p,
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
        for item in resp.json().get("data", []):
            url = item.get("imageURL") or item.get("imageUrl")
            if url:
                image_urls.append(url)
    paths = []
    for url in image_urls:
        r = requests.get(url)
        r.raise_for_status()
        fd, path = tempfile.mkstemp(suffix=".jpeg")
        os.close(fd)
        with open(path, "wb") as f:
            f.write(r.content)
        paths.append(path)
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
    local_audio_path, images, final_video_path = None, [], None
    try:
        # 1) Download & trim to bucket
        local_audio_path = download_audio(audio_url)
        secs = librosa.get_duration(path=local_audio_path)
        label, bucket_secs = categorize_length(secs)
        local_audio_path = trim_to_bucket(local_audio_path, bucket_secs)
        logger.info(f"Trimmed audio to {label} ({bucket_secs}s)")

        # 2) Upload trimmed MP3
        gcs_audio = f"audio/{task_id}_{bucket_secs}s.mp3"
        audio_url_gcs = upload_to_gcs(local_audio_path, gcs_audio)
        logger.info(f"Uploaded trimmed audio to GCS: {audio_url_gcs}")

        # 3) Beat analysis
        y, sr = librosa.load(local_audio_path)
        _, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        num_images = max(8, len(beat_frames) // 2)
        num_prompts = num_images // 2

        # 4) Gemini for image prompts
        prompts = get_image_prompts_from_gemini(
            vision=original_request.get("vision",""),
            mood=original_request.get("mood",""),
            age=original_request.get("age",""),
            num_prompts=num_prompts
        )

        # 5) Runware image generation
        images = generate_images(prompts, num_images)

        # 6) Assemble and upload final video
        final_video_path = assemble_video(local_audio_path, images)
        video_blob = f"complete/{task_id}.mp4"
        video_url = upload_to_gcs(final_video_path, video_blob)
        logger.info(f"--- Completed task {task_id}: {video_url} ---")

    except Exception as e:
        logger.error(f"!!! Task {task_id} FAILED: {e}", exc_info=True)
    finally:
        # Cleanup
        if local_audio_path and os.path.exists(local_audio_path):
            done = local_audio_path.replace(".mp3","_DONE.mp3")
            os.rename(local_audio_path, done)
        for p in ([final_video_path] + images):
            if p and os.path.exists(p):
                os.remove(p)

@celery_app.task
def poll_music_status(task_id, original_request):
    """
    Poll Apibox until music is ready, then trigger create_video_task.
    """
    if not APIBOX_KEY:
        raise Exception("Missing APIBOX_KEY")
    headers = {'Authorization': f'Bearer {APIBOX_KEY}'}
    status_url = f"https://apibox.erweima.ai/api/v1/generate/{task_id}"
    resp = requests.get(status_url, headers=headers)
    resp.raise_for_status()
    data = resp.json().get('data', {})
    state = data.get('state') or data.get('status')
    if state in ('complete','finished','success'):
        audio_url = data.get('audio_url') or data.get('result',{}).get('audioUrl')
        if not audio_url:
            raise Exception(f"No audio_url in completion response: {resp.text}")
        create_video_task.delay(task_id, audio_url, original_request)
    elif state in ('pending','running','processing'):
        # try again in 15s
        poll_music_status.apply_async((task_id, original_request), countdown=15)
    else:
        raise Exception(f"Music generation failed: {resp.text}")
