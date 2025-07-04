
import os

import tempfile

import requests

import logging

import json

import librosa

import itertools

import uuid # Added: For UUID generation

from celery import Celery

from google.cloud import storage

from PIL import Image

from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from moviepy.audio.io.AudioFileClip import AudioFileClip



# --- Configure logging ---

logging.basicConfig(

    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"

)

logger = logging.getLogger(__name__)



# --- Get environment variables ---

# Celery will get these from the worker's environment

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

RUNWARE_API_KEY = os.getenv("RUNWARE_API_KEY")

# Corrected URL: Removed /generate as per user clarification

RUNWARE_API_URL = os.getenv("RUNWARE_API_URL", "https://api.runware.ai/v1") 

GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")



# --- Initialize Celery ---

celery_app = Celery("tasks", broker=REDIS_URL)



# --- Helper Functions ---



def get_image_prompts_from_gemini(vision, mood, age, num_prompts):

    if not GEMINI_API_KEY:

        raise Exception("Missing GEMINI_API_KEY")

    prompt = f"Generate {num_prompts} unique, vivid AI art prompts based on the theme '{vision}', with a '{mood}' mood for a '{age}' audience. Return as a JSON list of strings."

    # Corrected Gemini model name to gemini-2.5-flash

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

        logger.info(f"Requesting image {i+1}/{num_images}...")

        

        # Corrected payload:

        # - Removed "lora" (civitai model) as per debugging

        # - Removed "CFGScale"

        # - Corrected "positivePrompt" to use current_prompt

        # - Model reverted to "runware:100@1" as it was the previous base model

        payload = [{

            "taskType":      "imageInference",

            "taskUUID":      str(uuid.uuid4()), # Added: UUID for task

            "positivePrompt": current_prompt, # Corrected: uses current_prompt from Gemini

            "model":         "runware:100@1", # Reverted to previous model

            "width":         896,

            "height":        1152,

            "steps": 12,

            "scheduler": "DPM++ 3M",        

            "numberResults": 1,

            "outputType":    "URL",

            "checkNSFW": True # Keep checkNSFW if needed for policy adherence

        }]

        

        resp = requests.post(RUNWARE_API_URL, headers=headers, json=payload)

        

        # Added for debugging: Log raw response from Runware AI

        logger.info(f"Runware API Raw Response: {resp.text}")

        

        resp.raise_for_status() # This line will raise HTTPError if status code is 4xx/5xx

        if resp.json().get("images"):

            image_urls.extend(resp.json()["images"])



    paths = []

    for url in image_urls:

        r = requests.get(url)

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



def upload_to_gcs(local_path, destination_blob_name):

    client = storage.Client()

    bucket = client.bucket(GCS_BUCKET_NAME)

    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(local_path)

    blob.make_public()

    return blob.public_url



# --- The Celery Task Definition ---



@celery_app.task # Ensure only one decorator is present

def create_video_task(task_id, audio_url, original_request):

    logger.info(f"--- [CELERY WORKER] Starting video task {task_id} ---")

    local_audio_path = None  # Initialize to None for finally block safety

    images = []  # Initialize to empty list for finally block safety

    final_video_path = None  # Initialize to None for finally block safety

    try:

        local_audio_path = download_audio(audio_url)

        y, sr = librosa.load(local_audio_path)

        _, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

        num_images = max(8, len(beat_frames) // 2)

        num_prompts = num_images // 2



        image_prompts = get_image_prompts_from_gemini(

            vision=original_request.get("vision", ""),

            mood=original_request.get("mood", ""),

            age=original_request.get("age", ""),

            num_prompts=num_prompts,

        )



        images = generate_images(image_prompts, num_images)

        final_video_path = assemble_video(local_audio_path, images)

        

        upload_to_gcs(final_video_path, f"complete/{task_id}.mp4")

        logger.info(f"--- [CELERY WORKER] Successfully completed task {task_id} ---")



    except Exception as e:

        logger.error(

            f"!!! [CELERY WORKER] FAILED task {task_id}: {e} !!!", exc_info=True

        )

    finally:

        # Rename the local audio file instead of deleting it

        if local_audio_path and os.path.exists(local_audio_path):

            new_audio_path = local_audio_path.replace(".mp3", "_DONE.mp3")

            os.rename(local_audio_path, new_audio_path)

            logger.info(f"Audio file renamed to: {new_audio_path}")



        # Clean up other temporary files (video and images)

        cleanup_files = [final_video_path] + images

        for p in cleanup_files:

            if p and os.path.exists(p):

                os.remove(p)
import os

import tempfile

import requests

import logging

import json

import librosa

import itertools

import uuid # Added: For UUID generation

from celery import Celery

from google.cloud import storage

from PIL import Image

from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from moviepy.audio.io.AudioFileClip import AudioFileClip



# --- Configure logging ---

logging.basicConfig(

    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"

)

logger = logging.getLogger(__name__)



# --- Get environment variables ---

# Celery will get these from the worker's environment

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

RUNWARE_API_KEY = os.getenv("RUNWARE_API_KEY")

# Corrected URL: Removed /generate as per user clarification

RUNWARE_API_URL = os.getenv("RUNWARE_API_URL", "https://api.runware.ai/v1") 

GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")



# --- Initialize Celery ---

celery_app = Celery("tasks", broker=REDIS_URL)



# --- Helper Functions ---



def get_image_prompts_from_gemini(vision, mood, age, num_prompts):

    if not GEMINI_API_KEY:

        raise Exception("Missing GEMINI_API_KEY")

    prompt = f"Generate {num_prompts} unique, vivid AI art prompts based on the theme '{vision}', with a '{mood}' mood for a '{age}' audience. Return as a JSON list of strings."

    # Corrected Gemini model name to gemini-2.5-flash

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

        logger.info(f"Requesting image {i+1}/{num_images}...")

        

        # Corrected payload:

        # - Removed "lora" (civitai model) as per debugging

        # - Removed "CFGScale"

        # - Corrected "positivePrompt" to use current_prompt

        # - Model reverted to "runware:100@1" as it was the previous base model

        payload = [{

            "taskType":      "imageInference",

            "taskUUID":      str(uuid.uuid4()), # Added: UUID for task

            "positivePrompt": current_prompt, # Corrected: uses current_prompt from Gemini

            "model":         "runware:100@1", # Reverted to previous model

            "width":         896,

            "height":        1152,

            "steps": 12,

            "scheduler": "DPM++ 3M",        

            "numberResults": 1,

            "outputType":    "URL",

            "checkNSFW": True # Keep checkNSFW if needed for policy adherence

        }]

        

        resp = requests.post(RUNWARE_API_URL, headers=headers, json=payload)

        

        # Added for debugging: Log raw response from Runware AI

        logger.info(f"Runware API Raw Response: {resp.text}")

        

        resp.raise_for_status() # This line will raise HTTPError if status code is 4xx/5xx

        if resp.json().get("images"):

            image_urls.extend(resp.json()["images"])



    paths = []

    for url in image_urls:

        r = requests.get(url)

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



def upload_to_gcs(local_path, destination_blob_name):

    client = storage.Client()

    bucket = client.bucket(GCS_BUCKET_NAME)

    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(local_path)

    blob.make_public()

    return blob.public_url



# --- The Celery Task Definition ---



@celery_app.task # Ensure only one decorator is present

def create_video_task(task_id, audio_url, original_request):

    logger.info(f"--- [CELERY WORKER] Starting video task {task_id} ---")

    local_audio_path = None  # Initialize to None for finally block safety

    images = []  # Initialize to empty list for finally block safety

    final_video_path = None  # Initialize to None for finally block safety

    try:

        local_audio_path = download_audio(audio_url)

        y, sr = librosa.load(local_audio_path)

        _, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

        num_images = max(8, len(beat_frames) // 2)

        num_prompts = num_images // 2



        image_prompts = get_image_prompts_from_gemini(

            vision=original_request.get("vision", ""),

            mood=original_request.get("mood", ""),

            age=original_request.get("age", ""),

            num_prompts=num_prompts,

        )



        images = generate_images(image_prompts, num_images)

        final_video_path = assemble_video(local_audio_path, images)

        

        upload_to_gcs(final_video_path, f"complete/{task_id}.mp4")

        logger.info(f"--- [CELERY WORKER] Successfully completed task {task_id} ---")



    except Exception as e:

        logger.error(

            f"!!! [CELERY WORKER] FAILED task {task_id}: {e} !!!", exc_info=True

        )

    finally:

        # Rename the local audio file instead of deleting it

        if local_audio_path and os.path.exists(local_audio_path):

            new_audio_path = local_audio_path.replace(".mp3", "_DONE.mp3")

            os.rename(local_audio_path, new_audio_path)

            logger.info(f"Audio file renamed to: {new_audio_path}")



        # Clean up other temporary files (video and images)

        cleanup_files = [final_video_path] + images

        for p in cleanup_files:

            if p and os.path.exists(p):

                os.remove(p)
import os

import tempfile

import requests

import logging

import json

import librosa

import itertools

import uuid # Added: For UUID generation

from celery import Celery

from google.cloud import storage

from PIL import Image

from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from moviepy.audio.io.AudioFileClip import AudioFileClip



# --- Configure logging ---

logging.basicConfig(

    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"

)

logger = logging.getLogger(__name__)



# --- Get environment variables ---

# Celery will get these from the worker's environment

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

RUNWARE_API_KEY = os.getenv("RUNWARE_API_KEY")

# Corrected URL: Removed /generate as per user clarification

RUNWARE_API_URL = os.getenv("RUNWARE_API_URL", "https://api.runware.ai/v1") 

GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")



# --- Initialize Celery ---

celery_app = Celery("tasks", broker=REDIS_URL)



# --- Helper Functions ---



def get_image_prompts_from_gemini(vision, mood, age, num_prompts):

    if not GEMINI_API_KEY:

        raise Exception("Missing GEMINI_API_KEY")

    prompt = f"Generate {num_prompts} unique, vivid AI art prompts based on the theme '{vision}', with a '{mood}' mood for a '{age}' audience. Return as a JSON list of strings."

    # Corrected Gemini model name to gemini-2.5-flash

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

        logger.info(f"Requesting image {i+1}/{num_images}...")

        

        # Corrected payload:

        # - Removed "lora" (civitai model) as per debugging

        # - Removed "CFGScale"

        # - Corrected "positivePrompt" to use current_prompt

        # - Model reverted to "runware:100@1" as it was the previous base model

        payload = [{

            "taskType":      "imageInference",

            "taskUUID":      str(uuid.uuid4()), # Added: UUID for task

            "positivePrompt": current_prompt, # Corrected: uses current_prompt from Gemini

            "model":         "runware:100@1", # Reverted to previous model

            "width":         896,

            "height":        1152,

            "steps": 12,

            "scheduler": "DPM++ 3M",        

            "numberResults": 1,

            "outputType":    "URL",

            "checkNSFW": True # Keep checkNSFW if needed for policy adherence

        }]

        

        resp = requests.post(RUNWARE_API_URL, headers=headers, json=payload)

        

        # Added for debugging: Log raw response from Runware AI

        logger.info(f"Runware API Raw Response: {resp.text}")

        

        resp.raise_for_status() # This line will raise HTTPError if status code is 4xx/5xx

        if resp.json().get("images"):

            image_urls.extend(resp.json()["images"])



    paths = []

    for url in image_urls:

        r = requests.get(url)

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



def upload_to_gcs(local_path, destination_blob_name):

    client = storage.Client()

    bucket = client.bucket(GCS_BUCKET_NAME)

    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(local_path)

    blob.make_public()

    return blob.public_url



# --- The Celery Task Definition ---



@celery_app.task # Ensure only one decorator is present

def create_video_task(task_id, audio_url, original_request):

    logger.info(f"--- [CELERY WORKER] Starting video task {task_id} ---")

    local_audio_path = None  # Initialize to None for finally block safety

    images = []  # Initialize to empty list for finally block safety

    final_video_path = None  # Initialize to None for finally block safety

    try:

        local_audio_path = download_audio(audio_url)

        y, sr = librosa.load(local_audio_path)

        _, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

        num_images = max(8, len(beat_frames) // 2)

        num_prompts = num_images // 2



        image_prompts = get_image_prompts_from_gemini(

            vision=original_request.get("vision", ""),

            mood=original_request.get("mood", ""),

            age=original_request.get("age", ""),

            num_prompts=num_prompts,

        )



        images = generate_images(image_prompts, num_images)

        final_video_path = assemble_video(local_audio_path, images)

        

        upload_to_gcs(final_video_path, f"complete/{task_id}.mp4")

        logger.info(f"--- [CELERY WORKER] Successfully completed task {task_id} ---")



    except Exception as e:

        logger.error(

            f"!!! [CELERY WORKER] FAILED task {task_id}: {e} !!!", exc_info=True

        )

    finally:

        # Rename the local audio file instead of deleting it

        if local_audio_path and os.path.exists(local_audio_path):

            new_audio_path = local_audio_path.replace(".mp3", "_DONE.mp3")

            os.rename(local_audio_path, new_audio_path)

            logger.info(f"Audio file renamed to: {new_audio_path}")



        # Clean up other temporary files (video and images)

        cleanup_files = [final_video_path] + images

        for p in cleanup_files:

            if p and os.path.exists(p):

                os.remove(p)
