import os
import tempfile
import requests
from dotenv import load_dotenv
from google.cloud import storage
from PIL import Image
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
import librosa

# Load environment variables
load_dotenv()
RUNWARE_API_KEY = os.getenv('RUNWARE_API_KEY')
RUNWARE_API_URL = os.getenv('RUNWARE_API_URL', 'https://api.runware.ai/v1/generate')
GCS_BUCKET_NAME = os.getenv('GCS_BUCKET_NAME')

def download_audio(source: str) -> str:
    if source.startswith('http'):
        resp = requests.get(source)
        resp.raise_for_status()
        fd, tmp = tempfile.mkstemp(suffix='.mp3')
        os.close(fd)
        with open(tmp, 'wb') as f:
            f.write(resp.content)
        return tmp
    if not GCS_BUCKET_NAME:
        raise Exception('GCS_BUCKET_NAME not set')
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(source)
    fd, tmp = tempfile.mkstemp(suffix='.mp3')
    os.close(fd)
    blob.download_to_filename(tmp)
    return tmp

def generate_images(prompt: str, num_images: int) -> list[str]:
    headers = {
        'Authorization': f'Bearer {RUNWARE_API_KEY}',
        'Content-Type': 'application/json'
    }
    payload = {'prompt': prompt, 'num_images': num_images}
    resp = requests.post(RUNWARE_API_URL, headers=headers, json=payload)
    resp.raise_for_status()
    data = resp.json()
    image_urls = data.get('images', [])
    paths = []
    for url in image_urls:
        r = requests.get(url)
        r.raise_for_status()
        fd, img_path = tempfile.mkstemp(suffix='.png')
        os.close(fd)
        with open(img_path, 'wb') as f:
            f.write(r.content)
        paths.append(img_path)
    return paths

def assemble_video(audio_path: str, image_paths: list[str]) -> str:
    duration = librosa.get_duration(filename=audio_path)
    fps = len(image_paths) / duration if duration > 0 else 24
    clip = ImageSequenceClip(image_paths, fps=fps)
    audio_clip = AudioFileClip(audio_path)
    final = clip.set_audio(audio_clip).set_duration(audio_clip.duration)
    fd, tmp_video = tempfile.mkstemp(suffix='.mp4')
    os.close(fd)
    final.write_videofile(tmp_video, codec='libx264', audio_codec='aac')
    return tmp_video

def create_demo_video(image_prompt: str, audio_path: str, output_path: str) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    local_audio = download_audio(audio_path)
    y, sr = librosa.load(local_audio)
    _, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    num_images = max(1, len(beat_frames))
    images = generate_images(image_prompt, num_images)
    tmp_video = assemble_video(local_audio, images)
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET_NAME)
    blob_name = f"videos/{os.path.basename(output_path)}"
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(tmp_video)
    blob.make_public()
    public_url = blob.public_url
    os.replace(tmp_video, output_path)
    for p in images + [local_audio]:
        try:
            os.remove(p)
        except:
            pass
    return public_url

async def generate_visuals(config, num_images):
    """
    Generates a sequence of images (visuals) for the video based on config.
    Orchestrates multiple parallel calls to generate_single_image.
    """
    global RUNWARE_API_KEY
    load_dotenv() # Load .env for local testing, App Engine provides env vars
    RUNWARE_API_KEY = os.getenv("RUNWARE_API_KEY")
    if not RUNWARE_API_KEY:
        log("❌ ERROR: RUNWARE_API_KEY not found in .env file or environment variables.")
        return None

    # Ensure the temporary directory for images exists and is clean for a new run.
    os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)
    # Clear existing files in the directory to prevent old images from being used.
    for filename in os.listdir(TEMP_IMAGE_DIR):
        file_path = os.path.join(TEMP_IMAGE_DIR, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path) # Remove file or symbolic link
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path) # Remove directory
        except Exception as e:
            log(f'Failed to delete {file_path}. Reason: {e}')

    # Determine image dimensions based on desired aspect ratio
    dims = ASPECT_RATIO_DIMS.get(config["aspect_ratio"], {"width": 1024, "height": 576})
    
    # Create a list of asynchronous tasks for image generation
    tasks = [
        generate_single_image(
            config["visual_prompt"],
            dims["width"],
            dims["height"],
            os.path.join(TEMP_IMAGE_DIR, f"{i:04d}.jpg") # Filename like 0000.jpg, 0001.jpg
        ) for i in range(num_images)
    ]
    
    # Run all image generation tasks concurrently
    results = await asyncio.gather(*tasks)
    image_paths = [res for res in results if res is not None] # Filter out failed generations

    if not image_paths or len(image_paths) < num_images:
        log(f"❌ WARNING: Image generation partially failed. Got {len(image_paths)} of {num_images} images.")
        if not image_paths: return None # No images generated at all

    log(f"✅ Successfully generated {len(image_paths)} images.")
    return image_paths

def process_images(image_paths, aspect_ratio):
    """
    Standardizes the size of all generated images to the final video dimensions.
    This ensures all images have consistent resolution for video assembly.
    """
    log("Standardizing final images...")
    # These are the *final output dimensions* for the video, potentially higher res than generated.
    final_dims = {
        '16:9 (Widescreen)': (1920, 1080),
        '4:3 (Classic)': (1440, 1080),
        '3:4 (Portrait)': (810, 1080)
    }
    target_size = final_dims.get(aspect_ratio, (1920, 1080)) # Default to widescreen if not found

    for image_path in image_paths:
        with Image.open(image_path) as img:
            # Convert to RGB to ensure compatibility (e.g., if some images are RGBA)
            # Resize using LANCZOS filter for high-quality downsampling.
            img = img.convert("RGB").resize(target_size, Image.LANCZOS)
            img.save(image_path, "JPEG", quality=95) # Overwrite with processed image

def upload_file_to_gcs(local_file_path, destination_blob_name):
    """
    Uploads a file from the local filesystem to a specified path in the GCS bucket.
    Returns the public URL of the uploaded file on success, None otherwise.
    """
    if not GCS_BUCKET_NAME:
        log("❌ ERROR: GCS_BUCKET_NAME environment variable not set. Cannot upload to GCS.")
        return None
    try:
        storage_client_local = storage.Client() # Create client within function for thread safety if needed
        bucket = storage_client_local.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(local_file_path)
        log(f"✅ File {local_file_path} uploaded to gs://{GCS_BUCKET_NAME}/{destination_blob_name}")
        # Return a public URL for retrieval
        return f"https://storage.googleapis.com/{GCS_BUCKET_NAME}/{destination_blob_name}"
    except Exception as e:
        log(f"❌ ERROR: Failed to upload {local_file_path} to GCS: {e}")
        return None

def download_file_from_gcs(source_blob_name, destination_local_path):
    """
    Downloads a file (blob) from the GCS bucket to a specified local path.
    Returns the local path on success, None otherwise.
    """
    if not GCS_BUCKET_NAME:
        log("❌ ERROR: GCS_BUCKET_NAME environment variable not set. Cannot download from GCS.")
        return None
    try:
        storage_client_local = storage.Client()
        bucket = storage_client_local.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_local_path)
        log(f"✅ File gs://{GCS_BUCKET_NAME}/{source_blob_name} downloaded to {destination_local_path}")
        return destination_local_path
    except Exception as e:
        log(f"❌ ERROR: Failed to download {source_blob_name} from GCS: {e}")
        return None

def main(audio_gcs_path, config_path):
    """
    Main function to orchestrate video creation.
    Downloads audio from GCS, generates visuals, assembles the video, and uploads to GCS.
    """
    log("--- VIDEO CREATOR SCRIPT STARTED ---")
    
    # Define a local path for the downloaded audio file
    local_audio_path = os.path.join("/tmp", "downloaded_audio.mp3")
    
    # Download the audio file from GCS using the provided GCS path (blob name)
    downloaded_audio_local_path = download_file_from_gcs(audio_gcs_path, local_audio_path)
    if not downloaded_audio_local_path:
        log(f"❌ ERROR: Could not download audio from GCS path: {audio_gcs_path}")
        return None # Indicate failure

    try:
        # Load the video configuration from the JSON file
        with open(config_path, "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        log(f"❌ ERROR: Config file not found at {config_path}")
        return None
    except json.JSONDecodeError:
        log(f"❌ ERROR: Failed to parse JSON from config file at {config_path}")
        return None
    except Exception as e:
        log(f"❌ ERROR: Failed to load config from {config_path}: {e}")
        return None

    log(f"Analyzing beats for {downloaded_audio_local_path}...")
    try:
        # Load audio and detect beats using librosa
        y, sr = librosa.load(downloaded_audio_local_path, sr=None)
        _, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beats = librosa.frames_to_time(beat_frames, sr=sr).tolist()
        log(f"Detected {len(beats)} beats.")
    except Exception as e:
        log(f"❌ ERROR: Failed to analyze audio beats: {e}")
        return None

    if not beats:
        log("❌ ERROR: No beats detected or audio file problematic. Cannot create video.")
        return None

    # Calculate the number of images needed based on pacing and number of beats
    pacing_divisor = PACING_MAP.get(config["pacing"], 1)
    num_images = math.ceil(len(beats) / pacing_divisor)
    log(f"Pacing '{config['pacing']}' requires {num_images} images.")

    # Generate the visual images asynchronously
    image_paths = asyncio.run(generate_visuals(config, num_images))
    if not image_paths:
        log("❌ ERROR: No images generated. Cannot create video.")
        return None

    # Process (resize/convert) the generated images for final video assembly
    process_images(image_paths, config["aspect_ratio"])

    log("Assembling final video...")
    audio_duration = librosa.get_duration(path=downloaded_audio_local_path)
    # Calculate frames per second for the video clip. Ensure it's not zero.
    fps = len(image_paths) / audio_duration if audio_duration > 0 and len(image_paths) > 0 else 24

    # Define the local and GCS paths for the final video output
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_video_local_filename = f"final_video_{timestamp}_{os.urandom(4).hex()}.mp4" # Add random suffix for uniqueness
    output_video_local_path = os.path.join("/tmp", output_video_local_filename)
    output_gcs_blob_name = f"videos/{output_video_local_filename}" # Desired path in GCS bucket

    try:
        # Ensure temporary audio directory for moviepy exists and is clean
        os.makedirs(TEMP_MOVIEPY_AUDIO_DIR, exist_ok=True)
        for filename in os.listdir(TEMP_MOVIEPY_AUDIO_DIR):
            file_path = os.path.join(TEMP_MOVIEPY_AUDIO_DIR, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                log(f'Failed to delete {file_path}. Reason: {e}')

        # Create AudioFileClip from the downloaded audio
        with AudioFileClip(downloaded_audio_local_path) as audio_clip:
            # Create ImageSequenceClip from the generated images
            with ImageSequenceClip(image_paths, fps=fps) as clip:
                # Set the duration of the video clip to match the audio duration
                final_clip = clip.set_duration(audio_clip.duration).set_audio(audio_clip)

                # Apply fade effects to the video and audio
                final_clip = fadeout(final_clip, 2)
                final_clip = audio_fadeout(final_clip, 2)

                # Write the final video file locally to /tmp
                final_clip.write_videofile(
                    output_video_local_path,
                    codec="libx264",
                    audio_codec="aac",
                    # Specify a temporary audio file path within /tmp for moviepy
                    temp_audiofile=os.path.join(TEMP_MOVIEPY_AUDIO_DIR, 'moviepy_temp_audio.m4a'),
                    remove_temp=True, # Moviepy will try to remove its temp files
                    threads=4 # Use multiple threads for faster encoding
                )

        log(f"✅ Video created locally: {output_video_local_path}")

        # Upload the final video to Google Cloud Storage
        gcs_video_url = upload_file_to_gcs(output_video_local_path, output_gcs_blob_name)
        if gcs_video_url:
            log(f"✅✅✅ Video successfully uploaded to GCS: {gcs_video_url}")
            return gcs_video_url # Return the GCS URL on success
        else:
            log("❌ ERROR: Video upload to GCS failed.")
            return None

    except Exception as e:
        log(f"❌ ERROR during video assembly or GCS upload: {e}")
        return None # Indicate failure
    finally:
        # --- Cleanup Temporary Files and Directories ---
        if os.path.exists(TEMP_IMAGE_DIR):
            shutil.rmtree(TEMP_IMAGE_DIR)
            log(f"Cleaned up {TEMP_IMAGE_DIR}")
        if os.path.exists(TEMP_MOVIEPY_AUDIO_DIR):
            shutil.rmtree(TEMP_MOVIEPY_AUDIO_DIR)
            log(f"Cleaned up {TEMP_MOVIEPY_AUDIO_DIR}")
        if os.path.exists(downloaded_audio_local_path):
            os.remove(downloaded_audio_local_path)
            log(f"Cleaned up {downloaded_audio_local_path}")
        if os.path.exists(output_video_local_path):
            os.remove(output_video_local_path) # Clean up local video file after upload
            log(f"Cleaned up {output_video_local_path}")


if __name__ == "__main__":
    # This block runs when video_creator.py is executed directly (e.g., by app.py's subprocess.run)
    if len(sys.argv) != 3:
        print("Usage: python video_creator.py <gcs_path_to_audio_file> <path_to_config_file>")
        sys.exit(1)

    # sys.argv[1] is the GCS blob name for the audio file
    audio_file_gcs_path = sys.argv[1] 
    config_file_path = sys.argv[2] # Path to the job config JSON file

    # Call the main video creation function
    final_video_url = main(audio_file_gcs_path, config_file_path)
    if final_video_url:
        log(f"Video creation completed. Final video URL: {final_video_url}")
        # IMPORTANT: Print the URL in a parsable format for app.py
        # app.py will look for this specific string to get the final video URL.
        print(f"FINAL_VIDEO_URL: {final_video_url}")
        sys.exit(0) # Indicate success
    else:
        log("❌ Video creation failed. Exiting video_creator.py.")
        sys.exit(1) # Indicate failure
