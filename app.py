from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import os
import logging
import json
import queue
from dotenv import load_dotenv
import music_creator
import video_creator # Keep for download_audio and GCS upload helpers
from google.cloud import storage

# Load environment variables
load_dotenv()

# Configure logging
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=getattr(logging, log_level), format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

tasks_data: dict[str, dict] = {}
app = Flask(__name__, static_folder='static', template_folder='templates')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/create', methods=['POST'])
def create():
    data = request.get_json() or {}
    logger.info(f"Create request: {data}")
    try:
        tags = music_creator.get_tags_from_gemini(
            target=data.get('artist','') or data.get('vision',''),
            kind='artist' if data.get('artist') else 'vision',
            length=data.get('length'),
            mood=data.get('mood'),
            age=data.get('age')
        )
        task_id = music_creator.start_music_generation(tags)
        logger.info(f"Music task started: {task_id}")

        tasks_data[task_id] = { 'queue': queue.Queue(), 'request_data': data }
        return jsonify({'success': True, 'task_id': task_id})
    except Exception as e:
        logger.error("Failed during task creation", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/events/<task_id>')
def events(task_id):
    def gen():
        task_info = tasks_data.get(task_id)
        if not task_info:
            yield 'event: error\ndata: {"message":"Unknown or expired task ID."}\n\n'
            return

        message = task_info['queue'].get()
        event, data = message
        yield f'event: {event}\ndata: {json.dumps(data)}\n\n'
        
        if task_id in tasks_data:
            del tasks_data[task_id]
            logger.info(f"Cleaned up task data for {task_id}")

    return Response(stream_with_context(gen()), mimetype='text/event-stream')

@app.route('/music-callback', methods=['POST'])
def music_callback():
    task_id = None
    try:
        payload = request.get_json()
        if not payload:
            logger.error("Callback received empty payload.")
            return 'Empty payload', 400

        callback_data = payload.get('data', {})
        if callback_data.get('callbackType') != 'complete':
            return 'Intermediate callback received', 200

        task_id = callback_data.get('task_id')
        song_list = callback_data.get('data', [])
        audio_url = next((song.get('audio_url') for song in song_list if song.get('audio_url')), None)

        if not task_id or not audio_url:
            return 'Missing data in final callback', 400
        
        task_info = tasks_data.get(task_id)
        if not task_info:
            return 'Unknown task', 200

        original_data = task_info['request_data']
        q = task_info['queue']
        
        # --- NEW LOGIC: UPLOAD JOB TO GCS ---
        logger.info(f"Uploading job files for task {task_id} to GCS...")
        
        # 1. Download the audio locally
        local_audio_path = video_creator.download_audio(audio_url)
        
        # 2. Upload the audio to the 'pending' folder
        gcs_audio_path = f"pending/{task_id}.mp3"
        video_creator.upload_to_gcs(local_audio_path, gcs_audio_path)
        os.remove(local_audio_path) # Clean up local file

        # 3. Create the JSON job file
        job_data = {
            "task_id": task_id,
            "original_request": original_data,
            "audio_gcs_path": gcs_audio_path
        }
        
        # 4. Upload the JSON to the 'pending' folder
        storage_client = storage.Client()
        bucket = storage_client.bucket(os.getenv('GCS_BUCKET_NAME'))
        json_blob = bucket.blob(f"pending/{task_id}.json")
        json_blob.upload_from_string(json.dumps(job_data, indent=2), content_type='application/json')
        
        logger.info(f"Successfully handed off job {task_id} to GCS.")

        # --- Notify frontend that the handoff was successful ---
        q.put(('complete', {'message': f"Video processing for task {task_id} has started."}))

    except Exception as e:
        logger.error(f"Error in GCS handoff for task {task_id}", exc_info=True)
        if task_id and task_id in tasks_data:
            tasks_data[task_id]['queue'].put(('error', {'message': f"Failed to start video processing: {str(e)}"}))
        return 'Internal Server Error', 500

    return '', 204

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
