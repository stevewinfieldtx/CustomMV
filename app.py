from flask import Flask, render_template, request, jsonify
import os
import logging
from dotenv import load_dotenv
import music_creator
from celery_worker import create_video_task # Import the new task

# Load environment variables
load_dotenv()

# Configure logging
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=getattr(logging, log_level), format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static', template_folder='templates')

# --- We no longer need the complex SSE/Queue logic ---
# A simple dictionary to hold original request data is enough
tasks_data = {} 

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

        # Store the original request data to be used in the callback
        tasks_data[task_id] = data
        
        # --- Return a simple success message to the frontend ---
        return jsonify({'success': True, 'message': f"Music task {task_id} started. Video will be created in the background."})
    except Exception as e:
        logger.error("Failed during task creation", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/music-callback', methods=['POST'])
def music_callback():
    try:
        payload = request.get_json()
        if not payload: return 'Empty payload', 400

        callback_data = payload.get('data', {})
        if callback_data.get('callbackType') != 'complete':
            return 'Intermediate callback', 200

        task_id = callback_data.get('task_id')
        song_list = callback_data.get('data', [])
        audio_url = next((song.get('audio_url') for song in song_list if song.get('audio_url')), None)

        if not task_id or not audio_url:
            return 'Missing data', 400
        
        original_request = tasks_data.pop(task_id, {}) # Get and remove data
        if not original_request:
            return 'Unknown task', 200

        # --- THIS IS THE KEY CHANGE ---
        # Instead of doing the work here, we send it to our Celery worker.
        # .delay() sends the task to the Redis queue.
        create_video_task.delay(task_id, audio_url, original_request)
        logger.info(f"Task {task_id} sent to Celery worker.")

    except Exception as e:
        logger.error(f"Error in music callback: {e}", exc_info=True)
        return 'Internal Server Error', 500

    return '', 204

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
