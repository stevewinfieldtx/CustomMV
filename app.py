from flask import Flask, render_template, request, jsonify
from urllib.parse import urljoin
import os
import logging
from dotenv import load_dotenv
import music_creator
from celery_worker import create_video_task

# Load environment variables\load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO').upper(),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask
app = Flask(__name__, static_folder='static', template_folder='templates')

# In-memory store for original requests
tasks_data = {}

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/create', methods=['POST'])
def create():
    data = request.get_json() or {}
    logger.info(f"Create request: {data}")
    try:
        # Build Suno AI tag prompt
        tags = music_creator.get_tags_from_gemini(
            target=data.get('artist', '') or data.get('vision', ''),
            kind='artist' if data.get('artist') else 'vision',
            length=f"{data.get('length')} sec",
            mood=data.get('mood'),
            age=data.get('age')
        )
        # Determine correct callback URL
        callback_url = urljoin(request.url_root, 'music-callback')
        # Start music generation
        task_id = music_creator.start_music_generation(tags, callback_url=callback_url)
        logger.info(f"Music task started: {task_id}")

        # Store original request for callback handling
        tasks_data[task_id] = data

        return jsonify({
            'success': True,
            'message': f"Music task {task_id} started. Video will process in background."
        }), 200

    except Exception as e:
        logger.error("Failed to create music task", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/music-callback', methods=['POST'])
def music_callback():
    try:
        payload = request.get_json()
        if not payload:
            return 'Empty payload', 400

        callback_data = payload.get('data', {})
        if callback_data.get('callbackType') != 'complete':
            return 'Intermediate callback', 200

        task_id = callback_data.get('task_id') or callback_data.get('taskId')
        # Extract audio_url from nested data list
        song_list = callback_data.get('data', [])
        audio_url = next((s.get('audio_url') or s.get('url') for s in song_list if s.get('audio_url') or s.get('url')), None)

        if not task_id or not audio_url:
            return 'Missing data', 400

        original = tasks_data.pop(task_id, None)
        if not original:
            return 'Unknown task', 200

        # Enqueue celery video-creation job
        create_video_task.delay(task_id, audio_url, original)
        logger.info(f"Enqueued video task {task_id}")

        return '', 204

    except Exception as e:
        logger.error(f"Error in music callback: {e}", exc_info=True)
        return 'Internal Server Error', 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
