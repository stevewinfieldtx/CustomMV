#app.py
from flask import Flask, render_template, request, jsonify
from urllib.parse import urljoin
import os
import logging
from dotenv import load_dotenv
import music_creator
from celery_worker import poll_music_status

# Load env vars
load_dotenv()

# Logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO').upper(),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/create', methods=['POST'])
def create():
    data = request.get_json() or {}
    logger.info(f"Create request: {data}")
    try:
        # Generate Suno tags via Gemini
        tags = music_creator.get_tags_from_gemini(
            target=data.get('artist') or data.get('vision', ''),
            kind='artist' if data.get('artist') else 'vision',
            length=f"{data.get('length')} sec",
            mood=data.get('mood'),
            age=data.get('age')
        )
        # Kick off music generation and polling
        callback_url = urljoin(request.url_root, 'music-callback')
        task_id = music_creator.start_music_generation(tags, callback_url)
        logger.info(f"Music generation started: {task_id}")

        # Enqueue polling to trigger video creation
        poll_music_status.delay(task_id, data)
        logger.info(f"Scheduled polling for {task_id}")

        return jsonify({
            'success': True,
            'message': f"Music task {task_id} started; video will follow shortly."
        }), 200

    except Exception as e:
        logger.error("Failed in /create", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
