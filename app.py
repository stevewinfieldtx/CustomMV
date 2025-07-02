from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import os
import logging
import json
import queue
from dotenv import load_dotenv
import music_creator
import video_creator

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

@app.route('/health', methods=['GET'])
def health():
    missing = []
    if not os.getenv('GEMINI_API_KEY'): missing.append('GEMINI_API_KEY')
    if not os.getenv('APIBOX_KEY'): missing.append('APIBOX_KEY')
    if not os.getenv('RUNWARE_API_KEY'): missing.append('RUNWARE_API_KEY')
    if not os.getenv('GCS_BUCKET_NAME'): missing.append('GCS_BUCKET_NAME')
    return jsonify({'status': 'ok' if not missing else 'missing_env_vars', 'missing': missing})

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

        tasks_data[task_id] = {
            'queue': queue.Queue(),
            'request_data': data
        }
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

        q = task_info['queue']
        message = q.get()
        event, data = message
        
        yield f'event: {event}\ndata: {json.dumps(data)}\n\n'
        
        if task_id in tasks_data:
            del tasks_data[task_id]
            logger.info(f"Cleaned up task data for {task_id}")

    return Response(stream_with_context(gen()), mimetype='text/event-stream')

# --- MODIFIED: This is the updated callback handler ---
@app.route('/music-callback', methods=['POST'])
def music_callback():
    task_id = None
    try:
        payload = request.get_json()
        if not payload:
            logger.error("Music callback received empty payload.")
            return 'Empty payload', 400

        # The actual data is nested inside the 'data' key
        callback_data = payload.get('data', {})
        
        # Only proceed when the generation is fully complete
        if callback_data.get('callbackType') != 'complete':
            logger.info(f"Received intermediate callback of type '{callback_data.get('callbackType')}'. Ignoring.")
            return 'Intermediate callback received', 200

        # Correctly parse the nested structure
        task_id = callback_data.get('task_id')
        song_list = callback_data.get('data', [])
        
        # Find the first song in the list that has a valid audio_url
        audio_url = next((song.get('audio_url') for song in song_list if song.get('audio_url')), None)

        if not task_id or not audio_url:
            logger.error(f"Callback missing task_id or audio_url. Full payload: {payload}")
            return 'Missing data in final callback', 400
        
        logger.info(f"Received FINAL music callback for task_id: {task_id}")

        task_info = tasks_data.get(task_id)
        if not task_info:
            logger.warning(f"Callback received for an unknown or expired task_id: {task_id}")
            return 'Unknown task', 200

        original_data = task_info['request_data']
        q = task_info['queue']

        prompt_parts = [
            original_data.get('vision', ''),
            f"mood: {original_data.get('mood', '')}",
            f"target audience age: {original_data.get('age', '')}",
            f"in the style of: {original_data.get('artist', '')}"
        ]
        image_prompt = ", ".join(filter(None, prompt_parts))
        logger.info(f"Generated image prompt for task {task_id}: {image_prompt}")

        video_url = video_creator.create_demo_video(
            image_prompt=image_prompt,
            audio_path=audio_url,
            output_path=f"static/output/{task_id}.mp4"
        )

        q.put(('complete', {'audio_url': audio_url, 'video_url': video_url}))
        logger.info(f"Successfully processed task {task_id}. Video at {video_url}")

    except Exception as e:
        logger.error(f"Error processing music callback for task {task_id}", exc_info=True)
        if task_id and task_id in tasks_data:
            q = tasks_data[task_id]['queue']
            q.put(('error', {'message': f"Video creation failed. Please check logs. Error: {str(e)}"}))
        return 'Internal Server Error', 500

    return '', 204
    
if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
