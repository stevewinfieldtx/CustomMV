from flask import Flask, render_template, request, jsonify
import os, logging
from dotenv import load_dotenv
import music_creator
from celery_worker import create_video_task

load_dotenv()
logging.basicConfig(level=os.getenv('LOG_LEVEL','INFO').upper(), 
                    format='%(asctime)s - %(levelname)s - %(message)s')
app = Flask(__name__, static_folder='static', template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/create', methods=['POST'])
def create():
    # … your create logic …
    return jsonify({ 'success': True }), 200

@app.route('/music-callback', methods=['POST'])
def music_callback():
    # … your callback logic …
    return '', 204

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)), debug=True)
