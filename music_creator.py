#music_creator.py
import os
import requests

def get_tags_from_gemini(target: str,
                         kind: str = 'artist',
                         length: str = '60 sec',
                         mood: str = None,
                         age: str = None) -> str:
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise Exception('Missing GEMINI_API_KEY')

    lines = [
        'You are a music AI assistant for Suno.',
        f'Reference: {kind} [{target}]',
        f'Song Length: {length}'
    ]
    if mood:
        lines.append(f'Mood: {mood}')
    if age:
        lines.append(f'Audience Age: {age}')
    lines.append('Return only a comma-separated list of tags (no extra text).')

    prompt = '\n'.join(lines)
    url = (
        f'https://generativelanguage.googleapis.com/v1/models/'
        f'gemini-2.5-flash:generateContent?key={api_key}'
    )
    resp = requests.post(
        url,
        json={'contents': [{'parts': [{'text': prompt}]}]},
        headers={'Content-Type': 'application/json'}
    )
    resp.raise_for_status()

    try:
        body = resp.json()
    except ValueError:
        raise Exception(f"Invalid JSON from Gemini API: {resp.text}")

    candidates = body.get('candidates')
    if not candidates or not isinstance(candidates, list):
        raise Exception(f"No candidates in Gemini response: {body}")

    text = (
        candidates[0]
        .get('content', {})
        .get('parts', [{}])[0]
        .get('text', '')
    )
    if not text:
        raise Exception(f"Empty tag list from Gemini: {body}")

    tags = [t.strip() for t in text.split(',') if t.strip()]
    return ', '.join(tags)

def start_music_generation(prompt: str, callback_url: str) -> str:
    api_key = os.getenv('APIBOX_KEY')
    if not api_key:
        raise Exception('Missing APIBOX_KEY')

    payload = {
        'prompt': prompt,
        'customMode': False,
        'instrumental': True,
        'model': 'V4_5',
        'callbackUrl': callback_url
    }

    resp = requests.post(
        'https://apibox.erweima.ai/api/v1/generate',
        headers={
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        },
        json=payload
    )
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        raise Exception(f"Music API request failed: {e}, body: {resp.text}")

    try:
        body = resp.json()
    except ValueError:
        raise Exception(f"Invalid JSON from music API: {resp.text}")

    data = body.get('data')
    if not isinstance(data, dict):
        raise Exception(f"Unexpected data format from music API: {body}")

    task_id = data.get('taskId') or data.get('task_id')
    if not task_id:
        raise Exception(f"Missing taskId in music API response: {body}")

    return task_id
