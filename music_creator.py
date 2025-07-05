import os
import requests
from urllib.parse import urljoin


def get_tags_from_gemini(target: str,
                         kind: str = 'artist',
                         length: str = '60 sec',
                         mood: str = None,
                         age: str = None) -> str:
    """
    Call Google Gemini to generate a comma-separated list of Suno tags
    for a given target (artist or vision), length, mood, and age.
    Returns a single string of tags joined by commas.
    """
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
        f'https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash'
        f':generateContent?key={api_key}'
    )
    payload = {'contents': [{'parts': [{'text': prompt}]}]}
    resp = requests.post(url, json=payload,
                         headers={'Content-Type': 'application/json'})
    resp.raise_for_status()

    text = resp.json()['candidates'][0]['content']['parts'][0]['text']
    # Clean and return
    tags = [t.strip() for t in text.split(',') if t.strip()]
    return ', '.join(tags)


def start_music_generation(prompt: str, callback_url: str) -> str:
    """
    Kick off a Suno/Apibox music generation, passing in the prompt (tags)
    and the webhook callback_url. Returns the taskId.
    """
    api_key = os.getenv('APIBOX_KEY')
    if not api_key:
        raise Exception('Missing APIBOX_KEY')

    payload = {
        'prompt': prompt,
        'customMode': False,
        'instrumental': True,
        'model': 'V4_5',
        'callBackUrl': callback_url
    }
    resp = requests.post(
        'https://apibox.erweima.ai/api/v1/generate',
        headers={'Authorization': f'Bearer {api_key}',
                 'Content-Type': 'application/json'},
        json=payload
    )
    resp.raise_for_status()
    return resp.json()['data']['taskId']
