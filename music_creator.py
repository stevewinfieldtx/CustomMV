import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

def get_tags_from_gemini(target: str, kind: str='artist', length: str='1 min', mood: str=None, age: str=None) -> str:
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key: raise Exception('Missing GEMINI_API_KEY')
    lines = [
        'You are a music AI assistant for Suno.',
        f'Reference: {kind} [{target}]',
        f'Song Length: {length}',
    ]
    if mood: lines.append(f'Mood: {mood}')
    if age:  lines.append(f'Audience Age: {age}')
    lines += [
        'Return only a comma-separated list of tags (no extra text).'
    ]
    prompt = '\n'.join(lines)
    url = f'https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent?key={api_key}'
    resp = requests.post(url, json={'contents':[{'parts':[{'text':prompt}]}]}, headers={'Content-Type':'application/json'})
    resp.raise_for_status()
    text = resp.json()['candidates'][0]['content']['parts'][0]['text']
    return ', '.join(t.strip() for t in text.split(',') if t.strip())


def start_music_generation(prompt: str) -> str:
    api_key = os.getenv('APIBOX_KEY')
    if not api_key: raise Exception('Missing APIBOX_KEY')
    callback_url = f"https://{os.getenv('RAILWAY_PUBLIC_DOMAIN','example.com')}/music-callback"
    payload = {"prompt":prompt, "customMode":False, "instrumental":True, "model":"V4_5", "callBackUrl":callback_url}
    resp = requests.post('https://apibox.erweima.ai/api/v1/generate', headers={'Authorization':f'Bearer {api_key}'}, json=payload)
    resp.raise_for_status()
    return resp.json()['data']['taskId']