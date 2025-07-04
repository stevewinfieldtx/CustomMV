# Fix for both celery_worker.py and video_creator.py

# 1. CORRECT API URL (fix video_creator.py)
RUNWARE_API_URL = os.getenv('RUNWARE_API_URL', 'https://api.runware.ai/v1')  # No /generate!

# 2. CORRECT PAYLOAD FOR FLUX.1 SCHNELL
def generate_images(prompts, num_images):
    if not RUNWARE_API_KEY:
        raise Exception("RUNWARE_API_KEY is not set.")
    
    headers = {
        "Authorization": f"Bearer {RUNWARE_API_KEY}",
        "Content-Type": "application/json",
    }
    
    image_urls = []
    prompt_cycle = itertools.cycle(prompts)
    
    for i in range(num_images):
        current_prompt = next(prompt_cycle)
        logger.info(f"Requesting image {i+1}/{num_images}...")
        
        # CORRECT payload for Flux.1 Schnell
        payload = [{
            "taskType": "imageInference",
            "model": "runware:100@1",
            "positivePrompt": current_prompt,
            "numberResults": 1,
            "outputFormat": "JPEG",
            "width": 896,
            "height": 1152,
            "steps": 8, 
            "scheduler": "euler", 
            "checkNSFW": True
        }]
        
        resp = requests.post(RUNWARE_API_URL, headers=headers, json=payload)
        resp.raise_for_status()
        
        # Check response structure
        response_data = resp.json()
        if response_data.get('images'):
            image_urls.extend(response_data['images'])
        elif response_data.get('data') and response_data['data'].get('images'):
            image_urls.extend(response_data['data']['images'])
        else:
            logger.warning(f"No images returned for prompt: {current_prompt}")
    
    # Download images
    paths = []
    for i, url in enumerate(image_urls):
        r = requests.get(url)
        r.raise_for_status()
        fd, path = tempfile.mkstemp(suffix=".jpeg")
        os.close(fd)
        with open(path, "wb") as f:
            f.write(r.content)
        paths.append(path)
    
    return paths

# 3. ALTERNATIVE: Check Runware documentation for exact model name
# Common Flux.1 Schnell model names might be:
# - "flux-schnell"
# - "flux-1-schnell"
# - "black-forest-labs/flux-1-schnell"
# - Or check their API docs for the exact identifier
