@celery_app.task
def poll_music_status(task_id, original_request):
    """
    Poll Apibox’s record-info endpoint until the MP3 is ready,
    then enqueue the video task.
    """
    if not APIBOX_KEY:
        raise Exception("Missing APIBOX_KEY")

    status_url = "https://apibox.erweima.ai/api/v1/generate/record-info"
    headers = { 'Authorization': f'Bearer {APIBOX_KEY}' }
    params  = { 'task_id': task_id }

    resp = requests.get(status_url, headers=headers, params=params)
    resp.raise_for_status()
    body = resp.json()

    # Apibox wraps results under data → data (a list of entries)
    info_list = body.get('data', {}).get('data', [])
    state     = body.get('data', {}).get('status')  # e.g. "PENDING", "SUCCESS", etc.

    if state == "SUCCESS" and info_list:
        audio_url = info_list[0].get('audio_url') or info_list[0].get('audioUrl')
        if not audio_url:
            raise Exception(f"No audio_url in record-info response: {body}")
        # Kick off video creation
        create_video_task.delay(task_id, audio_url, original_request)

    elif state in ("PENDING", "TEXT_SUCCESS", "FIRST_SUCCESS"):
        # retry after 15 seconds
        poll_music_status.apply_async((task_id, original_request), countdown=15)

    else:
        # any other status is a hard failure
        raise Exception(f"Music generation failed (state={state}): {body}")
