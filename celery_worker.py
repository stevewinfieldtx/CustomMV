@celery_app.task
def create_video_task(task_id, audio_url, original_request):
    # ... (code inside try block) ...
    try:
        local_audio_path = download_audio(audio_url)
        # ... (rest of your video creation logic) ...
        final_video_path = assemble_video(local_audio_path, images)
        upload_to_gcs(final_video_path, f"complete/{task_id}.mp4")
        logger.info(f"--- [CELERY WORKER] Successfully completed task {task_id} ---")

    except Exception as e: # This 'except' must be correctly indented
        logger.error(f"!!! [CELERY WORKER] FAILED task {task_id}: {e} !!!", exc_info=True)
    finally: # This 'finally' must be aligned with 'try' and 'except'
        # Rename the local audio file instead of deleting it
        if local_audio_path and os.path.exists(local_audio_path):
            new_audio_path = local_audio_path.replace('.mp3', '_DONE.mp3')
            os.rename(local_audio_path, new_audio_path)
            logger.info(f"Audio file renamed to: {new_audio_path}")

        # Clean up other temporary files (video and images)
        cleanup_files = [final_video_path] + images
        for p in cleanup_files:
            if p and os.path.exists(p): os.remove(p)
