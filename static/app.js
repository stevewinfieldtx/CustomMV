document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('videoForm');
    const btn = document.getElementById('submitButton');
    const resultDiv = document.getElementById('result');
    const originalButtonText = 'Create My Music Video';

    form.addEventListener('submit', async e => {
        e.preventDefault();
        btn.disabled = true;
        btn.innerHTML = `<span class="loading-spinner" style="width: 1.2rem; height: 1.2rem; border-width: 2px; margin-right: 0.5rem;"></span>Creating...`;
        
        resultDiv.innerHTML = `
            <div class="result-card processing-card">
                <h3 class="result-title processing-title">
                    <div class="loading-spinner"></div>
                    Generating...
                </h3>
                <div class="result-content" id="status-message">
                    Requesting music generation...
                </div>
            </div>`;
            
        const statusMessage = document.getElementById('status-message');

        const fd = new FormData(form);
        const payload = {
            mood: fd.get('mood'),
            age: fd.get('age'),
            pricing: fd.get('pricing'),
            length: fd.get('length'),
            artist: fd.get('artist'),
            vision: fd.get('vision')
        };

        try {
            const createResponse = await fetch('/create', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!createResponse.ok) {
                const errData = await createResponse.json();
                throw new Error(errData.error || 'Failed to start the creation process.');
            }
            
            const { task_id } = await createResponse.json();
            statusMessage.textContent = 'Music task started. Waiting for completion to create video...';
            
            const evtSource = new EventSource(`/events/${task_id}`);

            evtSource.addEventListener('complete', event => {
                const { audio_url, video_url } = JSON.parse(event.data);
                resultDiv.innerHTML = `
                    <div class="result-card success-card">
                        <h3 class="result-title success-title"><i class="fas fa-check-circle"></i> Creation Successful!</h3>
                        <div class="result-content">
                            <p>Your AI-generated music and video are ready.</p>
                             <audio controls src="${audio_url}" style="width: 100%; margin-top: 1rem;"></audio>
                             <video controls src="${video_url}" style="width: 100%; margin-top: 1rem;"></video>
                             <br>
                            <a href="${video_url}" target="_blank" class="video-link" style="margin-top: 1rem;">
                                <i class="fas fa-play-circle"></i> View Video in New Tab
                            </a>
                        </div>
                    </div>`;
                btn.disabled = false;
                btn.innerHTML = `<i class="fas fa-magic"></i> ${originalButtonText}`;
                evtSource.close();
            });

            evtSource.addEventListener('error', event => {
                let message = "An unknown error occurred.";
                try {
                    const data = JSON.parse(event.data);
                    message = data.message || message;
                } catch(e) {
                    console.error("Could not parse error event data:", event.data);
                }

                resultDiv.innerHTML = `
                    <div class="result-card error-card">
                        <h3 class="result-title error-title"><i class="fas fa-times-circle"></i> An Error Occurred</h3>
                        <div class="result-content">
                            <p>${message}</p>
                            <button class="try-again-button" onclick="location.reload()">Try Again</button>
                        </div>
                    </div>`;
                btn.disabled = false;
                btn.innerHTML = `<i class="fas fa-magic"></i> ${originalButtonText}`;
                evtSource.close();
            });

        } catch (err) {
            console.error(err);
            resultDiv.innerHTML = `
                <div class="result-card error-card">
                    <h3 class="result-title error-title"><i class="fas fa-exclamation-triangle"></i> Submission Failed</h3>
                    <div class="result-content">
                        <p>${err.message}</p>
                        <button class="try-again-button" onclick="location.reload()">Try Again</button>
                    </div>
                </div>`;
            btn.disabled = false;
            btn.innerHTML = `<i class="fas fa-magic"></i> ${originalButtonText}`;
        }
    });
});