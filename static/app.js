document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('videoForm');
    const btn = document.getElementById('submitButton');
    const resultDiv = document.getElementById('result');
    const originalButtonText = 'Create My Music Video';

    form.addEventListener('submit', async e => {
        e.preventDefault();
        btn.disabled = true;
        btn.innerHTML = `<span class="loading-spinner" style="width: 1.2rem; height: 1.2rem; border-width: 2px; margin-right: 0.5rem;"></span>Creating...`;
        resultDiv.innerHTML = `<div class="result-card processing-card">...</div>`;

        try {
            const fd = new FormData(form);
            const payload = {
                mood: fd.get('mood'), age: fd.get('age'), pricing: fd.get('pricing'),
                length: fd.get('length'), artist: fd.get('artist'), vision: fd.get('vision')
            };
            
            const createResponse = await fetch('/create', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!createResponse.ok) throw new Error((await createResponse.json()).error);
            
            const { task_id } = await createResponse.json();
            const evtSource = new EventSource(`/events/${task_id}`);

            evtSource.addEventListener('complete', event => {
                const { message } = JSON.parse(event.data);
                resultDiv.innerHTML = `
                    <div class="result-card success-card">
                        <h3 class="result-title success-title"><i class="fas fa-check-circle"></i> Success!</h3>
                        <div class="result-content">
                            <p>${message}</p>
                            <p>Your video is being created and will appear in the public gallery shortly.</p>
                        </div>
                    </div>`;
                evtSource.close();
                btn.disabled = false;
                btn.innerHTML = `<i class="fas fa-magic"></i> ${originalButtonText}`;
            });

            evtSource.addEventListener('error', event => {
                // This handles errors from the web app (e.g., GCS handoff failed)
                const { message } = JSON.parse(event.data);
                resultDiv.innerHTML = `<div class="result-card error-card"><h3>Error</h3><p>${message}</p></div>`;
                evtSource.close();
                btn.disabled = false;
                btn.innerHTML = `<i class="fas fa-magic"></i> ${originalButtonText}`;
            });

        } catch (err) {
            resultDiv.innerHTML = `<div class="result-card error-card"><h3>Submission Failed</h3><p>${err.message}</p></div>`;
            btn.disabled = false;
            btn.innerHTML = `<i class="fas fa-magic"></i> ${originalButtonText}`;
        }
    });
});
