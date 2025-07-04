document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('videoForm');
    const btn = document.getElementById('submitButton');
    const resultDiv = document.getElementById('result');
    const originalButtonText = 'Create My Music Video';

    form.addEventListener('submit', async e => {
        e.preventDefault();
        btn.disabled = true;
        btn.innerHTML = `<span class="loading-spinner"></span>Creating...`;
        
        try {
            const fd = new FormData(form);
            const payload = {
                mood: fd.get('mood'),
                age: fd.get('age'),
                pricing: fd.get('pricing'),
                length: parseInt(fd.get('length'), 10),  // now numeric seconds
                artist: fd.get('artist'),
                vision: fd.get('vision')
            };
            
            const createResponse = await fetch('/create', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const data = await createResponse.json();
            if (!createResponse.ok) throw new Error(data.error);
            
            resultDiv.innerHTML = `
                <div class="result-card success-card">
                    <h3 class="result-title success-title"><i class="fas fa-check-circle"></i> Success!</h3>
                    <div class="result-content">
                        <p>${data.message}</p>
                        <p>Please check your Google Cloud Storage 'complete' folder for the final ${payload.length}s video in a few minutes.</p>
                    </div>
                </div>`;

        } catch (err) {
            resultDiv.innerHTML = `<div class="result-card error-card"><h3>Error</h3><p>${err.message}</p></div>`;
        } finally {
            btn.disabled = false;
            btn.innerHTML = `<i class="fas fa-magic"></i> ${originalButtonText}`;
        }
    });
});


