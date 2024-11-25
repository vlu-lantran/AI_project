
const startBtn = document.getElementById('startProcessing');
const stopBtn = document.getElementById('stopProcessing');
const downloadBtn = document.getElementById('downloadVideo');
const statusDiv = document.getElementById('processingStatus');
const errorDiv = document.getElementById('errorMessage');
const successDiv = document.getElementById('successMessage');
const uploadForm = document.getElementById('uploadForm');
const videoStream = document.getElementById('videoStream');
const resultsContainer = document.querySelector('.results-container');
const resultsTableBody = document.getElementById('resultsTableBody');
const deleteBtn = document.getElementById('deleteVideo');

// Thêm vào danh sách các button đã có
deleteBtn.disabled = true;

let resultsUpdateInterval = null;

function showError(message) {
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
    successDiv.style.display = 'none';
    setTimeout(() => {
        errorDiv.style.display = 'none';
    }, 5000);
}

function showSuccess(message) {
    successDiv.textContent = message;
    successDiv.style.display = 'block';
    errorDiv.style.display = 'none';
    setTimeout(() => {
        successDiv.style.display = 'none';
    }, 5000);
}

async function updateResults() {
    try {
        const response = await fetch('/get_results');
        const data = await response.json();
        
        if (data.success && data.results.length > 0) {
            resultsContainer.style.display = 'block';
            resultsTableBody.innerHTML = data.results.map(result => `
                <tr>
                    <td>${result.Object}</td>
                    <td>${result.Appear}</td>
                    <td>${result.Disappear}</td>
                    <td>${result.Status}</td>
                    <td>${result.Appearance}</td>
                </tr>
            `).join('');
        }
    } catch (error) {
        console.error('Error fetching results:', error);
    }
}

uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const formData = new FormData(uploadForm);

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (result.success) {
            videoStream.src = '/video_feed';
            startBtn.disabled = false;
            stopBtn.disabled = true;
            downloadBtn.disabled = true;
            deleteBtn.disabled = false;  // Enable nút delete khi upload thành công
            statusDiv.textContent = 'Status: Video ready';
            statusDiv.className = 'status idle';
            resultsContainer.style.display = 'none';
            showSuccess(result.message);
        } else {
            showError(result.error);
        }
    } catch (error) {
        showError('Error uploading video: ' + error.message);
    }
});

startBtn.addEventListener('click', async () => {
    try {
        const response = await fetch('/start_processing');
        const result = await response.json();
        
        if (result.success) {
            startBtn.disabled = true;
            stopBtn.disabled = false;
            downloadBtn.disabled = true;
            statusDiv.textContent = 'Status: Detection in progress';
            statusDiv.className = 'status processing';
            showSuccess(result.message);
            
            // Start updating results
            resultsUpdateInterval = setInterval(updateResults, 1000);
        } else {
            showError(result.error);
        }
    } catch (error) {
        showError('Error starting detection: ' + error.message);
    }
});

stopBtn.addEventListener('click', async () => {
    try {
        const response = await fetch('/stop_processing');
        const result = await response.json();
        
        if (result.success) {
            startBtn.disabled = false;
            stopBtn.disabled = true;
            downloadBtn.disabled = false;
            statusDiv.textContent = 'Status: Detection stopped';
            statusDiv.className = 'status idle';
            showSuccess(result.message);
            
            // Stop updating results
            if (resultsUpdateInterval) {
                clearInterval(resultsUpdateInterval);
            }
        } else {
            showError(result.error);
        }
    } catch (error) {
        showError('Error stopping detection: ' + error.message);
    }
});

downloadBtn.addEventListener('click', async () => {
    try {
        window.location.href = '/download_video';
        showSuccess('Download started');
    } catch (error) {
        showError('Error downloading video: ' + error.message);
    }
});
