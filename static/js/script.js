// DOM Elements
const startBtn = document.getElementById('startProcessing');
const stopBtn = document.getElementById('stopProcessing');
const resumeBtn = document.getElementById('resumeProcessing');
const deleteBtn = document.getElementById('deleteVideo');
const downloadBtn = document.getElementById('downloadVideo');

const statusDiv = document.getElementById('processingStatus');
const errorDiv = document.getElementById('errorMessage');
const successDiv = document.getElementById('successMessage');
const uploadForm = document.getElementById('uploadForm');
const videoStream = document.getElementById('videoStream');
const resultsContainer = document.querySelector('.results-container');
const resultsTableBody = document.getElementById('resultsTableBody');
// Thêm các DOM elements mới
const frameSlider = document.getElementById('frameSlider');
const currentFrameSpan = document.getElementById('currentFrame');
const totalFramesSpan = document.getElementById('totalFrames');

// Initial state setup
const initializeButtonStates = () => {
    startBtn.disabled = true;
    stopBtn.disabled = true;
    resumeBtn.disabled = true;
    deleteBtn.disabled = true;
    downloadBtn.disabled = true;
};
// Hàm cập nhật thông tin video
const updateVideoInfo = async () => {
    try {
        const response = await fetch('/get_video_info');
        const data = await response.json();
        
        if (data.success) {
            frameSlider.max = data.total_frames - 1;
            totalFramesSpan.textContent = data.total_frames;
        }
    } catch (error) {
        console.error('Error getting video info:', error);
    }
};

// Utility functions
const showMessage = (type, message) => {
    if (type === 'error') {
        errorDiv.textContent = message;
        errorDiv.style.display = 'block';
        successDiv.style.display = 'none';
    } else {
        successDiv.textContent = message;
        successDiv.style.display = 'block';
        errorDiv.style.display = 'none';
    }

    setTimeout(() => {
        errorDiv.style.display = 'none';
        successDiv.style.display = 'none';
    }, 5000);
};

const updateButtonStates = (state) => {
    const states = {
        initial: () => {
            startBtn.disabled = true;
            stopBtn.disabled = true;
            resumeBtn.disabled = true;
            deleteBtn.disabled = true;
            downloadBtn.disabled = true;
            statusDiv.textContent = 'Status: Waiting for video';
            statusDiv.className = 'status idle';
        },
        videoUploaded: () => {
            startBtn.disabled = false;
            stopBtn.disabled = true;
            resumeBtn.disabled = true;
            deleteBtn.disabled = false;
            downloadBtn.disabled = true;
            statusDiv.textContent = 'Status: Video ready';
            statusDiv.className = 'status idle';
        },
        processing: () => {
            startBtn.disabled = true;
            stopBtn.disabled = false;
            resumeBtn.disabled = true;
            deleteBtn.disabled = false;
            downloadBtn.disabled = true;
            statusDiv.textContent = 'Status: Detection in progress';
            statusDiv.className = 'status processing';
        },
        stopped: () => {
            startBtn.disabled = false;
            stopBtn.disabled = true;
            resumeBtn.disabled = false;
            deleteBtn.disabled = false;
            downloadBtn.disabled = false;
            statusDiv.textContent = 'Status: Detection stopped';
            statusDiv.className = 'status idle';
        }
    };

    states[state]();
};

let resultsUpdateInterval = null;

// Results update function
const updateResults = async () => {
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
};

// Sửa uploadForm event listener
uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const formData = new FormData(uploadForm);

    try {
        const response = await fetch('/upload', { method: 'POST', body: formData });
        const result = await response.json();

        if (result.success) {
            // Start video stream
            videoStream.src = '/video_feed';
            updateButtonStates('videoUploaded');
            
            // Cập nhật thông tin video
            updateVideoInfo();
            
            // Reset processing state khi upload
            await fetch('/stop_processing');
            
            resultsContainer.style.display = 'none';
            showMessage('success', result.message);
        } else {
            showMessage('error', result.error);
        }
    } catch (error) {
        showMessage('error', 'Error uploading video: ' + error.message);
    }
});

// Sửa startBtn event listener
startBtn.addEventListener('click', async () => {
    const currentFrame = parseInt(frameSlider.value);
    
    try {
        const response = await fetch('/start_processing');
        const result = await response.json();
        
        if (result.success) {
            updateButtonStates('processing');
            showMessage('success', result.message);
            resultsUpdateInterval = setInterval(updateResults, 1000);
        } else {
            showMessage('error', result.error);
        }
    } catch (error) {
        showMessage('error', 'Error starting detection: ' + error.message);
    }
});

stopBtn.addEventListener('click', async () => {
    try {
        const response = await fetch('/stop_processing');
        const result = await response.json();
        
        if (result.success) {
            updateButtonStates('stopped');
            showMessage('success', result.message);
            
            if (resultsUpdateInterval) {
                clearInterval(resultsUpdateInterval);
            }
        } else {
            showMessage('error', result.error);
        }
    } catch (error) {
        showMessage('error', 'Error stopping detection: ' + error.message);
    }
});

resumeBtn.addEventListener('click', async () => {
    try {
        const response = await fetch('/resume_processing');
        const result = await response.json();
        
        if (result.success) {
            updateButtonStates('processing');
            showMessage('success', result.message);
            resultsUpdateInterval = setInterval(updateResults, 1000);
        } else {
            showMessage('error', result.error);
        }
    } catch (error) {
        showMessage('error', 'Error resuming detection: ' + error.message);
    }
});

deleteBtn.addEventListener('click', async () => {
    try {
        const response = await fetch('/delete_video', { method: 'POST' });
        
        // Kiểm tra xem response có phải là JSON không
        const contentType = response.headers.get('content-type');
        if (!contentType || !contentType.includes('application/json')) {
            throw new Error('Response is not JSON');
        }
        
        const result = await response.json();

        if (result.success) {
            videoStream.src = '';
            updateButtonStates('initial');
            resultsContainer.style.display = 'none';
            showMessage('success', result.message);
        } else {
            showMessage('error', result.error || 'Unexpected response from server');
        }
    } catch (error) {
        console.error('Delete error:', error);
        showMessage('error', 'Error deleting video: ' + error.message);
    }
});

downloadBtn.addEventListener('click', async () => {
    try {
        window.location.href = '/download_video';
        showMessage('success', 'Download started');
    } catch (error) {
        showMessage('error', 'Error downloading video: ' + error.message);
    }
});

// Thêm event listener cho frame slider
frameSlider.addEventListener('input', async (e) => {
    const frameNumber = parseInt(e.target.value);
    currentFrameSpan.textContent = frameNumber;
    
    try {
        const response = await fetch(`/seek_frame?frame=${frameNumber}`);
        const data = await response.json();
        
        if (data.success) {
            // Cập nhật thông tin frame
            currentFrameSpan.textContent = data.current_frame;
            totalFramesSpan.textContent = data.total_frames;
        } else {
            showMessage('error', data.error || 'Error seeking frame');
            // Optionally reset slider to a valid position
            e.target.value = 0;
            currentFrameSpan.textContent = 0;
        }
    } catch (error) {
        showMessage('error', 'Error seeking frame: ' + error.message);
        // Optionally reset slider
        e.target.value = 0;
        currentFrameSpan.textContent = 0;
    }
});

// Initialize page
initializeButtonStates();
