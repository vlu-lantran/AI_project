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
const frameSlider = document.getElementById('frameSlider');
const currentFrameSpan = document.getElementById('currentFrame');
const totalFramesSpan = document.getElementById('totalFrames');

// Global variables to manage state and intervals
let resultsUpdateInterval = null;
let frameUpdateInterval = null;
let videoProcessingActive = false;

// Initial state setup
const initializeButtonStates = () => {
    startBtn.disabled = true;
    stopBtn.disabled = true;
    resumeBtn.disabled = true;
    deleteBtn.disabled = true;
    downloadBtn.disabled = true;
};

// Stop all running intervals and reset state
const resetAllIntervals = () => {
    if (resultsUpdateInterval) {
        clearInterval(resultsUpdateInterval);
        resultsUpdateInterval = null;
    }
    
    if (frameUpdateInterval) {
        clearInterval(frameUpdateInterval);
        frameUpdateInterval = null;
    }
    
    videoProcessingActive = false;
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
            
            // Reset frame slider and text
            frameSlider.value = 0;
            currentFrameSpan.textContent = '0';
            totalFramesSpan.textContent = '0';
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

// Update video information
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

const updateResults = async () => {
    // Chỉ fetch kết quả nếu video đang được xử lý
    if (!videoProcessingActive) return;
    
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

// Upload form submission
uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const formData = new FormData(uploadForm);
    const videoFile = formData.get('video');
    try {
        // Lưu đường dẫn video vào localStorage
        localStorage.setItem('lastUploadedVideoPath', videoFile.name);
        // Reset any previous intervals
        resetAllIntervals();

        const response = await fetch('/upload', { method: 'POST', body: formData });
        const result = await response.json();

        if (result.success) {
            videoStream.src = '/video_feed';
            updateButtonStates('videoUploaded');
            
            await updateVideoInfo();
            
            await fetch('/stop_processing');
            
            frameSlider.value = 0;
            currentFrameSpan.textContent = '0';
            
            await fetch(`/seek_frame?frame=0`);
            
            resultsContainer.style.display = 'none';
            showMessage('success', result.message);
        } else {
            showMessage('error', result.error);
        }
    } catch (error) {
        showMessage('error', 'Error uploading video: ' + error.message);
    }
});

// Frame update interval management
function startFrameUpdateInterval() {
    resetAllIntervals();
    
    videoProcessingActive = true;
    
    frameUpdateInterval = setInterval(async () => {
        if (!videoProcessingActive) {
            resetAllIntervals();
            return;
        }
        
        try {
            const response = await fetch('/get_current_frame');
            const data = await response.json();
            
            if (data.success) {
                frameSlider.value = data.current_frame;
                currentFrameSpan.textContent = data.current_frame;
            }
        } catch (error) {
            console.error('Error updating frame:', error);
            resetAllIntervals();
        }
    }, 500);
}

// Start Processing Button
startBtn.addEventListener('click', async () => {
    try {
        const response = await fetch('/start_processing');
        const result = await response.json();
        
        if (result.success) {
            videoProcessingActive = true; // Đánh dấu video đang được xử lý
            updateButtonStates('processing');
            showMessage('success', result.message);
            // Reset and start intervals
            // resetAllIntervals();
            resultsUpdateInterval = setInterval(updateResults, 1000);
            // startFrameUpdateInterval();
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
            videoProcessingActive = false; // Dừng xử lý video
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

// Thêm hàm kiểm tra trạng thái video
const checkVideoStatus = async () => {
    try {
        const response = await fetch('/check_video_status');
        const data = await response.json();
        return data.success && data.video_exists;
    } catch (error) {
        console.error('Error checking video status:', error);
        return false;
    }
};

// Sửa đổi hàm resume để kiểm tra và khôi phục video nếu cần
resumeBtn.addEventListener('click', async () => {
    try {
        // Kiểm tra trạng thái video trước
        const videoExists = await checkVideoStatus();
        
        if (!videoExists) {
            // Nếu không có video, thử khôi phục lại video cuối cùng
            const lastUploadedVideo = localStorage.getItem('lastUploadedVideoPath');
            
            if (lastUploadedVideo) {
                // Thực hiện upload lại video
                const formData = new FormData();
                formData.append('video', new File([], lastUploadedVideo));
                
                const uploadResponse = await fetch('/upload', { 
                    method: 'POST', 
                    body: formData 
                });
                const uploadResult = await uploadResponse.json();
                
                if (!uploadResult.success) {
                    throw new Error('Could not restore video');
                }
            } else {
                showMessage('error', 'No video available to resume');
                return;
            }
        }
        
        // Tiếp tục logic resume ban đầu
        const response = await fetch('/resume_processing');
        const result = await response.json();
        
        if (result.success) {
            updateButtonStates('processing');
            showMessage('success', result.message);
            
            // Reset and start intervals
            resetAllIntervals();
            
            videoProcessingActive = true;
            resultsUpdateInterval = setInterval(updateResults, 1000);
            startFrameUpdateInterval();
        } else {
            showMessage('error', result.error);
        }
    } catch (error) {
        showMessage('error', 'Error resuming detection: ' + error.message);
    }
});

// Delete Video Button
deleteBtn.addEventListener('click', async () => {
    try {
        // Stop all active processing and intervals
        resetAllIntervals();
        
        const response = await fetch('/delete_video', { method: 'POST' });
        const result = await response.json();

        if (result.success) {
            // Reset video stream and UI
            videoStream.src = '';
            updateButtonStates('initial');
            
            // Clear results table body and hide results container
            resultsTableBody.innerHTML = '';
            resultsContainer.style.display = 'none';
            
            // Reset processing-related state variables
            videoProcessingActive = false;
            
            // Clear the last uploaded video path from localStorage if needed
            localStorage.removeItem('lastUploadedVideoPath');
            
            showMessage('success', result.message);
        } else {
            showMessage('error', result.error || 'Unexpected response from server');
        }
    } catch (error) {
        console.error('Delete error:', error);
        showMessage('error', 'Error deleting video: ' + error.message);
    }
});

// Modify download button event listener
downloadBtn.addEventListener('click', async () => {
    try {
        // First, fetch video results to show more context
        const resultsResponse = await fetch('/get_video_results');
        const resultsData = await resultsResponse.json();
        
        if (resultsData.success) {
            // Show detailed download information
            const anomaliesCount = resultsData.anomalies.length;
            const duration = resultsData.duration;
            
            const confirmDownload = confirm(
                `Video Processing Summary:\n` +
                `- Total Duration: ${duration} seconds\n` +
                `- Anomalies Detected: ${anomaliesCount}\n` +
                `Do you want to download the processed video?`
            );
            
            if (confirmDownload) {
                // Direct download
                window.location.href = '/download_video';
                showMessage('success', `Downloading video with ${anomaliesCount} anomalies`);
            }
        } else {
            throw new Error(resultsData.error || 'Unable to retrieve video details');
        }
    } catch (error) {
        showMessage('error', 'Error downloading video: ' + error.message);
    }
});

// Frame Slider Event Listener
frameSlider.addEventListener('input', async (e) => {
    const frameNumber = parseInt(e.target.value);
    currentFrameSpan.textContent = frameNumber;
    
    try {
        const response = await fetch(`/seek_frame?frame=${frameNumber}`);
        const data = await response.json();
        
        if (data.success) {
            currentFrameSpan.textContent = data.current_frame;
            totalFramesSpan.textContent = data.total_frames;
        } else {
            showMessage('error', data.error || 'Error seeking frame');
            e.target.value = 0;
            currentFrameSpan.textContent = 0;
        }
    } catch (error) {
        showMessage('error', 'Error seeking frame: ' + error.message);
        e.target.value = 0;
        currentFrameSpan.textContent = 0;
    }
});

// Initialize page
initializeButtonStates();