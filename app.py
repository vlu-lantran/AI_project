from flask import Flask, render_template, request, Response, jsonify, send_file
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import threading
from queue import Queue
import logging
import time
from modules.anomaly_detection import AnomalyDetector

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tạo thư mục cần thiết
for folder in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER']]:
    if not os.path.exists(folder):
        os.makedirs(folder)

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

# Global state
processing_state = {
    'is_processing': False,
    'is_running': True,
    'frame_count': 0,
    'video': None,
    'video_writer': None,
    'detector': None,
    'buffer': Queue(maxsize=30),
    'anomaly_results': None,
    'output_path': None,
    'fps': None,
    'width': None, 
    'height': None,
    'new_width': None,
    'new_height': None,
    'thread': None
}

def write_to_buffer(frame):
    """Ghi frame vào buffer"""
    if processing_state['buffer'].full():
        try:
            processing_state['buffer'].get_nowait()
        except:
            pass
    processing_state['buffer'].put(frame)

def read_from_buffer():
    """Đọc frame từ buffer"""
    return processing_state['buffer'].get()

def initialize_video_processing(source):
    """Khởi tạo xử lý video"""
    processing_state['detector'] = AnomalyDetector()
    processing_state['frame_count'] = 0
    
    # Khởi tạo video và thông tin
    processing_state['video'], video_info = processing_state['detector'].initialize_video(source)
    processing_state['fps'] = video_info['fps']
    processing_state['width'] = video_info['width']
    processing_state['height'] = video_info['height']
    processing_state['new_height'] = 420
    processing_state['new_width'] = int((processing_state['new_height'] / processing_state['height']) * processing_state['width'])
    
    # Khởi tạo writer
    processing_state['output_path'] = os.path.join(
        app.config['OUTPUT_FOLDER'], 
        f'output_{int(time.time())}.avi'
    )
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    processing_state['video_writer'] = cv2.VideoWriter(
        processing_state['output_path'], 
        fourcc,
        processing_state['fps'],
        (processing_state['new_width'], processing_state['new_height'])
    )

def process_frame(frame, current_time):
    """Xử lý một frame"""
    if processing_state['is_processing']:
        processed_frame, processing_state['anomaly_results'] = processing_state['detector'].process_frame(
            frame, current_time)
    else:
        processed_frame = frame.copy()
    
    # Resize và encode frame
    processed_frame = cv2.resize(processed_frame, 
                               (processing_state['new_width'], processing_state['new_height']))
    processing_state['video_writer'].write(processed_frame)
    ret, jpeg = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    
    if ret:
        write_to_buffer(jpeg.tobytes())

def process_video_loop():
    """Loop chính để xử lý video"""
    try:
        frame_skip_ratio = max(1, int(processing_state['fps'] / 24))
        
        while processing_state['is_running']:
            success, frame = processing_state['video'].read()
            if not success:
                processing_state['video'].set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            processing_state['frame_count'] += 1
            if processing_state['frame_count'] % frame_skip_ratio != 0:
                continue
            
            current_time = processing_state['frame_count'] / processing_state['fps']
            process_frame(frame, current_time)
            
        cleanup_resources()
            
    except Exception as e:
        logger.error(f"Error in process loop: {str(e)}")
        processing_state['is_running'] = False

def cleanup_resources():
    """Dọn dẹp tài nguyên"""
    if processing_state['video'] is not None:
        processing_state['video'].release()
    if processing_state['video_writer'] is not None:
        processing_state['video_writer'].release()

def allowed_file(filename):
    """Kiểm tra file có được phép không"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_frames():
    """Generator cho video stream"""
    while True:
        try:
            frame = read_from_buffer()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        except:
            break

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'video' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No video file provided'
            }), 400
            
        file = request.files['video']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
            
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                # Cleanup existing resources
                cleanup_resources()
                
                # Save and initialize new video
                file.save(filepath)
                initialize_video_processing(filepath)
                
                # Start processing thread
                processing_state['thread'] = threading.Thread(target=process_video_loop)
                processing_state['thread'].daemon = True
                processing_state['thread'].start()
                
                return jsonify({
                    'success': True,
                    'message': 'Video uploaded successfully',
                    'filename': filename
                }), 200
                
            except Exception as e:
                logger.error(f"Error processing file: {str(e)}")
                return jsonify({
                    'success': False,
                    'error': f'Error processing file: {str(e)}'
                }), 500
        else:
            return jsonify({
                'success': False,
                'error': 'Unsupported file format'
            }), 400
            
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

# Thêm route mới để xóa video
@app.route('/delete_video', methods=['POST'])
def delete_video():
    try:
        # Dừng processing thread nếu đang chạy
        processing_state['is_running'] = False
        if processing_state['thread'] is not None:
            processing_state['thread'].join(timeout=1.0)
        
        # Cleanup tài nguyên
        cleanup_resources()
        
        # Reset trạng thái
        processing_state['is_processing'] = False
        processing_state['video'] = None
        processing_state['video_writer'] = None
        processing_state['detector'] = None
        processing_state['anomaly_results'] = None
        processing_state['output_path'] = None
        processing_state['thread'] = None
        
        # Xóa file output nếu tồn tại
        if processing_state['output_path'] and os.path.exists(processing_state['output_path']):
            os.remove(processing_state['output_path'])
        
        # Clear buffer
        while not processing_state['buffer'].empty():
            try:
                processing_state['buffer'].get_nowait()
            except:
                pass
                
        return jsonify({
            'success': True,
            'message': 'Video deleted successfully'
        })
        
    except Exception as e:
        logger.error(f"Error deleting video: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Error deleting video: {str(e)}'
        }), 500

@app.route('/video_feed')
def video_feed():
    if processing_state['video'] is None:
        return "No video source", 404
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_processing')
def start_processing():
    if processing_state['video'] is not None:
        processing_state['is_processing'] = True
        return jsonify({
            'success': True,
            'message': 'Anomaly detection started'
        })
    return jsonify({
        'success': False,
        'error': 'No video source'
    })

@app.route('/stop_processing')
def stop_processing():
    if processing_state['video'] is not None:
        processing_state['is_processing'] = False
        return jsonify({
            'success': True,
            'message': 'Detection stopped'
        })
    return jsonify({
        'success': False,
        'error': 'No video source'
    })

@app.route('/get_results')
def get_results():
    if processing_state['video'] is not None:
        return jsonify({
            'success': True,
            'results': processing_state['anomaly_results']
        })
    return jsonify({
        'success': False,
        'error': 'No video source'
    })

@app.route('/download_video')
def download_video():
    if processing_state['output_path'] and os.path.exists(processing_state['output_path']):
        return send_file(processing_state['output_path'], 
                        as_attachment=True,
                        download_name='detected_video.avi')
    return jsonify({
        'success': False,
        'error': 'No video available for download'
    })

if __name__ == '__main__':
    app.run(debug=True, threaded=True)