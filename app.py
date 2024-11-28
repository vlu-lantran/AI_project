from flask import Flask, render_template, request, Response, jsonify, send_file
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import threading
from queue import Queue
import logging
import time
from modules.license_plate_detector import LicensePlateDetector

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    try:
        processing_state['detector'] = LicensePlateDetector()
        processing_state['frame_count'] = 0
        
        # Khởi tạo video và thông tin
        processing_state['video'], video_info = processing_state['detector'].initialize_video(source)
        
        # Validate video capture
        if processing_state['video'] is None or not processing_state['video'].isOpened():
            raise ValueError("Could not open video file")
        
        # Capture first frame for preview
        success, first_frame = processing_state['video'].read()
        if not success:
            raise ValueError("Could not read first frame from video")
        
        # Reset video to beginning
        processing_state['video'].set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Get video properties safely
        processing_state['fps'] = video_info.get('fps', 24)
        processing_state['width'] = video_info.get('width', first_frame.shape[1])
        processing_state['height'] = video_info.get('height', first_frame.shape[0])
        
        # Safe resize calculation
        target_height = 420
        scale = target_height / processing_state['height']
        processing_state['new_height'] = target_height
        processing_state['new_width'] = int(processing_state['width'] * scale)
        
        # Ensure width is at least 1
        processing_state['new_width'] = max(1, processing_state['new_width'])
        
        # Resize first frame safely
        try:
            first_frame_resized = cv2.resize(first_frame, 
                                             (processing_state['new_width'], processing_state['new_height']), 
                                             interpolation=cv2.INTER_AREA)
        except Exception as resize_error:
            logger.error(f"Frame resize error: {resize_error}")
            first_frame_resized = first_frame
        
        # Encode first frame
        ret, jpeg = cv2.imencode('.jpg', first_frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 80])
        
        if ret:
            write_to_buffer(jpeg.tobytes())
        
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
        
    except Exception as e:
        logger.error(f"Video initialization error: {str(e)}")
        # Reset processing state on error
        processing_state['video'] = None
        processing_state['video_writer'] = None
        raise
    
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
            # Check video exists
            if processing_state['video'] is None:
                break
            
            # Mặc định không xử lý khi chưa start processing
            if not processing_state['is_processing']:
                # Lấy frame hiện tại mà không di chuyển
                current_frame_pos = processing_state['video'].get(cv2.CAP_PROP_POS_FRAMES)
                processing_state['video'].set(cv2.CAP_PROP_POS_FRAMES, current_frame_pos)
                success, frame = processing_state['video'].read()
                
                if not success:
                    # Nếu không thể đọc frame, trở về frame đầu tiên
                    processing_state['video'].set(cv2.CAP_PROP_POS_FRAMES, 0)
                    success, frame = processing_state['video'].read()
                
                # Resize và encode frame để giữ preview
                frame = cv2.resize(frame, 
                                   (processing_state['new_width'], processing_state['new_height']))
                ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                
                if ret:
                    # Xóa buffer cũ và thêm frame mới
                    while not processing_state['buffer'].empty():
                        try:
                            processing_state['buffer'].get_nowait()
                        except:
                            pass
                    write_to_buffer(jpeg.tobytes())
                
                # Thêm delay để giảm tải CPU và ngăn video chạy liên tục
                time.sleep(0.1)  # Điều chỉnh thời gian delay phù hợp
                continue
            
            # Phần code xử lý frame khi processing được giữ nguyên
            success, frame = processing_state['video'].read()
            if not success:
                processing_state['video'].set(cv2.CAP_PROP_POS_FRAMES, 0)
                time.sleep(0.1)  # Prevent tight loop
                continue
            
            processing_state['frame_count'] += 1
            if processing_state['frame_count'] % frame_skip_ratio != 0:
                continue
            
            current_time = processing_state['frame_count'] / processing_state['fps']
            process_frame(frame, current_time)
            
            # Nếu đang processing, cập nhật frame count
            if processing_state['is_processing']:
                current_frame = int(processing_state['video'].get(cv2.CAP_PROP_POS_FRAMES))
                
                # Thêm logic để tự động cập nhật frame slider thông qua một global variable hoặc queue
                # Ví dụ: gửi frame hiện tại qua một global message hoặc queue để JavaScript có thể cập nhật
            
    except Exception as e:
        logger.error(f"Error in process loop: {str(e)}")
    finally:
        # Đảm bảo cleanup được gọi khi loop kết thúc
        cleanup_resources()
        logger.info("Video processing loop terminated")

def cleanup_resources():
    """Dọn dẹp tài nguyên"""
    try:
        # Đảm bảo việc dừng luồng xử lý
        processing_state['is_running'] = False
  
        processing_state['anomaly_results'] = []  # Add this line
        # Giải phóng video
        if processing_state['video'] is not None:
            try:
                processing_state['video'].release()
            except Exception as video_release_error:
                logger.error(f"Error releasing video: {video_release_error}")
            processing_state['video'] = None
        
        # Giải phóng video writer
        if processing_state['video_writer'] is not None:
            try:
                processing_state['video_writer'].release()
            except Exception as writer_release_error:
                logger.error(f"Error releasing video writer: {writer_release_error}")
            processing_state['video_writer'] = None
        
        # Đóng luồng nếu tồn tại
        if processing_state['thread'] is not None:
            try:
                processing_state['thread'].join(timeout=2.0)
            except Exception as thread_join_error:
                logger.error(f"Error joining thread: {thread_join_error}")
            processing_state['thread'] = None
        
        # Reset các trạng thái khác
        processing_state['is_processing'] = False
        processing_state['frame_count'] = 0
        processing_state['anomaly_results'] = None
        processing_state['detector'] = None
        
        # Xóa file output nếu tồn tại
        if processing_state['output_path'] and os.path.exists(processing_state['output_path']):
            try:
                os.remove(processing_state['output_path'])
            except Exception as file_delete_error:
                logger.error(f"Error deleting output file: {file_delete_error}")
        processing_state['output_path'] = None
        
        # Xóa các file upload cũ
        try:
            for filename in os.listdir(app.config['UPLOAD_FOLDER']):
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as file_delete_error:
                    logger.error(f"Error deleting upload file {filename}: {file_delete_error}")
        except Exception as folder_error:
            logger.error(f"Error accessing upload folder: {folder_error}")
        
        # Clear buffer
        while not processing_state['buffer'].empty():
            try:
                processing_state['buffer'].get_nowait()
            except Exception:
                break
        
        # Reset video-related configuration
        processing_state['fps'] = None
        processing_state['width'] = None 
        processing_state['height'] = None
        processing_state['new_width'] = None
        processing_state['new_height'] = None
        
        logger.info("Resources cleaned up successfully")
    except Exception as e:
        logger.error(f"Unexpected error during cleanup: {str(e)}")

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
        # Cleanup tài nguyên trước khi upload
        cleanup_resources()
        
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
                
                # Save and initialize new video
                file.save(filepath)
                initialize_video_processing(filepath)
                
                # Start processing thread
                processing_state['is_running'] = True
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
                # Cleanup nếu có lỗi
                cleanup_resources()
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
        # Cleanup nếu có lỗi
        cleanup_resources()
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/delete_video', methods=['POST'])
def delete_video():
    try:
        # Gọi hàm cleanup_resources để giải phóng tài nguyên
        cleanup_resources()
        
        return jsonify({
            'success': True,
            'message': 'Video deleted successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Error deleting video: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
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
        # Reset video về frame 0 khi start detection
        processing_state['video'].set(cv2.CAP_PROP_POS_FRAMES, 0)
        processing_state['frame_count'] = 0
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
    
@app.route('/resume_processing')
def resume_processing():
    if processing_state['video'] is not None:
        processing_state['is_processing'] = True
        return jsonify({
            'success': True,
            'message': 'Anomaly detection resumed'
        })
    return jsonify({
        'success': False,
        'error': 'No video source'
    })

@app.route('/seek_frame')
def seek_frame():
    """Tìm đến một frame cụ thể"""
    frame_number = int(request.args.get('frame', 0))
    
    if processing_state['video'] is not None:
        try:
            # Get total frames to validate input
            total_frames = int(processing_state['video'].get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Validate frame number
            if frame_number < 0 or frame_number >= total_frames:
                return jsonify({
                    'success': False,
                    'error': f'Invalid frame number. Must be between 0 and {total_frames-1}'
                }), 400
            
            # Đặt con trỏ video đến frame được yêu cầu
            processing_state['video'].set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            processing_state['frame_count'] = frame_number
            
            # Đọc frame
            success, frame = processing_state['video'].read()
            
            if success:
                # Resize frame safely
                try:
                    frame = cv2.resize(frame, 
                                   (processing_state['new_width'], processing_state['new_height']),
                                   interpolation=cv2.INTER_AREA)
                except Exception as resize_error:
                    logger.error(f"Frame resize error: {resize_error}")
                
                # Encode frame
                ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                
                if ret:
                    # Clear old buffer
                    while not processing_state['buffer'].empty():
                        try:
                            processing_state['buffer'].get_nowait()
                        except Exception:
                            break
                    
                    # Add new frame to buffer
                    write_to_buffer(jpeg.tobytes())
                
                return jsonify({
                    'success': True,
                    'current_frame': frame_number,
                    'total_frames': total_frames
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Could not read frame'
                }), 500
            
        except Exception as e:
            logger.error(f"Error seeking frame: {str(e)}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    return jsonify({
        'success': False,
        'error': 'No video loaded'
    }), 404

@app.route('/get_video_info')
def get_video_info():
    """Lấy thông tin video"""
    if processing_state['video'] is not None:
        return jsonify({
            'success': True,
            'total_frames': int(processing_state['video'].get(cv2.CAP_PROP_FRAME_COUNT)),
            'fps': processing_state['fps']
        })
    
    return jsonify({
        'success': False,
        'error': 'No video loaded'
    })

@app.route('/get_current_frame')
def get_current_frame():
    """Lấy frame hiện tại của video"""
    if processing_state['video'] is not None:
        current_frame = int(processing_state['video'].get(cv2.CAP_PROP_POS_FRAMES))
        return jsonify({
            'success': True,
            'current_frame': current_frame
        })
    
    return jsonify({
        'success': False,
        'error': 'No video loaded'
    })

@app.route('/check_video_status')
def check_video_status():
    """Kiểm tra xem video có còn tồn tại không"""
    try:
        if processing_state['video'] is not None:
            # Kiểm tra xem video capture có còn hoạt động không
            if processing_state['video'].isOpened():
                return jsonify({
                    'success': True,
                    'video_exists': True
                })
        
        return jsonify({
            'success': False,
            'video_exists': False,
            'error': 'No video source available'
        }), 404
    except Exception as e:
        logger.error(f"Error checking video status: {str(e)}")
        return jsonify({
            'success': False,
            'video_exists': False,
            'error': str(e)
        }), 500

@app.route('/get_results')
def get_results():
    try:
        if processing_state['video'] is None:
            return jsonify({
                'success': False,
                'error': 'No video source'
            }), 404

        # Rest of the existing code remains the same
        if processing_state['video'] is not None:
            results = processing_state['anomaly_results'] or []
            
            return jsonify({
                'success': True,
                'results': results
            })
    except Exception as e:
        logger.error(f"Unexpected error in get_results: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Unexpected error retrieving results'
        }), 500

@app.route('/get_video_results')
def get_video_results():
    """Get comprehensive video processing results"""
    try:
        results = {
            'success': True,
            'anomalies': processing_state.get('anomaly_results', []),
            'total_frames': int(processing_state['video'].get(cv2.CAP_PROP_FRAME_COUNT)) if processing_state.get('video') else 0,
            'fps': processing_state.get('fps'),
            'duration': int(processing_state['video'].get(cv2.CAP_PROP_FRAME_COUNT) / processing_state.get('fps', 1)) if processing_state.get('video') else 0
        }
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error getting video results: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/download_video')
def download_video():
    try:
        # Ensure video writer is fully released
        if processing_state['video_writer'] is not None:
            processing_state['video_writer'].release()
        
        # Check if output video exists
        if processing_state['output_path'] and os.path.exists(processing_state['output_path']):
            # Generate more descriptive filename
            timestamp = int(time.time())
            anomaly_count = len(processing_state.get('anomaly_results', []))
            filename = f'detected_video_{timestamp}_anomalies_{anomaly_count}.avi'
            
            # Return file for download WITHOUT deleting
            return send_file(
                processing_state['output_path'], 
                as_attachment=True,
                download_name=filename,
                mimetype='video/x-msvideo'
            )
        else:
            return jsonify({
                'success': False,
                'error': 'No processed video available for download'
            }), 404
    except Exception as e:
        logger.error(f"Download video error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Error preparing download: {str(e)}'
        }), 500

@app.route('/reset_state')
def reset_state():
    """Reset trạng thái video về mặc định"""
    try:
        # Dừng xử lý
        processing_state['is_processing'] = False
        
        # Đặt lại vị trí frame về 0
        if processing_state['video'] is not None:
            processing_state['video'].set(cv2.CAP_PROP_POS_FRAMES, 0)
            processing_state['frame_count'] = 0
        
        return jsonify({
            'success': True,
            'message': 'Video state reset'
        })
    except Exception as e:
        logger.error(f"Error resetting state: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
