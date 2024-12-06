import cv2
from paddleocr import PaddleOCR
from ultralytics import YOLO
import json
import time
import argparse
import os
import logging
import re
import sys

# Logging setup
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='license_plate_debug.log')

# Argparse setup
parser = argparse.ArgumentParser(description='License Plate Detection')
parser.add_argument('--source', type=str, help='Path to input video', required=True)
parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
parser.add_argument('--confidence_threshold', type=float, default=0.5, help='OCR confidence threshold')
parser.add_argument('--process_every_n_frame', type=int, default=3, help='Process every n-th frame')
parser.add_argument('--skip_frames', type=int, default=0, help='Number of initial frames to skip')
args = parser.parse_args()
        
# Input/output paths
input_video_path = args.source
output_directory = args.output_dir
confidence_threshold = args.confidence_threshold
process_every_n_frame = max(1, args.process_every_n_frame)  # Minimum 1
skip_frames = max(0, args.skip_frames)
video_name = os.path.splitext(os.path.basename(input_video_path))[0]
output_video_path = os.path.join(output_directory, f'{video_name}_out_plate.mp4')
output_jsonl_path = os.path.join(output_directory, f'{video_name}_out_plate.js')

# Create output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
    
# Initialize models
try:
    plate_detector = YOLO('models/license_plate_detector.pt')
    ocr = PaddleOCR(lang='en')
except Exception as e:
    logging.error(f"Error initializing models: {e}")
    sys.exit(1)

# Open input video
cap = cv2.VideoCapture(input_video_path)

# Check if video opened successfully
if not cap.isOpened():
    logging.error(f"Cannot open video file: {input_video_path}")
    sys.exit(1)

# Get video properties with error handling
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
fps_input = int(cap.get(cv2.CAP_PROP_FPS) or 30 )
# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'X264')
out = cv2.VideoWriter(output_video_path, fourcc, fps_input, (original_width, original_height))

# Tracking unique plates
unique_plates = {}

# FPS counter variables
frame_count = 0
processed_frame_count = 0
start_time = time.time()
fps = 0

# Skip initial frames if specified
for _ in range(skip_frames):
    cap.read()

# Regex pattern for valid Vietnamese license plates
VALID_PLATE_PATTERN = re.compile(r'^(\d{2}[-][A-Z0-9]{4,6})$')

def is_valid_vietnamese_plate(plate_text):
    """
    Validate Vietnamese license plate format
    """
    logging.debug(f"Validating plate: {plate_text}")
    
    # Check using regex pattern
    if VALID_PLATE_PATTERN.match(plate_text):
        return True
    
    logging.debug(f"Invalid plate format: {plate_text}")
    return False

def save_to_jsonl(plate_data, frame_timestamp):
    """
    Save unique, valid license plates to JSONL with video frame timestamp
    """
    try:
        plate_text, plate_confidence = plate_data
        
        if is_valid_vietnamese_plate(plate_text):
            # Check if this plate is already recorded with higher confidence
            if plate_text not in unique_plates or plate_confidence > unique_plates[plate_text]['confidence']:
                with open(output_jsonl_path, 'a', encoding='utf-8') as f:
                    data = {
                        "video_timestamp": frame_timestamp,
                        "plate_number": plate_text,
                        "confidence": float(plate_confidence)
                    }
                    json.dump(data, f, ensure_ascii=False)
                    f.write('\n')
                
                # Update unique plates tracking
                unique_plates[plate_text] = {
                    'confidence': plate_confidence,
                    'timestamp': frame_timestamp
                }
                
                logging.info(f"Saved plate: {plate_text} with confidence {plate_confidence}")
            else:
                logging.debug(f"Plate {plate_text} already exists with higher confidence")
        else:
            logging.debug(f"Plate {plate_text} did not pass validation")
    except Exception as e:
        logging.error(f"Error saving plate: {e}")

# Video processing
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate actual FPS
        frame_count += 1
        if frame_count % 24 == 0:  # Update FPS every 20 frames
            end_time = time.time()
            elapsed_time = end_time - start_time
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()

        # Get current frame timestamp in seconds
        current_frame_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        # Process every n-th frame
        if processed_frame_count % process_every_n_frame != 0:
            # Resize frame before writing
            resized_frame = cv2.resize(frame, (original_width, original_height), interpolation=cv2.INTER_AREA)
            out.write(resized_frame)
            processed_frame_count += 1
            continue

        # Detect license plates on original frame
        results_plate = plate_detector(frame)
        plate_boxes = results_plate[0].boxes

        logging.debug(f"Found {len(plate_boxes)} plate boxes")

        # Resize frame for output
        resized_frame = cv2.resize(frame, (original_width, original_height), interpolation=cv2.INTER_AREA)

        # Display FPS on resized frame
        cv2.putText(resized_frame, f'FPS: {fps:.2f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        for plate_box in plate_boxes:
            # Get coordinates on original frame
            px1, py1, px2, py2 = map(int, plate_box.xyxy[0])
            
            # Scale coordinates for resized frame
            scale_x = original_width / original_width
            scale_y = original_height / original_height
            
            rx1 = int(px1 * scale_x)
            ry1 = int(py1 * scale_y)
            rx2 = int(px2 * scale_x)
            ry2 = int(py2 * scale_y)

            # Extract plate from original frame
            plate_img = frame[py1:py2, px1:px2]

            # OCR
            ocr_result = ocr.ocr(plate_img, cls=False)
            plate_text = ''
            plate_confidence = 0.0

            if ocr_result and ocr_result[0]:
                for line in ocr_result[0]:
                    text, confidence = line[1]
                    plate_text += text
                    plate_confidence += confidence
                plate_confidence /= len(ocr_result[0])
                
                logging.debug(f"OCR Result: {plate_text}, Confidence: {plate_confidence}")
            else:
                logging.debug("No OCR result found")
                continue

            # Save high-confidence plates with deduplication
            if plate_confidence > 0.95:
                save_to_jsonl((plate_text, plate_confidence), current_frame_timestamp)

                # Draw bounding box and text on resized frame
                cv2.rectangle(resized_frame, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)
                cv2.putText(resized_frame, f'{plate_text} ({plate_confidence:.2f})', (rx1, ry1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else: 
                cv2.rectangle(resized_frame, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)
                cv2.putText(resized_frame, f'{plate_text} ({plate_confidence:.2f})', (rx1, ry1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        # Write resized frame to output video
        out.write(resized_frame)

        # Display video
        cv2.imshow('Video', resized_frame)
        processed_frame_count += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    logging.error(f"Unexpected error during video processing: {e}")
finally:
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

logging.info("License plate detection completed.")
print("License plate detection completed successfully.")