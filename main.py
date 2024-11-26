import cv2
from paddleocr import PaddleOCR
from ultralytics import YOLO
import json
from datetime import datetime
import time
import argparse
import os
import logging
import re

# Logging setup
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='license_plate_debug.log')

# Argparse setup
parser = argparse.ArgumentParser(description='License Plate Detection')
parser.add_argument('--source', type=str, help='Path to input video', required=True)
parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
parser.add_argument('--confidence_threshold', type=float, default=0.5, help='OCR confidence threshold')
args = parser.parse_args()
        
# Input/output paths
input_video_path = args.source
output_directory = args.output_dir
confidence_threshold = args.confidence_threshold

output_video_path = os.path.join(output_directory, 'detected_video.mp4')
output_jsonl_path = os.path.join(output_directory, 'detected_plates.jsonl')

# Create output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
    
# Initialize models
plate_detector = YOLO('models/license_plate_detector.pt')
ocr = PaddleOCR(lang='en')

# Open input video
cap = cv2.VideoCapture(input_video_path)
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_input = int(cap.get(cv2.CAP_PROP_FPS))

# Calculate new dimensions for 420p
aspect_ratio = original_width / original_height
new_height = 420
new_width = int(new_height * aspect_ratio)

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps_input, (new_width, new_height))

# Tracking unique plates
unique_plates = {}

# FPS counter variables
frame_count = 0
start_time = time.time()
fps = 0

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

def save_to_jsonl(plate_data):
    """
    Save unique, valid license plates to JSONL
    """
    try:
        plate_text, plate_confidence = plate_data
        
        if is_valid_vietnamese_plate(plate_text):
            # Check if this plate is already recorded with higher confidence
            if plate_text not in unique_plates or plate_confidence > unique_plates[plate_text]['confidence']:
                with open(output_jsonl_path, 'a', encoding='utf-8') as f:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    data = {
                        "timestamp": timestamp,
                        "plate_number": plate_text,
                        "confidence": float(plate_confidence)
                    }
                    json.dump(data, f, ensure_ascii=False)
                    f.write('\n')
                
                # Update unique plates tracking
                unique_plates[plate_text] = {
                    'confidence': plate_confidence,
                    'timestamp': timestamp
                }
                
                logging.info(f"Saved plate: {plate_text} with confidence {plate_confidence}")
            else:
                logging.debug(f"Plate {plate_text} already exists with higher confidence")
        else:
            logging.debug(f"Plate {plate_text} did not pass validation")
    except Exception as e:
        logging.error(f"Error saving plate: {e}")

# Video processing
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Calculate actual FPS
    frame_count += 1
    if frame_count % 20 == 0:  # Update FPS every 20 frames
        end_time = time.time()
        elapsed_time = end_time - start_time
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()

    # Process every 5th frame
    if cap.get(cv2.CAP_PROP_POS_FRAMES) % 5 != 0:
        # Resize frame before writing
        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        out.write(resized_frame)
        continue

    # Detect license plates on original frame
    results_plate = plate_detector(frame)
    plate_boxes = results_plate[0].boxes

    logging.debug(f"Found {len(plate_boxes)} plate boxes")

    # Resize frame for output
    resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Display FPS on resized frame
    cv2.putText(resized_frame, f'FPS: {fps:.2f}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    for plate_box in plate_boxes:
        # Get coordinates on original frame
        px1, py1, px2, py2 = map(int, plate_box.xyxy[0])
        
        # Scale coordinates for resized frame
        scale_x = new_width / original_width
        scale_y = new_height / original_height
        
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
        if plate_confidence > 0.85:
            save_to_jsonl((plate_text, plate_confidence))

            # Draw bounding box and text on resized frame
            cv2.rectangle(resized_frame, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)
            cv2.putText(resized_frame, f'{plate_text} ({plate_confidence:.2f})', (rx1, ry1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Write resized frame to output video
    out.write(resized_frame)

    # Display video
    cv2.imshow('Video', resized_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

logging.info("License plate detection completed.")