import argparse
import json
import os
import time
from collections import defaultdict
import numpy as np

import cv2
import torch
from ultralytics import YOLO
import logging

def setup_logging(log_dir):
    """Set up logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'anomaly_detection.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def calculate_area_confidence(contour, img):
    """
    Calculate confidence based on contour characteristics
    
    Args:
        contour: OpenCV contour
        img: Original image
    
    Returns:
        float: Confidence score
    """
    img_area = img.shape[0] * img.shape[1]
    contour_area = cv2.contourArea(contour)
    
    # Area-based confidence
    area_ratio = contour_area / img_area
    
    # Compactness calculation
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    compactness = 1 - (contour_area / hull_area if hull_area > 0 else 1)
    
    # Weighted confidence score
    confidence = 0.6 * area_ratio + 0.4 * compactness
    return min(max(confidence, 0), 1)

def non_max_suppression(boxes, confidences, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression to reduce overlapping detections
    
    Args:
        boxes: List of bounding boxes
        confidences: Corresponding confidence scores
        iou_threshold: Intersection over Union threshold
    
    Returns:
        Filtered boxes and confidences
    """
    boxes = np.array(boxes)
    confidences = np.array(confidences)
    
    indices = np.argsort(confidences)[::-1]
    keep = []

    while indices.size > 0:
        current = indices[0]
        keep.append(current)
        
        ious = calculate_iou(boxes[current], boxes[indices[1:]])
        mask = ious <= iou_threshold
        indices = indices[1:][mask]
    
    return boxes[keep], confidences[keep]

def calculate_iou(box, boxes):
    """
    Calculate Intersection over Union between boxes
    
    Args:
        box: Reference box
        boxes: Array of comparison boxes
    
    Returns:
        IoU values
    """
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[0] + box[2], boxes[:, 0] + boxes[:, 2])
    y2 = np.minimum(box[1] + box[3], boxes[:, 1] + boxes[:, 3])
    
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    union = box[2] * box[3] + boxes[:, 2] * boxes[:, 3] - intersection
    
    return intersection / union

def anomaly_detect(source_dir, output_dir='output/', config=None):
    """
    Process all videos in a source directory and generate anomaly detection results
    
    Args:
        source_dir (str): Directory containing input videos
        output_dir (str): Directory to save output videos and results
        config (dict): Configuration parameters
    """
    logger = setup_logging(output_dir)
    
    config = config or {
        'weights': 'yolov8s-oiv7.pt',
        'imgsz': 640,
        'confidence_threshold': 0.5,
        'iou_threshold': 0.5
    }

    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)

    video_files = [
        f for f in os.listdir(source_dir) 
        if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    ]

    logger.info(f"Found {len(video_files)} videos to process")

    for video_filename in video_files:
        video_path = os.path.join(source_dir, video_filename)
        logger.info(f"Processing video: {video_filename}")

        video_output_dir = os.path.join(output_dir, os.path.splitext(video_filename)[0])
        os.makedirs(video_output_dir, exist_ok=True)

        try:
            _process_single_video(
                source=video_path, 
                save_dir=video_output_dir,
                config=config
            )
        except Exception as e:
            logger.error(f"Error processing {video_filename}: {e}")

    end_time = time.time()
    logger.info(f"Total processing time: {end_time - start_time:.2f} seconds")

def _process_single_video(source=0, save_dir='output/', config=None):
    """
    Process a single video for anomaly detection
    
    Args:
        source (str/int): Video source path or camera index
        save_dir (str): Directory to save results
        config (dict): Configuration parameters
    """
    config = config or {}
    logger = logging.getLogger(__name__)
    start_time = time.time()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load configuration with defaults
    weights = config.get('weights', 'yolov8s-oiv7.pt')
    imgsz = config.get('imgsz', 640)
    confidence_threshold = config.get('confidence_threshold', 0.5)
    iou_threshold = config.get('iou_threshold', 0.5)

    try:
        model = YOLO(weights)
        names = model.names
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None
    
    object_appearances = defaultdict(list)
    background_objects = {}

    # Camera handling
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    
    # Default to camera if no video path is provided
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False, varThreshold=50)

    dataset = cv2.VideoCapture(source)
    if not dataset.isOpened():
        print(f"Failed to load video or open camera at {source}")
        return

    total_frames = int(dataset.get(cv2.CAP_PROP_FRAME_COUNT)) if source != 0 else float('inf')
    fps = dataset.get(cv2.CAP_PROP_FPS) if source != 0 else 24
    #frame_skip_ratio = max(1, int(fps / 24))
    frame_skip_ratio = 1  # Process all frames

    # Get frame dimensions
    width = int(dataset.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(dataset.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_path = os.path.join(save_dir, 'output.mp4')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Detect background objects in the first frame
    ret, first_frame = dataset.read()
    if ret:
        background_results = model(first_frame, imgsz=imgsz)
        for result in background_results:
            boxes = result.boxes
            for box in boxes:
                obj_id = int(box.cls[0])
                obj_name = names[obj_id]
                background_objects[obj_name] = True

    frame_count = 0
    while True:
        ret, img = dataset.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip_ratio != 0:
            continue

        current_time = frame_count / fps

        # Show progress for non-camera sources
        if source != 0:
            progress_percentage = (frame_count / total_frames) * 100
            print(f"Processing: {progress_percentage:.2f}% complete", end='\r')

        # Apply background subtraction
        fgmask = fgbg.apply(img)
        bin_img = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)[1]
        contour_list, _ = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        anomalies_in_frame = []
        # Dynamic contour area threshold calculation
        img_area = img.shape[0] * img.shape[1]
        min_contour_threshold = 0.01 * img_area  # 1% of image area
        max_contour_threshold = 0.5 * img_area   # 50% of image area

        for contour in contour_list:
            contour_area = cv2.contourArea(contour)
            # Use dynamic thresholding instead of fixed value
            if min_contour_threshold < contour_area < max_contour_threshold:
                x, y, w, h = cv2.boundingRect(contour)
                # Estimate area and compactness confidence
                area_confidence = cv2.contourArea(contour) / (img.shape[0] * img.shape[1])
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                compactness = 1 - (area_confidence if hull_area == 0 else (cv2.contourArea(contour) / hull_area))
                confidence = (0.6 * area_confidence) + (0.4 * compactness)
                conf_score = round(min(max(confidence, 0), 1), 2)

                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(
                    img, f"Anomaly Detected: {conf_score}",
                    (x, y - 10 if y > 20 else y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                )

                anomalies_in_frame.append({
                    'Confidence': conf_score,
                    'Position': {'x': x, 'y': y, 'width': w, 'height': h}
                })

        # Detect objects in the current frame
        results = model(img, imgsz=imgsz)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                obj_id = int(box.cls[0])
                obj_name = names[obj_id]
                if obj_name in background_objects:
                    continue

                if not object_appearances[obj_id] or current_time - object_appearances[obj_id][-1]['Disappear'] > 1:
                    object_appearances[obj_id].append({'Appear': current_time, 'Disappear': None})

                object_appearances[obj_id][-1]['Disappear'] = current_time
        
        anomaly_results = []
        for obj_id, appearances in object_appearances.items():
            obj_name = names[obj_id]
            for idx, times in enumerate(appearances):
                anomaly_results.append({
                    'Object': obj_name,
                    'Appear': f"{times['Appear']:.2f}s" if times['Appear'] is not None else "Not detected",
                    'Disappear': f"{times['Disappear']:.2f}s" if times['Disappear'] is not None else "Not detected",
                    'Status': 'Abnormal',
                    'Appearance': f'Appearance {idx + 1}',
                    'Frame Anomalies': anomalies_in_frame
                })

        # Draw bbox detections
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                obj_id = int(box.cls[0])
                obj_name = names[obj_id]
                confidence = box.conf[0]
                if obj_name in background_objects or confidence < 0.5:
                    continue
                color = (0, 255, 0) if obj_name not in background_objects else (0, 0, 255)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                label = f"{obj_name} {confidence:.2f}"
                cv2.putText(
                    img, label,
                    (x1, y1 - 10 if y1 > 20 else y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                )

        img_resized = cv2.resize(img, (width, height))
        video_writer.write(img_resized)

    dataset.release()
    video_writer.release()

    json_output_path = os.path.join(save_dir, 'anomaly_results.json')
    with open(json_output_path, 'w') as json_file:
        json.dump(anomaly_results, json_file, indent=4)

    end_time = time.time()
    print(f"\nAnomaly Detection Results for video: {len(anomaly_results)} anomalies detected")
    print(f"Results saved to: {json_output_path}")
    print(f"Processing time: {end_time - start_time:.2f} seconds")

    return anomaly_results

def main():
    """Main execution entry point"""
    parser = argparse.ArgumentParser(description='Anomaly Detection Script')
    parser.add_argument('--source_dir', type=str, required=True, help='Directory containing input videos')
    parser.add_argument('--output_dir', type=str, default='output/', help='Directory to save output videos and results')
    args = parser.parse_args()

    try:
        with torch.no_grad():
            anomaly_detect(
                source_dir=args.source_dir, 
                output_dir=args.output_dir
            )
    except Exception as e:
        logging.error(f"Anomaly detection failed: {e}")

if __name__ == '__main__':
    main()
