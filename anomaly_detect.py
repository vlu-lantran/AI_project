import argparse
import json
import os
import time
from collections import defaultdict

import cv2
import torch
from ultralytics import YOLO


def anomaly_detect(source=0, save_dir='output/', weights='yolov8s-oiv7.pt', imgsz=640):
    start_time = time.time()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLO(weights)
    names = model.names

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
    frame_skip_ratio = max(1, int(fps / 24))

    # Get frame dimensions
    width = int(dataset.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(dataset.get(cv2.CAP_PROP_FRAME_HEIGHT))
    new_height = 420
    new_width = int((new_height / height) * width)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video_path = os.path.join(save_dir, 'output.avi')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (new_width, new_height))

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

        for contour in contour_list:
            if cv2.contourArea(contour) > 17000:
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
                    'Appearance': f'Appearance {idx + 1}'
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

        img_resized = cv2.resize(img, (new_width, new_height))
        cv2.imshow('Anomaly Detection Stream', img_resized)
        video_writer.write(img_resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    dataset.release()
    video_writer.release()
    cv2.destroyAllWindows()

    json_output_path = os.path.join(save_dir, 'anomaly_results.json')
    with open(json_output_path, 'w') as json_file:
        json.dump(anomaly_results, json_file, indent=4)

    end_time = time.time()
    print(f"\nAnomaly Detection Results: {anomaly_results}")
    print(f"Results saved to: {json_output_path}")
    print(f"Total processing time: {end_time - start_time:.2f} seconds")

    return anomaly_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default=0, help='Video file path or camera index (default: 0 for camera)')
    args = parser.parse_args()

    with torch.no_grad():
        anomaly_detect(source=args.source)
