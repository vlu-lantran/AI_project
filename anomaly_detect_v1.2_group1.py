import argparse
import time
import json 
from pathlib import Path
import cv2
import torch
from collections import defaultdict
from ultralytics import YOLO

def anomaly_detect():
    source, weights, imgsz = opt.source, opt.weights, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://')) or source.endswith('.mp4')

    save_dir = Path(opt.project) / opt.name
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLO(weights)
    names = model.names

    object_appearances = defaultdict(list)  # Track multiple appearances
    background_objects = {}

    if webcam or source.endswith('.mp4'):
        dataset = cv2.VideoCapture(source)
        if not dataset.isOpened():
            print(f"Failed to load video {source}")
            return
        fps = dataset.get(cv2.CAP_PROP_FPS)
        width = int(dataset.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(dataset.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video_path = str(save_dir / 'output.mp4')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        frame_count = 0
        ret, first_frame = dataset.read() # Detect static bg obj in first frames
        if ret:
            background_results = model(first_frame, imgsz=imgsz)
            for result in background_results:
                boxes = result.boxes
                for box in boxes:
                    obj_id = int(box.cls[0])
                    obj_name = names[obj_id]
                    # Mark objects in the first frame as background objects
                    background_objects[obj_name] = True

        while True:
            ret, img = dataset.read()
            if not ret: break
            frame_count += 1
            current_time = frame_count / fps
            results = model(img, imgsz=imgsz)

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    obj_id = int(box.cls[0])
                    obj_name = names[obj_id]
                    if obj_name in background_objects: continue # Ign bg obj
                    # Record appearance time
                    if not object_appearances[obj_id] or current_time - object_appearances[obj_id][-1]['Disappear'] > 1:
                        # New appearance
                        object_appearances[obj_id].append({'Appear': current_time, 'Disappear': None})
                    # Last disappearance time
                    object_appearances[obj_id][-1]['Disappear'] = current_time

            # Disappear times for objects not detected in current frame
            for obj_id, appearances in object_appearances.items():
                if appearances and current_time - appearances[-1]['Disappear'] > 1:
                    appearances[-1]['Disappear'] = appearances[-1]['Disappear']  # Finalize disappearance

            # Format the results for output after process current frame
            anomaly_results = []
            # Inside the loop for generating anomaly results
            for obj_id, appearances in object_appearances.items():
                obj_name = names[obj_id]
                for idx, times in enumerate(appearances):
                    appear_time = times['Appear']
                    disappear_time = times['Disappear']
                    status = 'Abnormal' # Abnormal when appear
                    anomaly_results.append({
                        'Object': obj_name,
                        'Appear': f"{appear_time:.2f}s" if appear_time is not None else "Not detected",
                        'Disappear': f"{disappear_time:.2f}s" if disappear_time is not None else "Not detected",
                        'Status': status,
                        'Appearance': f'Appearance {idx + 1}'  # Track
                    })

            for result in results: # bounding boxes 
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    obj_id = int(box.cls[0])
                    obj_name = names[obj_id]
                    # If obj is currently active based on its appear/disappear
                    for anomaly in anomaly_results:
                        if anomaly['Object'] == obj_name and \
                                float(anomaly['Appear'][:-1]) <= current_time <= float(anomaly['Disappear'][:-1]):
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0),2)
                            cv2.putText(img, f"{obj_name} ({anomaly['Status']})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            video_writer.write(img)

    else:  # Image process (single image)
        img = cv2.imread(source)
        results = model(img, imgsz=imgsz)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                obj_id = int(box.cls[0])
                obj_name = names[obj_id]
                if obj_name in background_objects: continue
                object_appearances[obj_id].append({'Appear': 0, 'Disappear': 0})

    if webcam or source.endswith('.mp4'):
        dataset.release()
        video_writer.release()

    # Save anomaly results to a JSON file
    json_output_path = save_dir / 'anomaly_results.json'
    with open(json_output_path, 'w') as json_file:
        json.dump(anomaly_results, json_file, indent=4)

    print(f"Anomaly Detection Results: {anomaly_results}")
    print(f"Results saved to: {json_output_path}")
    return anomaly_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--weights', type=str, default='yolov8s-oiv7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    opt = parser.parse_args()

    with torch.no_grad(): anomaly_detect()
