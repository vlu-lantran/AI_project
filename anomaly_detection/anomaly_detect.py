import cv2
import torch
from collections import defaultdict
from ultralytics import YOLO
import argparse
import time
import json
from pathlib import Path


def anomaly_detect(source, save_dir, input_type, weights='yolov8s-oiv7.pt', imgsz=640):
    # source, weights, imgsz = opt.source, opt.weights, opt.img_size
    # save_img = not opt.nosave and not source.endswith('.txt')
    # webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://')) or source.endswith('.mp4')
    # save_dir = Path(opt.project) / opt.name
    # save_dir.mkdir(parents=True, exist_ok=True)
    
    # LT: Restructure input definition
    if input_type == "video":
        webcam = True
    elif input_type == "img":
        save_img = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #LT: Un use variable?
    model = YOLO(weights)
    names = model.names

    object_appearances = defaultdict(list) # Track multiple appearances
    background_objects = {}

    # Detect fluctuations of environments
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False, varThreshold=50)
    if webcam or source.endswith('.mp4'):
        dataset = cv2.VideoCapture(source)
        if not dataset.isOpened():
            print(f"Failed to load video {source}")
            return
        fps = dataset.get(cv2.CAP_PROP_FPS)
        width = int(dataset.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(dataset.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video_path = save_dir + 'output.mp4'
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

            # Detect fluctuations of environments
            fgmask = fgbg.apply(img)
            bin_img = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)[1]
            contour_list, _ = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for i in range(len(contour_list)):
                if cv2.contourArea(contour_list[i]) > 17000:
                    x, y, w, h = cv2.boundingRect(contour_list[i])
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    font_scale = 0.6
                    thickness = 2
                    text_size = cv2.getTextSize("Anomaly Detected", cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                    text_x = x
                    text_y = y - 10 if y - 10 > 10 else y + text_size[1] + 10
                    cv2.putText(img, "Anomaly Detected", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 1, lineType=cv2.LINE_AA)
                    cv2.putText(img, "Anomaly Detected", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)
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
                if appearances and current_time - appearances[-1]['Disappear'] > 1: appearances[-1]['Disappear'] = appearances[-1]['Disappear']  # Finalize disappearance

            anomaly_results = [] # results output after process current frame
            # Loop for generating anomaly results
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
                        'Appearance': f'Appearance {idx + 1}'}) # Track

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get bounding box coordinates and class ID
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # coordinates
                    obj_id = int(box.cls[0])                # Object class ID
                    obj_name = names[obj_id]                # Object name
                    confidence = box.conf[0]                # Confidence score
                    
                    # Skip background objects and low-confidence detections
                    if obj_name in background_objects or confidence < 0.5: continue

                    # Calculate bounding box dimensions and ratio
                    width = x2 - x1
                    height = y2 - y1
                    bbox_area = width * height
                    frame_area = img.shape[0] * img.shape[1]
                    bbox_ratio = bbox_area / frame_area

                    # Check if obj is active based on its appear/disappear times
                    is_active = False
                    status_label = "Normal"  # Default status

                    for anomaly in anomaly_results:
                        # Ensure proper float conversion for matching times
                        appear_time = float(anomaly['Appear'][:-1]) if anomaly['Appear']!= "Not detected" else None
                        disappear_time = float(anomaly['Disappear'][:-1]) if anomaly['Disappear']!= "Not detected" else None
                        
                        if anomaly['Object'] == obj_name: # Match obj name and current time w abnormal result times
                            if appear_time is not None and disappear_time is not None:
                                # Debug print to check matching process
                                print(f"Checking {obj_name} | Appear: {appear_time} | Disappear: {disappear_time} | Current time: {current_time}")
                            
                            # Ensure current time is within the anomaly's appearance/disappearance window
                            if appear_time is not None and disappear_time is not None and appear_time <= current_time <= disappear_time:
                                is_active = True
                                status_label = anomaly['Status']
                                break  # If found a match, no need to check further anomalies for this obj

                    if not is_active: continue # Skip obj not currently active
                    # label obj name, confidence score, status, bbox ratio
                    label = f"{obj_name} {confidence:.2f} [{bbox_ratio:.2%}] ({status_label})"
                    # bounding box color based on the status
                    color = (0, 255, 0) if status_label == "Normal" else (0, 0, 255)  # Green for "Normal", Red for others
                    # Draw the bounding box
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)
                    # Adjust the font size and position for clarity
                    font_scale = 0.6  # avoid overlapping text
                    thickness = 2
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                    text_x = x1
                    text_y = y1 - 10 if y1 - 10 > 10 else y1 + text_size[1] + 10  # Ensure text is inside the frame
                    # Add shadow text for bbox
                    cv2.putText(img, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 1, lineType=cv2.LINE_AA)
                    cv2.putText(img, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)
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
    json_output_path = save_dir + 'anomaly_results.json'
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
    with torch.no_grad(): anomaly_detect(opt.source, opt.weights, opt.img_size)

                 

