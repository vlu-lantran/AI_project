import torch
import os
import cv2
from ultralytics import YOLO
import json
from collections import defaultdict

os.environ['KMP_DUPLICATE_LIB_OK'] = 'true'
if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    print("Using CPU")


def process_video_realtime(video_path, model_path, output_video_path, output_json_path):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    frame_width, frame_height, fps = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create an object to save the output video
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

    # Dictionary to store object tracking information
    objects = defaultdict(lambda: {'count': 0, 'tracks': {}})
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model.track(frame, persist=True)

        # Process detections using Torch
        if results[0].boxes.id is not None:
            process_detections_with_torch(results, model, objects, frame, frame_count, fps)

        out.write(frame)

        # Display the frame with detections
        cv2.imshow("Realtime Object Detection", frame)

        # Stop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Update progress for tracking
        frame_count += 1
        progress = (frame_count / total_frames) * 100
        print(f"\rProcessing: {progress:.2f}% complete", end="")

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Save tracking results to a JSON file
    save_tracking_results(objects, output_json_path, fps)

def process_detections_with_torch(results, model, objects, frame, frame_count, fps):
    boxes = results[0].boxes.xyxy.cpu().numpy()
    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
    classes = results[0].boxes.cls.cpu().numpy().astype(int)
    confidences = results[0].boxes.conf.cpu().numpy()

    for box, track_id, cls, confidence in zip(boxes, track_ids, classes, confidences):
        class_name = model.names[cls]
        
        # Only process people and vehicles
        if class_name not in ['person', 'car', 'truck', 'bus', 'motorcycle']: continue

        # Update or initialize tracking information for the object
        if track_id not in objects[class_name]['tracks']:
            objects[class_name]['count'] += 1
            count = objects[class_name]['count']
            objects[class_name]['tracks'][track_id] = {
                'Object ID': f"{class_name.capitalize()} {count}",
                'Class': class_name,
                'Time_of_appearance': frame_count / fps,
                'Time_of_disappearance': frame_count / fps,
                'bounding_box': box.tolist(),
                'Confidence': float(confidence),
            }

        # Update the information for an already tracked object
        objects[class_name]['tracks'][track_id].update({
            'Time_of_disappearance': frame_count / fps,
            'bounding_box': box.tolist(),
            'Confidence': float(confidence),
        })

        # Draw a bounding box around the object
        draw_bbox(frame, box.tolist(), objects[class_name]['tracks'][track_id]['Object ID'])
        label = objects[class_name]['tracks'][track_id]['Object ID']

def draw_bbox(frame, box, label):
    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
    cv2.putText(frame, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

def save_tracking_results(objects, output_json_path, fps):
    json_results = [
        {
            "Object ID": info['Object ID'],
            "Class": info['Class'],
            "Time appeared": f"{int(info['Time_of_appearance'] // 60):02d}:{int(info['Time_of_appearance'] % 60):02d}",
            "Time disappeared": f"{int(info['Time_of_disappearance'] // 60):02d}:{int(info['Time_of_disappearance'] % 60):02d}",
            "Bounding box": info['bounding_box'],
            "Confidence": f"{info['Confidence'] * 100:.2f}%",
        }
        for class_name, data in objects.items()
        for track_id, info in data['tracks'].items()
    ]

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    video_path = 'video2.mp4' 
    model_path = 'yolov8n.pt' 
    output_video_path = 'video2_output.avi'  
    output_json_path = 'tracking_resultsvideo2.json'  

    process_video_realtime(video_path, model_path, output_video_path, output_json_path)
