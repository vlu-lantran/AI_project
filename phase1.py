import torch
import os
import cv2
from ultralytics import YOLO
import json
import time
from collections import defaultdict
import argparse

# Check if GPU or CPU is being used
os.environ['KMP_DUPLICATE_LIB_OK'] = 'true'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def process_video(video_path, model_path, output_video_path, output_json_path):
    # Load the YOLO model
    model = YOLO(model_path).to(device)
    model.overrides['imgsz'] = 1088

    # Open the video and extract properties
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    input_fps = int(cap.get(cv2.CAP_PROP_FPS))
    input_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Prepare to write output video
    output_resolution = (input_width, input_height)
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), input_fps, output_resolution)
    
    # Data structure to store tracking information
    objects = defaultdict(lambda: {'count': 0, 'tracks': {}})
    frame_count = 0

    print("Processing video...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO model to detect and track objects
        results = model.track(frame, persist=True, iou=0.4)
        frame = cv2.GaussianBlur(frame, (5, 5), 0)  # Làm mượt để giảm nhiễu
        frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)  # Tăng độ tương phản
        frame = cv2.medianBlur(frame, 5)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)  # Track IDs
            classes = results[0].boxes.cls.cpu().numpy().astype(int)  # Class IDs
            confidences = results[0].boxes.conf.cpu().numpy()  # Confidence scores
            confidences = results[0].boxes.conf.cpu().numpy()

            for box, track_id, cls, confidence in zip(boxes, track_ids, classes, confidences):
                class_name = model.names[cls]
                if class_name not in ['person', 'car', 'truck', 'bus', 'motorcycle']:
                    continue

                # Calculate appearance and disappearance times
                current_time = frame_count / input_fps  # Time in seconds

                if track_id not in objects[class_name]['tracks']:
                    objects[class_name]['count'] += 1
                    count = objects[class_name]['count']
                    objects[class_name]['tracks'][track_id] = {
                        'Object ID': f"{class_name.capitalize()} {count}",
                        'Class': class_name,
                        'Time_of_appearance': current_time,
                        'Time_of_disappearance': current_time,
                        'bounding_box': box.tolist(),
                        'Confidence': float(confidence),
                    }
                else:
                    objects[class_name]['tracks'][track_id]['Time_of_disappearance'] = current_time

                # Draw bounding boxes and labels on the frame
                x1, y1, x2, y2 = map(int, box)
                object_id = objects[class_name]['tracks'][track_id]['Object ID']
                label = f"{object_id} {confidence:.2f}"

                # Tăng kích thước font chữ và độ dày
                font_scale = 0.8  # Kích thước chữ lớn hơn
                font_thickness = 2  # Độ dày chữ
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]

                # Tính vị trí hiển thị chữ
                text_x = x1
                text_y = y1 - 10 if y1 - 10 > 10 else y1 + 10  # Đảm bảo chữ không ra ngoài khung hình

                # Vẽ nhãn
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Vẽ khung đối tượng
                cv2.rectangle(frame, (text_x - 2, text_y - text_size[1] - 4),
                              (text_x + text_size[0] + 2, text_y + 4), (0, 255, 0), -1)  # Nền cho nhãn
                cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale, (0, 0, 0), font_thickness)  # Chữ màu đen trên nền xanh

        # Write processed frame to output video
        out.write(frame)
        frame_count += 1

        # Progress update
        print(f"\rProcessing: {frame_count / input_total_frames * 100:.2f}% complete", end="")

    # Release resources
    cap.release()
    out.release()

    # Save results to JSON
    json_results = [
        {
            "Object ID": info['Object ID'],
            "Class": info['Class'],
            "Time appeared (seconds)": round(info['Time_of_appearance'], 2),
            "Time disappeared (seconds)": round(info['Time_of_disappearance'], 2),
            "Bounding box": info['bounding_box'],
            "Confidence": f"{info['Confidence'] * 100:.2f}%",
        }
        for class_name, data in objects.items()
        for track_id, info in data['tracks'].items()
    ]

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, ensure_ascii=False, indent=4)

    print("\nProcessing complete. Results saved to JSON.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a video using YOLOv8 for object detection and tracking")
    parser.add_argument('--source', type=str, required=True, help="Path to the input video file")
    parser.add_argument('--model', type=str, default='yolov8m.pt', help="Path to the YOLO model file")
    parser.add_argument('--output_video', type=str, help="Path to save the output video")
    parser.add_argument('--output_json', type=str, help="Path to save the results JSON file")
    args = parser.parse_args()

    # Generate default output paths if not provided
    video_path = args.source
    output_video_path = args.output_video or video_path.replace(".mp4", "_outputdetect.avi")
    output_json_path = args.output_json or video_path.replace(".mp4", "_resultsdetect.json")

    process_video(
        video_path=video_path,
        model_path=args.model,
        output_video_path=output_video_path,
        output_json_path=output_json_path
    )
