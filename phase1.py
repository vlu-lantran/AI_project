import torch
import os
import cv2
from ultralytics import YOLO
import json
import time
from collections import defaultdict

# Kiểm tra sử dụng GPU hay CPU
os.environ['KMP_DUPLICATE_LIB_OK'] = 'true'
device = "GPU: " + torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
print(f"Using {device}")

def process_video(video_path, model_path, output_video_path, output_json_path):
    # Tải mô hình YOLO
    model = YOLO(model_path)
    model.overrides['imgsz'] = 1280
    
    # Mở video và lấy thông số
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    
    # Thông tin đầu vào
    input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = int(cap.get(cv2.CAP_PROP_FPS))
    input_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Input video resolution: {input_width}x{input_height}")
    print(f"Input video FPS: {input_fps}")
    
    # Chuẩn bị ghi video đầu ra (sửa codec và kích thước video)
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), input_fps, (input_width, input_height))

    # Biến lưu thông tin theo dõi
    objects = defaultdict(lambda: {'count': 0, 'tracks': {}})
    frame_count = 0
    start_time = time.time()

    # Duyệt qua từng khung hình
    while cap.isOpened():
        ret, frame = cap.read()
        elapsed_time = time.time() - start_time
        print(f"FPS: {frame_count / elapsed_time:.2f}", end="\r")  
        if not ret:
            break

        # Chạy mô hình để phát hiện và theo dõi
        results = model.track(frame, iou=0.4, persist=True)
        frame = cv2.GaussianBlur(frame, (5, 5), 0)  # Smooth the frame to reduce noise
        frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)  # Enhance contrast
        frame = cv2.medianBlur(frame, 5)  # Apply median filter to remove noise

        if results[0].boxes.id is not None:  # Kiểm tra nếu có đối tượng
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
           

            for box, track_id, cls, confidence in zip(boxes, track_ids, classes, confidences):
                class_name = model.names[cls]
                if class_name not in ['person', 'car', 'truck', 'bus', 'motorcycle']:
                    continue

                # Cập nhật thông tin đối tượng
                if track_id not in objects[class_name]['tracks']:
                    objects[class_name]['count'] += 1
                    count = objects[class_name]['count']
                    objects[class_name]['tracks'][track_id] = {
                        'Object ID': f"{class_name.capitalize()} {count}",
                        'Class': class_name,
                        'Time_of_appearance': frame_count / input_fps,
                        'Time_of_disappearance': frame_count / input_fps,
                        'bounding_box': box.tolist(),
                        'Confidence': float(confidence),
                    }
                else:
                    objects[class_name]['tracks'][track_id].update({
                        'Time_of_disappearance': frame_count / input_fps,
                        'bounding_box': box.tolist(),
                        'Confidence': float(confidence),
                    })

                # Draw bounding boxes and labels on the frame
                x1, y1, x2, y2 = map(int, box)
                object_id = objects[class_name]['tracks'][track_id]['Object ID']
                label = f"{object_id} {confidence:.2f}"

                # Increase font size and thickness
                font_scale = 0.8  # Larger font size
                font_thickness = 2  # Thicker text
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]

                # Calculate label position
                text_x = x1
                text_y = y1 - 10 if y1 - 10 > 10 else y1 + 10  # Ensure label stays within frame

                # Draw label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box
                cv2.rectangle(frame, (text_x - 2, text_y - text_size[1] - 4),
                              (text_x + text_size[0] + 2, text_y + 4), (0, 255, 0), -1)  # Background for label
                cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale, (0, 0, 0), font_thickness)  # Black text on green background
    

        # Ghi khung hình đã xử lý vào video đầu ra
        out.write(frame)

        # In tiến trình xử lý
        frame_count += 1
        print(f"\rProcessing: {frame_count / input_total_frames * 100:.2f}% complete", end="")

    # Kết thúc, giải phóng tài nguyên
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Lưu kết quả theo dõi vào file JSON
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

    print("\nProcessing complete. Results saved to JSON.")

def process_folder(input_folder, output_folder="DoneDetect", model_path='yolov8m.pt'):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):  # Sửa từ listdit thành listdir
        input_path = os.path.join(input_folder, file_name)  # Sửa từ os.path.json thành os.path.join
        
        if os.path.isfile(input_path) and file_name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Sửa từ os.path.islife thành os.path.isfile
            print(f"Processing video: {file_name}")

            output_video_path = os.path.join(output_folder, file_name.replace(".mp4", "_outputdetect.avi"))
            output_json_path = os.path.join(output_folder, file_name.replace(".mp4", "_resultsdetect.json"))

            process_video(
                video_path=input_path,
                model_path=model_path,
                output_video_path=output_video_path,
                output_json_path=output_json_path
            )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process videos in a folder using YOLOv8")
    parser.add_argument('--source_folder', type=str, required=True, help="Path to the folder containing input videos")
    args = parser.parse_args()

    process_folder(
        input_folder=args.source_folder
    )
