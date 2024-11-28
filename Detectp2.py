import torch
import os
import cv2
from ultralytics import YOLO
import json
import time
from collections import defaultdict

# Check if GPU or CPU is being used
os.environ['KMP_DUPLICATE_LIB_OK'] = 'true'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

def process_video(video_path, model_path, output_video_path, output_json_path):
    model = YOLO(model_path).to(device)

    # Open the video and extract properties
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    
    # Extract input video properties like width, height, FPS, and total frame count
    input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = int(cap.get(cv2.CAP_PROP_FPS))
    input_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Input video resolution: {input_width}x{input_height}")
    print(f"Input video FPS: {input_fps}")
    
    # Prepare to write output video (scaled to 640x360)
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), input_fps, (640, 360))
    
    # Initialize a dictionary to store tracking information
    objects = defaultdict(lambda: {'count': 0, 'tracks': {}})
    frame_count = 0
    start_time = time.time()

    # Iterate through each frame in the video
    while cap.isOpened():
        ret, frame = cap.read()
        elapsed_time = time.time() - start_time
        print(f"FPS: {frame_count / elapsed_time:.2f}", end="\r")  # Print FPS in real-time
        if not ret:
            break

        # Run YOLO model to detect and track objects
        frame_resized = cv2.resize(frame, (640, 360))  # Resize frame for display (optional)
        results = model.track(frame, conf=0.5, persist=True)

        if results[0].boxes.id is not None:  # Check if there are any detected objects
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()

            for box, track_id, cls, confidence in zip(boxes, track_ids, classes, confidences):
                class_name = model.names[cls]
                if class_name not in ['person', 'car', 'truck', 'bus', 'motorcycle']:  # Track only these classes
                    continue

                # Update object tracking information
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

                # Draw bounding box and label on the frame
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                cv2.putText(frame, objects[class_name]['tracks'][track_id]['Object ID'], 
                            (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Write the processed frame to the output video
        out.write(frame)

        # Display the frame in a window (press 'q' to stop)
        display_frame = cv2.resize(frame, (640, 360))
        cv2.imshow("Realtime Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


        frame_count += 1
        print(f"\rProcessing: {frame_count / input_total_frames * 100:.2f}% complete", end="")

    # Cleanup and release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()


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

if __name__ == "__main__":
    # Define the paths for input video, YOLO model, and output files
    video_path = 'video2.mp4' 
    model_path = 'yolov8n.pt' 
    output_video_path = 'video2_output.avi'  
    output_json_path = 'tracking_results_video2.json'  
    process_video(video_path, model_path, output_video_path, output_json_path)
