import os
import cv2
from ultralytics import YOLO
import json
from collections import defaultdict


def process_video(video_path, model_path, output_video_path, output_json_path):
    # Add function to ensure it runs smoothly on different platforms
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'true'

    # Define model and capture
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create VideoWriter object to save output results
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # Dictionary to store object tracking information
    objects = defaultdict(lambda: {'count': 0, 'tracks': {}})

    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Run YOLOv8 for each frame
        results = model.track(frame, persist=True)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()

            # Consider objects as required: Various types of vehicles and people
            for box, track_id, cls, confidence in zip(boxes, track_ids, classes, confidences):
                class_name = model.names[cls]
                # Skip if not a person or vehicle
                if class_name not in ['person', 'car', 'truck', 'bus', 'motorcycle']:
                    continue

                # Open as needed :D
                # if class_name in ['truck', 'bus', 'motorcycle']:
                #     class_name = 'car'

                # Process new object
                if track_id not in objects[class_name]['tracks']:
                    objects[class_name]['count'] += 1
                    count = objects[class_name]['count']
                    objects[class_name]['tracks'][track_id] = {
                        'Object ID': f"{class_name.capitalize()} {count}",
                        'Class': class_name,
                        'Time_of_appearance': frame_count / fps,
                        'Time_of_disappearance': frame_count / fps,
                        'bounding_box': box.tolist(),
                        'Confidence': float(confidence)
                    }
                # Update existing object
                else:
                    objects[class_name]['tracks'][track_id]['Time_of_disappearance'] = frame_count / fps
                    objects[class_name]['tracks'][track_id]['bounding_box'] = box.tolist()
                    objects[class_name]['tracks'][track_id]['Confidence'] = float(confidence)

                # Draw bounding box around the object
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                label = objects[class_name]['tracks'][track_id]['Object ID']
                cv2.putText(frame, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Output frame for the output video
        out.write(frame)

        # Update progress for easy tracking
        frame_count += 1
        progress = (frame_count / total_frames) * 100
        print(f"\rProcessing: {progress:.2f}% complete", end="")

    # Release resources for video recording and playback objects
    cap.release()
    out.release()

    print(f"\nProcessed video saved to {output_video_path}")

    # Convert tracking results to desired format
    json_results = []
    for class_name, data in objects.items():
        for track_id, info in data['tracks'].items():
            # Convert time to "mm:ss" format
            time_appeared = f"{int(info['Time_of_appearance'] // 60):02d}:{int(info['Time_of_appearance'] % 60):02d}"
            time_disappeared = f"{int(info['Time_of_disappearance'] // 60):02d}:{int(info['Time_of_disappearance'] % 60):02d}"

            json_results.append({
                "Object ID": info['Object ID'],
                "Class": info['Class'],
                "Time appeared": time_appeared,
                "Time disappeared": time_disappeared,
                "Bounding box": info['bounding_box'],
                "Confidence": f"{info['Confidence'] * 100:.2f}%"
            })

    # Write results as JSON file
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, ensure_ascii=False, indent=4)

    print(f"Tracking results saved to {output_json_path}")


# Main function
if __name__ == "__main__":
    video_path = 'video_test_2.mp4'
    model_path = 'yolov8m.pt'
    output_video_path = 'output_video_4.mp4'
    output_json_path = 'tracking_results_3.json'

    process_video(video_path, model_path, output_video_path, output_json_path)