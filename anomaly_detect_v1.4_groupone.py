import os
import cv2
import torch
import argparse
import json
import time
from collections import defaultdict
from ultralytics import YOLO


# Environment settings for single-threaded processing
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
cv2.setNumThreads(1)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def anomaly_detect(source, save_dir='output/', input_type='video', weights='yolov8s-oiv7.pt', imgsz=640):
    start_time = time.time()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLO(weights)
    names = model.names

    object_appearances = defaultdict(list)
    background_objects = {}

    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False, varThreshold=50)

    if source == '0':
        source = 0  # Handle camera input if source is '0'

    dataset = cv2.VideoCapture(source)

    if not dataset.isOpened():
        print(f"Failed to load video {source}")
        return

    total_frames = int(dataset.get(cv2.CAP_PROP_FRAME_COUNT)) if source != 0 else float('inf')
    fps = dataset.get(cv2.CAP_PROP_FPS) if source != 0 else 24  # Default FPS for camera input
    frame_skip_ratio = max(1, int(fps / 24))  # Calculate frame skip ratio for 24 FPS process
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
            continue  # Skip frames to maintain target FPS

        current_time = frame_count / fps

        # Calculate and display progress
        if source != 0:  # Only show progress for non-camera sources
            progress_percentage = (frame_count / total_frames) * 100
            print(f"Processing: {progress_percentage:.2f}% complete", end='\r')

        # Apply background subtraction
        fgmask = fgbg.apply(img)
        bin_img = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)[1]
        contour_list, _ = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contour_list:
            if cv2.contourArea(contour) > 17000:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(
                    img, "Anomaly Detected",
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
                    continue  # Skip objects identified as background

                # Check if a new or reappearance should be recorded
                if not object_appearances[obj_id] or current_time - object_appearances[obj_id][-1]['Disappear'] > 1:
                    object_appearances[obj_id].append({'Appear': current_time, 'Disappear': None})

                # Update disappearance time for the current appearance
                object_appearances[obj_id][-1]['Disappear'] = current_time

        # Draw detections on the frame for the output video
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

        video_writer.write(img)

        # Show video frame for real-time detection
        if source == 0:
            cv2.imshow('Anomaly Detection', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release resources after processing is finished
    dataset.release()
    video_writer.release()

    # Save anomaly results to a JSON file
    anomaly_results = []  # Ensure the anomaly results are computed outside the loop
    for obj_id, appearances in object_appearances.items():
        obj_name = names[obj_id]
        for idx, times in enumerate(appearances):
            anomaly_results.append({
                'Object': obj_name,
                'Appear': f"{times['Appear']:.2f}s" if times['Appear'] is not None else "Not detected",
                'Disappear': f"{times['Disappear']:.2f}s" if times['Disappear'] is not None else "Not detected",
                'Status': 'Abnormal',
                'Appearance': f'Appearance {idx + 1}'  # Unique appearance count
            })

    json_output_path = os.path.join(save_dir, 'anomaly_results.json')
    with open(json_output_path, 'w') as json_file:
        json.dump(anomaly_results, json_file, indent=4)

    # Measure and print the total processing time
    end_time = time.time()
    print(f"\nAnomaly Detection Results: {anomaly_results}")
    print(f"Results saved to: {json_output_path}")
    print(f"Total processing time: {end_time - start_time:.2f} seconds")

    # Display the output video after processing
    cap = cv2.VideoCapture(output_video_path)
    if not cap.isOpened():
        print(f"Failed to open the output video {output_video_path}")
        return

    # Read and display the output video frame by frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Processed Video', frame)

        # Wait for key press to move to the next frame
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

    # Release the video capture and close the window
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True, help='Path to the input source (e.g., video file or camera index 0).')
    args = parser.parse_args()
    source = 0 if args.source == '0' else args.source

    with torch.no_grad():
        anomaly_detect(source=source)
