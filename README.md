## Introduction
This branch developed for the phase two of A.I project focused on anomaly detection using the current model Yolov8-Open-ImageV7. The script now integrated with simple web detector to detect anomalies in video data, recording the appearance and disappearance times of objects, as well as when an object reappears or disappears again.

## Implemetation:
1. Download the source code from this branch in the repository.
2. Install the required libraries from the `requirements.txt` file.
3. The script has integrated YOLO requirements. If you want to change the YOLO model from Ultralytics, simply download the new model and change the file name in the following line:

```python
weights = 'yolov8s-oiv7.pt'
```
Weights: Path to the YOLO model weights file (e.g., yolov8s-oiv7.pt).

## Usage
To run the anomaly detection script with websites detect
```python
python app.py 
```
Upload your video and start the processing.

## Features

- **Anomaly Detection**: Analyze videos for anomalies, specifically focusing on detecting fluctuations in the environment such as explosions, fires, smoke, and other unusual events.
  
- **Object Behavior Detection**: Specifically track the first object (Object 1) in the video to detect its disappearance, logging the times and occurrences for further analysis.

- **Bounding Boxes**:
  1. **Fluctuation Detection**: Draw bounding boxes around environmental fluctuations (e.g., explosion, fire, smoke). These boxes are indicated in **red** and labeled as "Fluctuation" to distinguish them from tracked objects.
  2. **Object Detection**: Draw bounding boxes around detected objects, specifically focusing on Object 1. These boxes are color-coded based on the status of the object: **green** for "Objects" and **blue** for any anomalies detected.

- **Time Logging**: Log the appearance and disappearance times of similar objects to detect any different activity in the scene.
- **Real-time Detection**: Log the appearance and disappearance times by using camera with 24 frame per secs.
- **Output**: Generate a video output for camera format .avi extensions. Visualizes detected anomalies of objects, along with a JSON file summarizing the detection results, including anomaly occurrences and times.

## Main Libraries Used
1. **json**: Encoding and decoding JSON data.
2. **cv2**: Image processing and computer vision.
3. **torch**: Neural network processing.
4. **ultralytics.YOLO**: Used to deploy the YOLO object detection model.
5. **Background Substraction**: Use for analyzing anomaly in the environmental fluctuations in the script.

## Contribution
If you would like to contribute to the project, please create a pull request and clearly describe the changes you want to make.

## Authors
- **Lăng Nhật Tân**
- **Đỗ Lý Anh Kiệt**

## License
Thank you for your interest and use of our project! 🔥 🔥 🔥 
