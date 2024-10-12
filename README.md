## Introduction
This branch is for an A.I. project focused on anomaly detection using the current model Yolov8-Open-ImageV7. The script is designed to detect anomalies in video data, recording the appearance and disappearance times of objects, as well as when an object reappears or disappears again.

## Installation
1. Download the source code from this branch in the repository.
2. Install the required libraries from the `requirements.txt` file.
3. The script has integrated YOLO requirements. If you want to change the YOLO model from Ultralytics, simply download the new model and change the file name in the following line:

```python
parser.add_argument('--weights', type=str, default='yolov8s-oiv7.pt', help='model.pt path(s)')
#Example: default='yolo11n.pt'    
```
Run the source code:
```python
#Example
python anomaly_detect_v1.2_group1.py --source sample/sample.mp4
```

## Features
- Analyze anomalies in the video of object 1 and detect the disappearance of object 1.
- Log the times and number of occurrences of similar objects.

## Libraries Used
1. **argparse**: For parsing command-line arguments.
2. **time**: For time measurement.
3. **json**: For encoding and decoding JSON data.
4. **pathlib**: Provides easy and portable manipulation of file paths across platforms.
5. **cv2**: For image processing and computer vision.
6. **torch**: For neural network processing.
7. **collections.defaultdict**: Class to create default dictionaries where a default value is provided if the key does not exist.
8. **ultralytics.YOLO**: Used to deploy the YOLO object detection model.

## Contribution
If you would like to contribute to the project, please create a pull request and clearly describe the changes you want to make.

## Authors
- **Lăng Nhật Tân**
- **Đỗ Lý Anh Kiệt**
- **Lê Hải**

## License
Thank you for your interest and use of our project!
