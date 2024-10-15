## Introduction
This branch is for an A.I project focused on anomaly detection using the current model Yolov8-Open-ImageV7. This script is designed to detect anomalies in video data, recording the appearance and disappearance times of objects and environmental fluctuations as well as when an object reappears or disappears again.
## Installation
1. Download the source code from this branch in the repository.
2. Install the required libraries from the `requirements.txt` file (if needed).
3. The script has integrated with YOLO requirements. If you want to change the YOLO model from Ultralytics, simply download the new model and change the file name in the following line:
```python
parser.add_argument('--weights', type=str, default='yolov8s-oiv7.pt', help='model.pt path(s)')
#Example: default='yolo11n.pt'
#But i would highly recommend Yolov8 Models with the latest update.    
```
Run the source code:
```python
#Example
python anomaly_detect_v1.2_group1.py --source sample/sample.mp4
```
## Features
- Analyze anomalies in the video of object 1 and detect the disappearance of object 1.
- Log the times and number of occurrences of similar objects.
- Analyze anomalies in environmental fluctuations like explosion, fire, smoke, and accident.
## Libraries Used
1. **argparse**: Parsing command-line arguments.
2. **time**: Time measurement.
3. **json**: Encoding and decoding JSON data.
4. **pathlib**: Provides easy and portable manipulation of file paths across platforms.
5. **cv2**: Image processing and computer vision.
6. **torch**: Neural network processing.
7. **collections.defaultdict**: Class to create default dictionaries where a default value is provided if the key does not exist.
8. **ultralytics.YOLO**: Used to deploy the YOLO object detection model.
## Contribution
If you would like to contribute to the project, please create a pull request and clearly describe the changes you want to make.
## Authors
- **Lăng Nhật Tân**
- **Đỗ Lý Anh Kiệt**
- **Lê Hải**
## License
Thank you for your interest and use of our project! 🔥🔥🔥
