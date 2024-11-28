## Introduction
This branch developed for the phase two of A.I project focused on Vietnamese vehicle plate detection using the custom model Yolov8. The script now integrated with simple web detector to detect car plates in video data, recording the appearance and disappearance times of objects, as well as when an object reappears or disappears again.

## Implemetation:
1. Download the source code from this branch in the repository.
2. Install the required libraries from the `requirements.txt` file.
3. The script has compulsory python version 3.9.10

## Usage
To run the car plate detection script with websites detect you need to change the library of the modules 

```python
from modules.license_plate_detector import LicensePlateDetector 
```

```python
def initialize_video_processing(source):
    """Khởi tạo xử lý video"""
    try:
        processing_state['detector'] = LicensePlateDetector()
        processing_state['frame_count'] = 0
```

To run the anomalies detection script with websites detect you need to change the library of the modules 

```python
from modules.anomaly_detection import AnomalyDetector
```

```python
def initialize_video_processing(source):
    """Khởi tạo xử lý video"""
    try:
        processing_state['detector'] = AnomalyDetector()
        processing_state['frame_count'] = 0
```

Upload your video and start the processing.

## Contribution
If you would like to contribute to the project, please create a pull request and clearly describe the changes you want to make.

## Authors
- **Đỗ Lý Anh Kiệt**
- **Quang Mỹ Tâm**

## License
Thank you for your interest and use of our project! 🔥 🔥 🔥 
