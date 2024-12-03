# Introduction

This branch is developed for an AI project Object detection group2 before, with tasks such as optimizing performance, cleaning up code and unifying input.

## Example.

### Before


https://github.com/user-attachments/assets/d41565b7-a3da-4577-9c38-7418e3ae6d36

https://drive.google.com/file/d/1dnpuvEZVmkR_uQQHQv1FotCn9l0yYcBl/view?usp=drive_link



### After
https://drive.google.com/file/d/1W8BBMymxvh9PwrD6C0T94sCIweZcVUoA/view?usp=drive_link

https://drive.google.com/file/d/1rcKScVfRy7waydlpDyhxIo5ntwQJbYpA/view?usp=drive_link


# Installation

 1.Download the source code from this branch in the repository.
 
 2.If you want to change the Yolov8m model to another model, change at the line:


model_path = 'YOUR MODEL'

#Example: model_path = 'yolov8m.pt'    

 3. If you want to run the code with your input, change at line:


video_path = 'YOUR_INPUT_VIDEO.mp4'

#video_path = 'video_test.mp4'   


# Libraries Used
1. os: Prevents potential errors related to loading libraries on some systems.
2. cv2: For image processing and computer vision.
3. json: For encoding and decoding JSON data.
4. ultralytics.YOLO: Used to deploy the YOLO object detection model.
5. collections.defaultdict: Class to create default dictionaries where a default value is provided if the key does not exist.
6. Torch GPU Support: The library supports parallel computation on GPUs, speeding up the training and inference of models.


# Notes
Ensuring compatibility between your CUDA Toolkit version and NVIDIA driver version is crucial for optimal performance and stability
My      Driver Version: 566.14,  CUDA Version: 12.7

This code uses the GPU; if you don't have a GPU, you can use the CPU instead.If your hardware is not guaranteed, please consider before using this code.

+ In phase1.py, we have integrated automatic detection of videos in a folder and automatic export of detected videos and Json files to another file.
+ output_folder="DoneDetect"
+ output_json=" name_resultsdetect.json"
  ex: 19_resultsdetect.json
+ output_video= "name_outputdetect.avi"
  ex: 19_outputdetect.avi

run code:
  python3 name file.py --source_folder name folder (have videos)
  
  ex: python3 phase1.py --source khovideo/19.mp4 
      python3 phase1.py --source_folder khovideo
  




