import cv2
import torch
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
import os
from typing import Dict, List, Optional, Tuple

# Initialize list to save the time appear abnormal
list_time_appear = []
total_seconds_vid = 0
class AnomalyDetector:
    def __init__(self):
        # YOLO model path
        self.model = YOLO('models/yolov8s-oiv7.pt')  
        self.names = self.model.names 
        # To track object appearances
        self.object_appearances = defaultdict(list)  
        # Background objects detected in the first frame
        self.background_objects = set()
        # Background subtractor
        self.fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False, varThreshold=50)  
        # List to store anomaly detection results
        self.anomaly_results = []  
    
    def initialize_video(self, video_path: str) -> Tuple[cv2.VideoCapture, Dict]:
        """Initialize video capture and get video properties"""
        global total_seconds_vid
        video = cv2.VideoCapture(video_path)
        video1 = cv2.VideoCapture(video_path)
        video_info = {
            'fps': video.get(cv2.CAP_PROP_FPS),
            'width': int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'frame_count': int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        }
        # Total frame of video
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        # FPS of video
        fps = video.get(cv2.CAP_PROP_FPS)
        
        total_seconds = int(total_frames / fps)
        total_seconds_vid = total_seconds
        one_second = int(total_frames / total_seconds)
        frame_count = 0
        ret, first_frame = video1.read()
        if ret:
            background_results = self.model(first_frame)
            for result in background_results:
                boxes = result.boxes
                for box in boxes:
                    obj_id = int(box.cls[0])
                    obj_name = self.names[obj_id]
                    # if obj_name in ["Tree", "Building"]: self.background_objects.add(obj_name)
                    self.background_objects.add(obj_name)
        while True:
            ret, frame = video1.read()
            if not ret: break
            frame_count += 1
            current_time = int(frame_count / fps)
            if frame_count % one_second != 0: continue
            '''We only need to process follow second instead of processing frame by frame'''
            results = self.model(frame)
            is_detect = False
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w = x2 - x1
                    h = y2 - y1
                    if w * h > 1000:
                        obj_id = int(box.cls[0])
                        obj_name = self.names[obj_id]
                        if obj_name in self.background_objects:
                            continue  # Skip objects identified as background
                        # Check if YOLO can detect object that is in this time have abnormal object
                        if (len(list_time_appear) == 0) or (len(list_time_appear) > 0 and list_time_appear[-1]["Disappear"] != 0):
                            list_time_appear.append({
                                "Appear": current_time,
                                "Disappear": 0
                            })
                        is_detect = True
                    # This time don't have any abnormal object
                    if not is_detect:
                        if len(list_time_appear) > 0 and list_time_appear[-1]["Disappear"] == 0 and current_time - list_time_appear[-1]["Appear"] >= 1:
                            list_time_appear[-1]["Disappear"] = current_time
        # Reset video to the beginning
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return video, video_info
    
    def process_frame(self, frame: np.ndarray, current_time: float) -> Tuple[np.ndarray, List[dict]]:
        """Process a single frame for anomaly detection"""
        processed_frame = frame.copy()
        # global list_time_appear, total_seconds_vid
        if len(list_time_appear) > 0 and list_time_appear[-1]["Disappear"] == 0 and total_seconds_vid != 0: 
            list_time_appear[-1]["Disappear"] = total_seconds_vid 
        self.anomaly_results = list_time_appear.copy()
        # print(f"List time appear abnormal: {list_time_appear}")
        current_time = int(current_time)

        # Background subtraction
        fgmask = self.fgbg.apply(frame)
        bin_img = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)[1]
        contours, _ = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Only process large contours
            conf_score = 0
            area = cv2.contourArea(contour) 
            area_confidence = min(area / 50000, 1.0)
            x, y, w, h = cv2.boundingRect(contour)
            rect_area = w * h
            if area > 17000:
                is_detect = False
                for obj in list_time_appear:
                    if obj["Appear"] == current_time:
                        is_detect = True
                        break
                if not is_detect:
                    list_time_appear.append({
                        "Appear": current_time,
                        "Disappear": current_time
                    })
                compactness = area / rect_area
                
                # Motion direction penalty
                # Additional logic can be added here to penalize unusual motion directions
                # Combine factors (you can adjust weights)
                confidence = (0.6 * area_confidence) + (0.4 * compactness)
                conf_score = round(min(max(confidence, 0), 1), 2)  # Ensure between 0 and 1
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(processed_frame, f"Anomaly Detected: {conf_score}",
                            (x, y - 10 if y > 20 else y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        self.anomaly_results = list_time_appear.copy()
        return processed_frame, self.anomaly_results
    
    def reset(self):
        """Reset detector state"""
        self.object_appearances.clear()
        self.background_objects.clear()
        self.anomaly_results.clear()
