import cv2
import torch
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
import os
from typing import Dict, List, Optional, Tuple

class AnomalyDetector:
    def __init__(self):
        # YOLO model path
        self.model = YOLO('models/yolov8s-oiv7.pt')  
        self.names = self.model.names 
        # To track object appearances
        self.object_appearances = defaultdict(list)  
        # Background objects detected in the first frame
        self.background_objects = {}
        # Background subtractor
        self.fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False, varThreshold=50)  
        # List to store anomaly detection results
        self.anomaly_results = []  
    
    def initialize_video(self, video_path: str) -> Tuple[cv2.VideoCapture, Dict]:
        """Initialize video capture and get video properties"""
        video = cv2.VideoCapture(video_path)
        video_info = {
            'fps': video.get(cv2.CAP_PROP_FPS),
            'width': int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'frame_count': int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        }

        # Initialize background objects from the first frame
        ret, first_frame = video.read()
        if ret:
            background_results = self.model(first_frame, imgsz=640)
            for result in background_results:
                boxes = result.boxes
                for box in boxes:
                    obj_id = int(box.cls[0])
                    obj_name = self.names[obj_id]
                    self.background_objects[obj_name] = True
        
        # Reset video to the beginning
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)  
        return video, video_info
    
    def process_frame(self, frame: np.ndarray, current_time: float) -> Tuple[np.ndarray, List[dict]]:
        """Process a single frame for anomaly detection"""
        processed_frame = frame.copy()

        # Background subtraction
        fgmask = self.fgbg.apply(frame)
        bin_img = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)[1]
        contours, _ = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Only process large contours
            if cv2.contourArea(contour) > 17000:  
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(processed_frame, "Anomaly Detected",
                            (x, y - 10 if y > 20 else y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # YOLO detection
        results = self.model(frame, imgsz=640)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                obj_id = int(box.cls[0])
                obj_name = self.names[obj_id]

                # Skip if object is part of the background
                if obj_name in self.background_objects:
                    continue

                # Update appearance/disappearance times
                if (not self.object_appearances[obj_id] or 
                    current_time - self.object_appearances[obj_id][-1]['Disappear'] > 1):
                    self.object_appearances[obj_id].append({
                        'Appear': current_time,
                        'Disappear': None
                    })

                self.object_appearances[obj_id][-1]['Disappear'] = current_time
        
        # Update anomaly results
        self.anomaly_results.clear()
        for obj_id, appearances in self.object_appearances.items():
            obj_name = self.names[obj_id]
            for idx, times in enumerate(appearances):
                self.anomaly_results.append({
                    'Object': obj_name,
                    'Appear': f"{times['Appear']:.2f}s" if times['Appear'] is not None else "Not detected",
                    'Disappear': f"{times['Disappear']:.2f}s" if times['Disappear'] is not None else "Not detected",
                    'Status': 'Abnormal',
                    'Appearance': f'Appearance {idx + 1}'
                })
        
        # Draw bounding boxes for detections
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                obj_id = int(box.cls[0])
                obj_name = self.names[obj_id]
                confidence = box.conf[0]

                # Skip background objects or low-confidence detections
                if obj_name in self.background_objects or confidence < 0.5:
                    continue

                color = (0, 255, 0) if obj_name not in self.background_objects else (0, 0, 255)
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                label = f"{obj_name} {confidence:.2f}"
                cv2.putText(
                    processed_frame, label,
                    (x1, y1 - 10 if y1 > 20 else y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return processed_frame, self.anomaly_results
    
    def reset(self):
        """Reset detector state"""
        self.object_appearances.clear()
        self.background_objects.clear()
        self.anomaly_results.clear()
