import cv2
import torch
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
import os
import time
from typing import Dict, List, Optional, Tuple

class AnomalyDetector:
    def __init__(self):
        
        self.model = YOLO('models/yolov8s-oiv7.pt')
        self.names = self.model.names
        self.object_appearances = defaultdict(list)
        self.background_objects = {}
        self.fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False, varThreshold=50)
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
        
        # Initialize background objects from first frame
        ret, first_frame = video.read()
        if ret:
            background_results = self.model(first_frame, imgsz=640)
            for result in background_results:
                boxes = result.boxes
                for box in boxes:
                    obj_id = int(box.cls[0])
                    obj_name = self.names[obj_id]
                    self.background_objects[obj_name] = True
        
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
                if obj_name in self.background_objects:
                    continue

                if (not self.object_appearances[obj_id] or 
                    current_time - self.object_appearances[obj_id][-1]['Disappear'] > 1):
                    self.object_appearances[obj_id].append({
                        'Appear': current_time,
                        'Disappear': None
                    })

                self.object_appearances[obj_id][-1]['Disappear'] = current_time
        
        # Update anomaly results
        anomaly_results = []
        for obj_id, appearances in self.object_appearances.items():
            obj_name = self.names[obj_id]
            for idx, times in enumerate(appearances):
                anomaly_results.append({
                    'Object': obj_name,
                    'Appear': f"{times['Appear']:.2f}s" if times['Appear'] is not None else "Not detected",
                    'Disappear': f"{times['Disappear']:.2f}s" if times['Disappear'] is not None else "Not detected",
                    'Status': 'Abnormal',
                    'Appearance': f'Appearance {idx + 1}'
                })
        
        return processed_frame, anomaly_results
    
    def reset(self):
        """Reset detector state"""
        self.object_appearances.clear()
        self.background_objects.clear()
        self.anomaly_results.clear()