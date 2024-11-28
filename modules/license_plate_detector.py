import cv2
import json
import os
import re
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from paddleocr import PaddleOCR
from ultralytics import YOLO

class LicensePlateDetector:
    def __init__(self, 
                 plate_model_path: str = 'models/license_plate_detector.pt', 
                 output_dir: str = 'output', 
                 confidence_threshold: float = 0.85, 
                 log_level: int = logging.INFO):
        """
        Initialize License Plate Detector
        
        Args:
            plate_model_path (str): Path to the YOLO license plate detection model
            output_dir (str): Directory to store output files
            confidence_threshold (float): Minimum confidence for plate detection
            log_level (int): Logging level
        """
        # Setup logging
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        logging.basicConfig(
            level=log_level, 
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename=os.path.join(output_dir, 'license_plate_detector.log')
        )
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.confidence_threshold = confidence_threshold
        self.unique_plates: Dict[str, Dict] = {}
        
        # Detection models
        self.plate_detector = YOLO(plate_model_path)
        self.ocr = PaddleOCR(lang='en')
        
        # Plate validation regex
        self.VALID_PLATE_PATTERN = re.compile(r'^(\d{2}[-][A-Z0-9]{4,6})$')
        
        # Results storage
        self.detection_results: List[Dict] = []
    
    def initialize_video(self, video_path: str) -> Tuple[cv2.VideoCapture, Dict]:
        """
        Initialize video capture and get video properties
        
        Args:
            video_path (str): Path to input video file
        
        Returns:
            Tuple of video capture object and video information dictionary
        """
        video = cv2.VideoCapture(video_path)
        video_info = {
            'fps': video.get(cv2.CAP_PROP_FPS),
            'width': int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'frame_count': int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        }
        
        # Reset video to beginning
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return video, video_info
    
    def is_valid_vietnamese_plate(self, plate_text: str) -> bool:
        """
        Validate Vietnamese license plate format
        
        Args:
            plate_text (str): License plate text to validate
        
        Returns:
            bool: Whether the plate text is valid
        """
        self.logger.debug(f"Validating plate: {plate_text}")
        
        if self.VALID_PLATE_PATTERN.match(plate_text):
            return True
        
        self.logger.debug(f"Invalid plate format: {plate_text}")
        return False
    
    def process_frame(self, frame: np.ndarray, current_time: float) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process a single frame for license plate detection
        
        Args:
            frame (np.ndarray): Input video frame
            current_time (float): Current timestamp in the video
        
        Returns:
            Tuple of processed frame and list of detected plates
        """
        processed_frame = frame.copy()
        self.detection_results.clear()
        
        # Detect license plates
        results_plate = self.plate_detector(frame)
        plate_boxes = results_plate[0].boxes
        
        self.logger.debug(f"Found {len(plate_boxes)} plate boxes")
        
        for plate_box in plate_boxes:
            px1, py1, px2, py2 = map(int, plate_box.xyxy[0])
            plate_img = frame[py1:py2, px1:px2]
            
            # OCR detection
            ocr_result = self.ocr.ocr(plate_img, cls=False)
            plate_text = ''
            plate_confidence = 0.0
            
            if ocr_result and ocr_result[0]:
                for line in ocr_result[0]:
                    text, confidence = line[1]
                    plate_text += text
                    plate_confidence += confidence
                plate_confidence /= len(ocr_result[0])
                
                self.logger.debug(f"OCR Result: {plate_text}, Confidence: {plate_confidence}")
            else:
                self.logger.debug("No OCR result found")
                continue
            
            # Validate and process plate if confident enough
            if plate_confidence > self.confidence_threshold and self.is_valid_vietnamese_plate(plate_text):
                plate_entry = {
                    "plate_number": plate_text,
                    "confidence": float(plate_confidence),
                    "timestamp": int(current_time)
                }
                
                # Only add if this is a new or more confident plate
                if (plate_text not in self.unique_plates or 
                    plate_confidence > self.unique_plates[plate_text]['confidence']):
                    self.unique_plates[plate_text] = plate_entry
                    self.detection_results.append(plate_entry)
                
                # Visualize on frame
                cv2.rectangle(processed_frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
                cv2.putText(processed_frame, 
                            f'{plate_text} ({plate_confidence:.2f})', 
                            (px1, py1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return processed_frame, self.detection_results
    
    def save_results(self) -> None:
        """
        Save detected license plates to JSONL file
        """
        try:
            output_path = os.path.join(self.output_dir, 'detected_plates.jsonl')
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for plate_text, plate_data in self.unique_plates.items():
                    json.dump(plate_data, f, ensure_ascii=False)
                    f.write('\n')
            
            self.logger.info(f"Saved {len(self.unique_plates)} unique plates to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving plates: {e}")
    
    def reset(self) -> None:
        """
        Reset detector state
        """
        self.unique_plates.clear()
        self.detection_results.clear()