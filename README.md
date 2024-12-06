# License Plate Detection using YOLOv8 and OCR  

This project leverages **YOLOv8** for license plate detection and **PaddleOCR** for optical character recognition (OCR). The system processes input videos, detects license plates, extracts plate numbers with OCR, and saves valid results in a structured JSONL format.  

---

## Features  

- **License Plate Detection**: Utilizes YOLOv8 with a pre-trained and fine-tuned model for robust plate detection.  
- **OCR Integration**: Implements PaddleOCR to extract text from detected license plates.  
- **Validation**: Filters results using regex validation for Vietnamese license plate formats.  
- **Output**: Saves detected plates with timestamp and confidence scores in a JSONL file and overlays results on an output video.  
- **Performance Tracking**: Displays frames-per-second (FPS) on processed video frames.  

---

## Prerequisites  

1. **Python Libraries**:
   - OpenCV: `pip install opencv-python`
   - PaddleOCR: `pip install paddleocr`
   - ultralytics (YOLOv8): `pip install ultralytics`
   - Other: `argparse`, `os`, `logging`, `json`, `datetime`

2. **Models**:
   - YOLOv8 model file (`license_plate_detector.pt`) for license plate detection. Ensure this is present in the `models/` directory.

---

## Usage 

### 1. **Command Line Arguments**  

| Argument              | Description                                   | Default           |
|-----------------------|-----------------------------------------------|-------------------|
| `--source`            | Path to the input video (required).           | N/A               |
| `--output_dir`        | Directory for output files.                   | `output/`         |
| `--confidence_threshold` | Minimum OCR confidence for plate validation. | `0.5`             |

### 2. **Execution**  

Run the script with the following command:  
```python
python main.py --source path/to/video.mp4
```

## Output  

1. **Detected Video**:  
   - File saved as `detected_video.mp4` in the output directory.  
   - Shows bounding boxes around detected plates, plate numbers, and FPS.  

2. **JSONL File**:  
   - Detected license plates saved in `detected_plates.jsonl`.  
   - Each entry includes:  
     ```json
     {
       "timestamp": "2024-11-27 15:30:45",
       "plate_number": "51H-12345",
       "confidence": 0.92
     }
     ```

---

## System Workflow  

1. **Preprocessing**:
   - Video input is resized to 420p for faster processing while preserving aspect ratio.  
   
2. **Detection**:
   - YOLOv8 detects license plates in video frames.  

3. **OCR**:
   - PaddleOCR extracts text from detected plates.  
   - Confidence scores are averaged if multiple lines of text are detected.  

4. **Validation**:
   - Results are validated against a regex pattern for Vietnamese license plates.  
   - Plates with confidence above the specified threshold are saved.  

5. **Output**:
   - Valid detections are overlayed on video frames.  
   - Results are saved in a JSONL file for further analysis.  

---

## Code Highlights  

- **Validation of Plates**:  
   ```python
   VALID_PLATE_PATTERN = re.compile(r'^(\d{2}[-][A-Z0-9]{4,6})$')
   def is_valid_vietnamese_plate(plate_text):
       return VALID_PLATE_PATTERN.match(plate_text)
   ```

- **Saving Results**:  
   ```python
   def save_to_jsonl(plate_data):
       with open(output_jsonl_path, 'a') as f:
           json.dump(data, f)
           f.write('\n')
   ```

- **OCR and Detection**:  
   ```python
   ocr_result = ocr.ocr(plate_img, cls=False)
   if ocr_result:
       for line in ocr_result[0]:
           text, confidence = line[1]
   ```

---

## Improvements  

- **Real-Time Processing**: Optimize for live camera feeds.  
- **Multi-language OCR**: Extend support for different languages.  
- **Performance Metrics**: Track inference time and processing speed per frame.  

