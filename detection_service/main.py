import redis
import cv2
import numpy as np
import logging
import os
import time
import torch
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("detection-service")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")

# --- LOAD MODEL ---
try:
    model = YOLO("yolov8n-seg.pt")
    logger.info("YOLOv8 Nano Loaded.")
except:
    logger.error("Model load failed.")
    exit()

# --- PERFORMANCE TRACKER ---
class PerformanceTracker:
    def __init__(self):
        self.prev_frame_time = 0
        self.avg_fps = 0
        self.frame_counter = 0

    def update(self):
        new_frame_time = time.time()
        diff = new_frame_time - self.prev_frame_time
        fps = 1 / diff if diff > 0 else 0
        self.prev_frame_time = new_frame_time
        
        self.frame_counter += 1
        if self.frame_counter % 10 == 0: 
            self.avg_fps = fps
        return int(self.avg_fps)

tracker = PerformanceTracker()

# --- MODE 1: SEGMENTATION (FIXED & ROBUST) ---
def run_segmentation(frame):
    color = (0, 0, 255) # Red (BGR)
    results = model(frame, verbose=False, retina_masks=False, conf=0.4)
    
    if results[0].masks:
        idx = (results[0].boxes.cls == 0).nonzero(as_tuple=True)[0]
        if len(idx) > 0:
            masks = results[0].masks.data[idx]
            combined = torch.any(masks, dim=0).cpu().numpy().astype(np.uint8)
            
            # Resize mask to match frame size
            # cv2.resize can return slightly different sizes if not careful, 
            # so we explicitly use frame.shape dimensions.
            h, w = frame.shape[:2]
            resized_mask = cv2.resize(combined, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # --- ROBUST BLENDING FIX ---
            # 1. Create a colored image for the mask (Full Size)
            # Make a solid red image the same size as the frame
            colored_mask = np.zeros_like(frame, dtype=np.uint8)
            colored_mask[:] = color
            
            # 2. Use the binary mask to 'cut out' the red shape
            # We need a 3-channel mask to apply it to the 3-channel image
            mask_3ch = cv2.merge([resized_mask, resized_mask, resized_mask])
            
            # 3. Apply the mask: Keep red where mask is 1, keep 0 elsewhere
            colored_overlay = cv2.bitwise_and(colored_mask, colored_mask, mask=resized_mask)
            
            # 4. Blend only where the mask is present
            # We use 'where' to apply transparency only on the person
            # Formula: pixel = pixel*0.7 + red*0.3
            # This avoids "size mismatch" errors because we operate on the full frame arrays.
            person_pixels = (mask_3ch > 0)
            frame[person_pixels] = (frame[person_pixels] * 0.7 + colored_overlay[person_pixels] * 0.3).astype(np.uint8)

            # 5. Draw Outline
            contours, _ = cv2.findContours(resized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, contours, -1, color, 2)
            
    return frame

# --- MODE 2: DETECTION (OPTIMIZED) ---
def run_detection(frame):
    results = model(frame, verbose=False, conf=0.4)
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{model.names[cls]}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def draw_stats(frame, fps, mode):
    cv2.putText(frame, f"MODE: {mode.upper()} | FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return frame

def loop():
    r = redis.Redis(host=REDIS_HOST, port=6379, db=0)
    r.set("system_mode", "segmentation") 
    
    logger.info("Detection Loop Started (Fixed)")
    
    while True:
        try:
            _, data = r.brpop("raw_frames")
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None: continue

            # Resize first to ensure consistent dimensions for everything
            frame = cv2.resize(frame, (640, 480))

            mode = r.get("system_mode")
            mode = mode.decode('utf-8') if mode else "segmentation"

            if mode == "detection":
                processed = run_detection(frame)
            else:
                processed = run_segmentation(frame)

            fps = tracker.update()
            processed = draw_stats(processed, fps, mode)

            _, buf = cv2.imencode('.jpg', processed, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            r.lpush("processed_frames", buf.tobytes())
            r.ltrim("processed_frames", 0, 0)

        except Exception as e:
            logger.error(f"Error: {e}")
            time.sleep(0.1)

if __name__ == "__main__":
    loop()