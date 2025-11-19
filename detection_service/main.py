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
    # Using Nano model for speed
    model = YOLO("yolov8n-seg.pt")
    logger.info("YOLOv8 Nano Loaded.")
except:
    logger.error("Model load failed.")
    exit()

# --- PERFORMANCE TRACKER HELPER ---
class PerformanceTracker:
    def __init__(self):
        self.prev_frame_time = 0
        self.new_frame_time = 0
        self.fps = 0
        self.avg_fps = 0
        self.frame_counter = 0
        self.processing_times = []

    def update(self, start_proc_time):
        # Calculate Processing Latency (Inference Time)
        end_proc_time = time.time()
        proc_latency = (end_proc_time - start_proc_time) * 1000 # ms
        
        # Calculate FPS
        self.new_frame_time = time.time()
        diff = self.new_frame_time - self.prev_frame_time
        if diff > 0:
            self.fps = 1 / diff
        self.prev_frame_time = self.new_frame_time
        
        # Smoothing
        self.frame_counter += 1
        if self.frame_counter % 10 == 0: # Update average every 10 frames
            self.avg_fps = self.fps

        return proc_latency, int(self.avg_fps)

tracker = PerformanceTracker()

# --- MODE 1: SEGMENTATION ---
def run_segmentation(frame):
    color = (0, 0, 255) # Red
    # retina_masks=False is faster
    results = model(frame, verbose=False, retina_masks=False)
    if results[0].masks:
        idx = (results[0].boxes.cls == 0).nonzero(as_tuple=True)[0]
        if len(idx) > 0:
            masks = results[0].masks.data[idx]
            combined = torch.any(masks, dim=0).cpu().numpy().astype(np.uint8)
            resized_mask = cv2.resize(combined, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            # Outline
            contours, _ = cv2.findContours(resized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, contours, -1, color, 3)
            
            # Tint
            roi = frame[resized_mask > 0]
            overlay = np.full_like(roi, color)
            frame[resized_mask > 0] = cv2.addWeighted(roi, 0.6, overlay, 0.4, 0)
    return frame

# --- MODE 2: DETECTION (BOXES) ---
def run_detection(frame):
    results = model(frame, verbose=False)
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        
        if conf > 0.5:
            label = f"{model.names[cls]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 5), 0, 0.6, (0, 255, 0), 2)
    return frame

def draw_stats(frame, latency, fps, mode):
    """Draws Sci-Fi style metrics on the frame"""
    # Background panel for text
    cv2.rectangle(frame, (5, 5), (280, 110), (0, 0, 0), -1)
    cv2.rectangle(frame, (5, 5), (280, 110), (0, 255, 255), 1)
    
    # Text Settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 255, 255) # Cyan
    
    cv2.putText(frame, f"SYSTEM: ONLINE", (15, 30), font, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"MODE: {mode.upper()}", (15, 55), font, 0.6, color, 1)
    cv2.putText(frame, f"LATENCY: {latency:.1f} ms", (15, 80), font, 0.6, color, 1)
    cv2.putText(frame, f"FPS: {fps}", (15, 105), font, 0.6, color, 1)
    
    return frame

def loop():
    r = redis.Redis(host=REDIS_HOST, port=6379, db=0)
    r.set("system_mode", "segmentation") 
    
    logger.info("Detection Loop Started")
    while True:
        try:
            _, data = r.brpop("raw_frames")
            
            # Start Timer
            start_time = time.time()
            
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None: continue

            # Logic
            mode = r.get("system_mode")
            mode = mode.decode('utf-8') if mode else "segmentation"

            if mode == "detection":
                processed = run_detection(frame)
            else:
                processed = run_segmentation(frame)

            # Calculate Stats
            latency, fps = tracker.update(start_time)
            
            # Draw Stats on Frame
            processed = draw_stats(processed, latency, fps, mode)

            _, buf = cv2.imencode('.jpg', processed, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            r.lpush("processed_frames", buf.tobytes())
            r.ltrim("processed_frames", 0, 5)

        except Exception as e:
            logger.error(f"Error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    loop()