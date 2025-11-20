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

# --- CONFIG ---
AI_INTERVAL = 3      # Run AI only every 3rd frame (Speedup)
INFERENCE_SIZE = (640, 384) # Small size for fast AI processing

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

    def update(self, start_proc_time):
        # Calculate Processing Latency (Inference Time)
        end_proc_time = time.time()
        proc_latency = (end_proc_time - start_proc_time) * 1000 # ms
        
        # Calculate FPS
        new_frame_time = time.time()
        diff = new_frame_time - self.prev_frame_time
        fps = 0
        if diff > 0:
            fps = 1 / diff
        self.prev_frame_time = new_frame_time
        
        # Smoothing (update average every 10 frames)
        self.frame_counter += 1
        if self.frame_counter % 10 == 0: 
            self.avg_fps = fps

        return proc_latency, int(self.avg_fps)

tracker = PerformanceTracker()

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

class AIState:
    def __init__(self):
        self.last_mask = None
        self.last_boxes = []
        self.frame_count = 0

ai_state = AIState()

# --- MODE 1: SEGMENTATION ---
def get_segmentation_mask(small_frame, original_shape):
    """Runs AI on small frame, returns mask scaled to original size."""
    results = model(small_frame, verbose=False, retina_masks=False, conf=0.4)
    
    if results[0].masks:
        # Find people (class 0)
        idx = (results[0].boxes.cls == 0).nonzero(as_tuple=True)[0]
        if len(idx) > 0:
            masks = results[0].masks.data[idx]
            combined = torch.any(masks, dim=0).cpu().numpy().astype(np.uint8)
            
            # Resize the small mask to the ORIGINAL high-res size
            orig_h, orig_w = original_shape[:2]
            full_mask = cv2.resize(combined, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            return full_mask
            
    return None

def apply_segmentation(frame, mask):
    """Applies the red tint and outline using the mask."""
    if mask is None: return frame
    
    color = (0, 0, 255) # Red
    
    # 1. Draw Outline (Contours)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, color, 3)
    
    # 2. Apply Tint (Vectorized for speed)
    mask_bool = mask > 0
    
    # Extract ROI
    roi = frame[mask_bool]
    
    # Create red overlay
    overlay = np.full_like(roi, color)
    
    # Blend: 70% Video + 30% Red
    frame[mask_bool] = cv2.addWeighted(roi, 0.7, overlay, 0.3, 0)
    
    return frame

# --- MODE 2: DETECTION ---
def get_detections(small_frame, original_shape):
    """Runs AI on small frame, returns scaled boxes."""
    results = model(small_frame, verbose=False, conf=0.4)
    boxes = []
    
    # Calculate Scale Factors
    orig_h, orig_w = original_shape[:2]
    small_h, small_w = small_frame.shape[:2]
    x_scale = orig_w / small_w
    y_scale = orig_h / small_h

    for box in results[0].boxes:
        # Get small coordinates
        sx1, sy1, sx2, sy2 = map(int, box.xyxy[0])
        
        # Scale up to original size
        x1 = int(sx1 * x_scale)
        y1 = int(sy1 * y_scale)
        x2 = int(sx2 * x_scale)
        y2 = int(sy2 * y_scale)
        
        cls = int(box.cls[0])
        label = f"{model.names[cls]}"
        boxes.append((x1, y1, x2, y2, label))
        
    return boxes

def apply_detections(frame, boxes):
    """Draws boxes on the frame."""
    for (x1, y1, x2, y2, label) in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame

def loop():
    r = redis.Redis(host=REDIS_HOST, port=6379, db=0)
    r.set("system_mode", "segmentation") 
    
    logger.info("High-Performance Detection Loop Started")
    
    while True:
        try:
            _, data = r.brpop("raw_frames")
            
            # Start Timer for Latency Calculation
            start_time = time.time()

            # Decode original High-Quality frame
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None: continue

            # Check mode
            mode_bytes = r.get("system_mode")
            mode = mode_bytes.decode('utf-8') if mode_bytes else "segmentation"

            # --- FRAME SKIPPING LOGIC ---
            ai_state.frame_count += 1
            run_ai = (ai_state.frame_count % AI_INTERVAL == 0)

            if run_ai:
                # 1. Create Small Copy for AI
                small_frame = cv2.resize(frame, INFERENCE_SIZE)
                
                # 2. Run AI on Small Copy
                if mode == "detection":
                    ai_state.last_boxes = get_detections(small_frame, frame.shape)
                    ai_state.last_mask = None
                else:
                    ai_state.last_mask = get_segmentation_mask(small_frame, frame.shape)
                    ai_state.last_boxes = []

            # --- APPLY RESULTS TO HIGH-RES FRAME ---
            if mode == "detection":
                frame = apply_detections(frame, ai_state.last_boxes)
            else:
                frame = apply_segmentation(frame, ai_state.last_mask)

            # --- DRAW METRICS ---
            latency, fps = tracker.update(start_time)
            frame = draw_stats(frame, latency, fps, mode)

            # Encode and Send
            _, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            r.lpush("processed_frames", buf.tobytes())
            r.ltrim("processed_frames", 0, 0)

        except Exception as e:
            logger.error(f"Error: {e}")
            time.sleep(0.1)

if __name__ == "__main__":
    loop()