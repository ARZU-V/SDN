import redis
import cv2
import numpy as np
import logging
import os
import time
import torch
from ultralytics import YOLO

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("detection-service")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")

# --- FAST SETUP: Use Nano Model ---
# Using YOLOv8 Nano for maximum speed
try:
    model = YOLO("yolov8n-seg.pt")
    logger.info("Successfully loaded YOLOv8n-seg (Nano) model.")
except Exception as e:
    logger.error(f"Could not load YOLO model. {e}")
    exit()
# ----------------------------------

def segment_and_color_frame(frame):
    """
    Performs fast segmentation:
    1. Tints the person red (semi-transparent) so they are visible.
    2. Draws a solid red outline around them.
    """
    # Color (BGR) -> Red
    color = (0, 0, 255)
    
    try:
        # Run YOLOv8 Nano
        # verbose=False: Don't print to console (saves time)
        results = model(frame, verbose=False, retina_masks=False)
        
        # Only proceed if masks are found
        if results[0].masks is not None:
            # 1. Filter classes to find only 'person' (class index 0)
            person_indices = (results[0].boxes.cls == 0).nonzero(as_tuple=True)[0]
            
            if len(person_indices) > 0:
                # 2. Combine all person masks
                person_masks = results[0].masks.data[person_indices]
                combined_mask = torch.any(person_masks, dim=0)
                mask_np = combined_mask.cpu().numpy().astype(np.uint8)
                
                # 3. Resize mask to frame size
                # INTER_NEAREST is fastest
                full_size_mask = cv2.resize(mask_np, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                
                # Convert to 0-255 range for Contours
                full_size_mask = full_size_mask * 255
                
                # --- FEATURE 1: Outline ---
                # Find the edges of the mask
                contours, _ = cv2.findContours(full_size_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # Draw the contour (Outline) -> Thickness 3
                cv2.drawContours(frame, contours, -1, color, 3)
                
                # --- FEATURE 2: Transparent Tint ---
                # Create boolean mask
                mask_bool = full_size_mask > 0
                
                # Extract strictly the pixels belonging to the person (Region of Interest)
                # This is faster than blending the whole frame
                roi = frame[mask_bool]
                
                # Create a red block of the same shape
                overlay = np.full_like(roi, color)
                
                # Blend: 70% Original Image + 30% Red Overlay
                blended = cv2.addWeighted(roi, 0.7, overlay, 0.3, 0)
                
                # Put the blended pixels back into the frame
                frame[mask_bool] = blended

    except Exception as e:
        logger.error(f"Error in segmentation: {e}")
        
    return frame

def processing_loop():
    logger.info(f"Connecting to Redis at {REDIS_HOST}")
    r = redis.Redis(host=REDIS_HOST, port=6379, db=0)
    
    while True:
        try:
            # Get the latest frame
            _, frame_data = r.brpop("raw_frames")
            
            # Decode
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            # Process (Segmentation with Tint + Outline)
            processed_frame = segment_and_color_frame(frame)

            # Encode
            ret, buffer = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ret:
                continue
            
            # Push to broadcast queue
            r.lpush("processed_frames", buffer.tobytes())
            r.ltrim("processed_frames", 0, 10)

        except Exception as e:
            logger.error(f"Error in processing loop: {e}")
            time.sleep(1)

if __name__ == "__main__":
    processing_loop()