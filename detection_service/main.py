import redis
import cv2
import numpy as np
import logging
import os
import time
from ultralytics import YOLO  # --- EDIT: New import ---

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("detection-service")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")

# --- EDIT: New YOLOv8 Segmentation Model Setup ---
# This will download 'yolov8s-seg.pt' the first time it's run.
try:
    model = YOLO("yolov8s-seg.pt")
    logger.info("Successfully loaded YOLOv8s-seg model.")
except Exception as e:
    logger.error(f"Could not load YOLO model. {e}")
    exit()
# --- End of Setup ---

# --- EDIT: This function is now completely different ---
def segment_and_color_frame(frame):
    """
    Performs segmentation on a frame and colors in the 'person' class.
    """
    # Create a copy to draw on, and a color for the fill
    # We use BGR format for OpenCV, so (0, 0, 255) is RED.
    color = (0, 0, 255)
    
    try:
        # Run the YOLOv8 model on the frame
        # verbose=False silences the text output for each frame
        results = model(frame, verbose=False)
        
        # Check if any masks were found
        if results[0].masks:
            # Loop through all detected masks
            for cls, mask in zip(results[0].boxes.cls.int(), results[0].masks.data):
                
                # We only care about class '0', which is 'person'
                if cls == 0:
                    # 1. Get the mask (it's a tensor), move to CPU, and convert to numpy
                    mask_np = mask.cpu().numpy().astype(np.uint8)
                    
                    # 2. Resize the mask from its small size to the full frame size
                    # We use INTER_NEAREST for a sharp edge, no blur.
                    full_size_mask = cv2.resize(mask_np, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

                    # 3. Create a boolean version of the mask
                    mask_bool = full_size_mask.astype(bool)
                    
                    # 4. Use "Numpy magic" to apply the color
                    # This finds all pixels in the 'frame' where 'mask_bool' is True
                    # and sets their color to our defined 'color'.
                    frame[mask_bool] = color
                    
    except Exception as e:
        logger.error(f"Error in segment_and_color_frame: {e}")
        
    return frame

# -----------------------------------------------------------------

def processing_loop():
    logger.info(f"Connecting to Redis at {REDIS_HOST}")
    r = redis.Redis(host=REDIS_HOST, port=6379, db=0)
    
    while True:
        try:
            # Wait for a frame from the 'raw_frames' queue (blocking pop)
            _, frame_data = r.brpop("raw_frames")
            
            # 1. Decode
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            # 2. Process (Run our new segmentation function)
            processed_frame = segment_and_color_frame(frame)

            # 3. Encode (Increased quality for less blur)
            ret, buffer = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ret:
                continue
            
            # 4. Push to the next queue
            r.lpush("processed_frames", buffer.tobytes())
            r.ltrim("processed_frames", 0, 10)

        except Exception as e:
            logger.error(f"Error in processing loop: {e}")
            time.sleep(1)

if __name__ == "__main__":
    processing_loop()