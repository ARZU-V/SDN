import redis
import logging
import os
import time
import cv2
import numpy as np
from PIL import Image
from transformers import pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("caption-service")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")

# Load VLM Model (BLIP is fast and lightweight)
# This will download automatically on first run (~900MB)
try:
    logger.info("Loading Captioning Model...")
    captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    logger.info("Model Loaded.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    exit()

def loop():
    r = redis.Redis(host=REDIS_HOST, port=6379, db=0)
    
    while True:
        try:
            # Get the LATEST frame (we don't need every frame)
            # lrange(0, 0) peeks at the newest item without removing it
            data = r.lrange("raw_frames", 0, 0)
            
            if data:
                # Decode
                nparr = np.frombuffer(data[0], np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None: continue

                # Convert to RGB (OpenCV is BGR)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)

                # Generate Caption
                # max_new_tokens=20 keeps it short and fast
                result = captioner(pil_image, max_new_tokens=20)[0]['generated_text']
                logger.info(f"Caption: {result}")

                # Push caption to Redis (broadcast service will pick this up)
                r.set("latest_caption", result)
            
            # Sleep for 2 seconds (VLM is slow, don't choke the CPU)
            time.sleep(2)

        except Exception as e:
            logger.error(f"Error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    loop()