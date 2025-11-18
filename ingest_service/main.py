import asyncio
import redis
import logging
import os
import time
import cv2
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ingest-service")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")

async def ingest_loop():
    logger.info(f"Connecting to Redis at {REDIS_HOST}")
    r = redis.Redis(host=REDIS_HOST, port=6379, db=0)
    
    # --- WEBCAM SETUP ---
    cap = None
    # Try to find a working camera
    for i in range(3):
        logger.info(f"Testing webcam index {i}...")
        temp_cap = cv2.VideoCapture(i)
        if temp_cap.isOpened():
            ret, frame = temp_cap.read()
            if ret:
                logger.info(f"Found working webcam at index {i}")
                cap = temp_cap
                break
            else:
                temp_cap.release()
    
    if cap is None:
        logger.error("No working webcam found! Please check your camera.")
        return

    logger.info("Starting Webcam Stream...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame from webcam.")
                break
            
            # Encode frame to JPEG
            # We use a reasonable quality (80) to keep speed high
            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            
            if ret:
                # Push to Redis queue
                frame_bytes = buffer.tobytes()
                r.lpush("raw_frames", frame_bytes)
                
                # Keep queue small to reduce latency
                r.ltrim("raw_frames", 0, 10)
            
            # Control framerate (approx 30 FPS)
            await asyncio.sleep(0.03)

    except Exception as e:
        logger.error(f"Stream error: {e}")
    finally:
        cap.release()
        logger.info("Webcam released.")

if __name__ == "__main__":
    asyncio.run(ingest_loop())