import asyncio
import websockets
import redis
import logging
import os
import time
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ingest-service")

PHONE_WS_URL = os.getenv("PHONE_WS_URL", "ws://192.168.1.101:8080/video")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")

async def ingest_loop():
    logger.info(f"Connecting to Redis at {REDIS_HOST}")
    r = redis.Redis(host=REDIS_HOST, port=6379, db=0)
    
    # Helper to check for local cameras
    def get_local_cam():
        for i in range(3):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret: return cap
                cap.release()
        return None

    while True:
        # --- MODE A: PHONE CONNECTION (DISABLED FOR DEBUGGING) ---
        # try:
        #     logger.info(f"Connecting to Phone: {PHONE_WS_URL} ...")
        #     async with websockets.connect(PHONE_WS_URL, max_size=10_000_000, open_timeout=3) as ws:
        #         logger.info("Phone Connected! Streaming...")
        #         async for msg in ws:
        #             if isinstance(msg, bytes):
        #                 r.lpush("raw_frames", msg)
        #                 r.ltrim("raw_frames", 0, 5) # Keep queue short
        # except Exception as e:
        #     logger.warning(f"Phone unavailable ({e}). Switching to Webcam backup...")

        # --- MODE B: WEBCAM BACKUP (FORCED ALWAYS ON) ---
        cap = get_local_cam()
        if cap:
            logger.info("Starting Webcam Stream...")
            # Removed time limit loop, now runs until error
            while True:
                ret, frame = cap.read()
                if not ret: 
                    logger.error("Failed to read frame.")
                    break
                
                _, buf = cv2.imencode('.jpg', frame)
                if _:
                    r.lpush("raw_frames", buf.tobytes())
                    r.ltrim("raw_frames", 0, 5)
                
                await asyncio.sleep(0.03) # ~30 FPS
            cap.release()
        else:
            logger.error("No Webcam found. Retrying in 2s...")
            await asyncio.sleep(2)

if __name__ == "__main__":
    asyncio.run(ingest_loop())