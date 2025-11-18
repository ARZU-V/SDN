import asyncio
import websockets
import redis
import logging
import os
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ingest-service")

# Get config from environment variables
PHONE_WS_URL = os.getenv("PHONE_WS_URL", "ws://192.168.1.101:8080/video")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")

async def ingest_loop():
    logger.info(f"Connecting to Redis at {REDIS_HOST}")
    r = redis.Redis(host=REDIS_HOST, port=6379, db=0)
    
    while True:
        try:
            logger.info(f"Connecting to phone at {PHONE_WS_URL}...")
            async with websockets.connect(PHONE_WS_URL, max_size=1_000_000) as websocket:
                logger.info("Successfully connected to phone.")
                async for frame_data in websocket:
                    if isinstance(frame_data, bytes):
                        # Put the raw frame onto the 'raw_frames' queue
                        r.lpush("raw_frames", frame_data)
                        
                        # Prune the queue to stop it from using all memory
                        r.ltrim("raw_frames", 0, 10) 

        except Exception as e:
            logger.error(f"Error in ingest loop: {e}. Retrying in 3s...")
            await asyncio.sleep(3)

if __name__ == "__main__":
    asyncio.run(ingest_loop())