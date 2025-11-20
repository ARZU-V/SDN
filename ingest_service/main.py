import asyncio
import websockets
import redis
import logging
import os
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ingest-service")

# Config
PHONE_WS_URL = os.getenv("PHONE_WS_URL", "ws://192.168.1.101:8080/video")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")

async def ingest_loop():
    logger.info(f"Connecting to Redis at {REDIS_HOST}")
    r = redis.Redis(host=REDIS_HOST, port=6379, db=0)

    while True:
        # --- MOBILE CONNECTION ONLY ---
        try:
            logger.info(f"Connecting to Phone at {PHONE_WS_URL} ...")
            
            async with websockets.connect(PHONE_WS_URL, max_size=10_000_000, open_timeout=3) as ws:
                logger.info("Successfully connected to Phone! Streaming...")
                
                async for msg in ws:
                    if isinstance(msg, bytes):
                        # Push frame to Redis
                        r.lpush("raw_frames", msg)
                        
                        # --- OPTIMIZATION: ZERO LATENCY ---
                        # Trim to [0, 0] keeps ONLY the 1 latest frame.
                        # We drop everything else. This prevents lag buildup.
                        r.ltrim("raw_frames", 0, 0)

        except Exception as e:
            logger.warning(f"Phone connection failed: {e}")
            logger.info("Retrying in 2 seconds...")
            await asyncio.sleep(2)

if __name__ == "__main__":
    asyncio.run(ingest_loop())