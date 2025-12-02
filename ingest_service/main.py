import asyncio
import websockets
import redis
import logging
import os
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ingest-service")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")

async def ingest_loop():
    logger.info(f"Connecting to Redis at {REDIS_HOST}")
    r = redis.Redis(host=REDIS_HOST, port=6379, db=0)
    
    # Default URL in case Redis is empty (Safety fallback)
    current_url = "ws://192.168.1.101:8080/video" 

    while True:
        # 1. Dynamic Config Check
        # Ask Redis: "What is the current phone IP?"
        config_url = r.get("config:phone_url")
        
        if config_url:
            # Decode bytes to string
            new_url = config_url.decode('utf-8')
            if new_url != current_url:
                logger.info(f"Configuration changed! Switching target to: {new_url}")
                current_url = new_url
        else:
            logger.warning("No IP set in Redis. Waiting for configuration...")
            # Wait for user to run set_ip.py
            await asyncio.sleep(2) 
            continue

        # 2. Connection Logic
        try:
            logger.info(f"Connecting to Phone at {current_url} ...")
            
            async with websockets.connect(current_url, max_size=10_000_000, open_timeout=3) as ws:
                logger.info("Successfully connected to Phone! Streaming...")
                
                async for msg in ws:
                    if isinstance(msg, bytes):
                        r.lpush("raw_frames", msg)
                        r.ltrim("raw_frames", 0, 0) # Zero latency optimization

        except Exception as e:
            logger.warning(f"Phone connection failed ({e}).")
            logger.info("Checking for new IP configuration in 2 seconds...")
            await asyncio.sleep(2)

if __name__ == "__main__":
    asyncio.run(ingest_loop())