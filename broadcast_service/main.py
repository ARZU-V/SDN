import asyncio
import websockets
import redis
import logging
import os
import base64
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("broadcast-service")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
clients = set()
last_frame = None
last_frame_time = 0

async def register(websocket):
    logger.info(f"New Quest client connected: {websocket.remote_address}")
    clients.add(websocket)

async def unregister(websocket):
    logger.info(f"Quest client disconnected: {websocket.remote_address}")
    clients.remove(websocket)

async def ws_handler(websocket):
    await register(websocket)
    try:
        # Keep the connection alive
        await websocket.wait_closed()
    finally:
        await unregister(websocket)

async def broadcast_loop():
    """Pulls frames from Redis and broadcasts to all clients."""
    global last_frame, last_frame_time
    logger.info(f"Connecting to Redis at {REDIS_HOST}")
    r = redis.Redis(host=REDIS_HOST, port=6379, db=0)

    while True:
        try:
            # Get the latest frame from the 'processed_frames' queue
            # We use brpop to wait efficiently
            _, frame_data = r.brpop("processed_frames")
            
            # Cache the frame
            last_frame = base64.b64encode(frame_data).decode('utf-8')
            last_frame_time = time.time()

            # Broadcast to all connected clients
            if clients:
                await asyncio.gather(
                    *[client.send(last_frame) for client in clients],
                    return_exceptions=True
                )
        except Exception as e:
            logger.error(f"Error in broadcast loop: {e}")
            time.sleep(1)

async def main():
    # Start the broadcast loop as a background task
    asyncio.create_task(broadcast_loop())
    
    # Start the WebSocket server for clients to connect
    port = 8765
    logger.info(f"Starting WebSocket broadcast server on ws://0.0.0.0:{port}")
    await websockets.serve(ws_handler, "0.0.0.0", port)
    
    # Keep the server running
    await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())