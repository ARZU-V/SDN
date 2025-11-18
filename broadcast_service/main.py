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
# A set of connected WebSocket clients
clients = set()

async def register(websocket):
    """Register a new client."""
    clients.add(websocket)
    logger.info(f"New Quest client connected: {websocket.remote_address}. Total clients: {len(clients)}")

async def unregister(websocket):
    """Unregister a client."""
    clients.remove(websocket)
    logger.info(f"Quest client disconnected: {websocket.remote_address}. Total clients: {len(clients)}")

async def ws_handler(websocket):
    """
    Handles the WebSocket connection.
    This function MUST NOT return until the connection is closed.
    """
    await register(websocket)
    try:
        # We need to keep this coroutine running so the connection stays open.
        # We iterate over messages (even if we don't expect any) to detect disconnections.
        async for message in websocket:
            # We can log messages if needed, but mostly we just wait here.
            pass
    except websockets.exceptions.ConnectionClosed:
        logger.info("Client connection closed normally.")
    except Exception as e:
        logger.error(f"Error in ws_handler: {e}")
    finally:
        await unregister(websocket)

async def broadcast_loop():
    """
    Continuously pulls frames from Redis and broadcasts them to all connected clients.
    """
    logger.info(f"Connecting to Redis at {REDIS_HOST}...")
    
    # Retry logic for Redis connection
    r = None
    while r is None:
        try:
            r = redis.Redis(host=REDIS_HOST, port=6379, db=0)
            r.ping() # Test connection
            logger.info("Successfully connected to Redis!")
        except redis.ConnectionError:
            logger.error("Redis not ready. Retrying in 2s...")
            time.sleep(2)

    logger.info("Starting broadcast loop...")
    
    while True:
        try:
            # If no clients are connected, just wait and skip processing to save CPU
            if not clients:
                await asyncio.sleep(0.5)
                continue

            # Get the latest frame from the 'processed_frames' queue
            # timeout=0.1 allows us to check for clients periodically
            result = r.brpop("processed_frames", timeout=0.1)
            
            if result:
                _, frame_data = result
                
                # Encode to base64 for the browser
                frame_b64 = base64.b64encode(frame_data).decode('utf-8')

                # Broadcast to all connected clients
                # We create a list of tasks so one slow client doesn't block others
                if clients:
                    # websockets.broadcast is efficient but requires the library version.
                    # We'll stick to manual gather for compatibility.
                    await asyncio.gather(
                        *[client.send(frame_b64) for client in clients],
                        return_exceptions=True
                    )
            
            # Small sleep to prevent tight loop if Redis is empty but clients exist
            await asyncio.sleep(0.001)

        except Exception as e:
            logger.error(f"Error in broadcast loop: {e}")
            await asyncio.sleep(1)

async def main():
    # Start the broadcast loop as a background task
    asyncio.create_task(broadcast_loop())
    
    port = 8765
    logger.info(f"Starting WebSocket broadcast server on ws://0.0.0.0:{port}")
    
    # Use 'async with' to manage the server lifecycle properly
    async with websockets.serve(ws_handler, "0.0.0.0", port):
        # Run forever
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())