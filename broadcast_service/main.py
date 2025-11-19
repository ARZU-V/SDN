import asyncio
import websockets
import redis
import logging
import os
import base64
import time
import json
import ssl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("broadcast-service")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
clients = set()

async def ws_handler(websocket):
    clients.add(websocket)
    logger.info("Secure Quest connected!")
    try:
        r = redis.Redis(host=REDIS_HOST, port=6379, db=0)
        async for message in websocket:
            try:
                data = json.loads(message)
                if "mode" in data:
                    r.set("system_mode", data["mode"])
            except: pass
    except: pass
    finally:
        clients.remove(websocket)

async def broadcast_loop():
    # Retry Redis connection
    r = None
    while r is None:
        try:
            r = redis.Redis(host=REDIS_HOST, port=6379, db=0)
            r.ping()
        except:
            logger.error("Waiting for Redis...")
            time.sleep(2)

    while True:
        try:
            if not clients:
                await asyncio.sleep(0.5)
                continue
            
            res = r.brpop("processed_frames", timeout=0.5)
            if res:
                frame_b64 = base64.b64encode(res[1]).decode('utf-8')
                await asyncio.gather(*[c.send(frame_b64) for c in clients], return_exceptions=True)
            await asyncio.sleep(0.001)
        except Exception as e:
            logger.error(f"Broadcast error: {e}")
            await asyncio.sleep(1)

async def main():
    asyncio.create_task(broadcast_loop())
    
    port = 8765
    ssl_context = None

    # --- CHECK FOR SSL CERTIFICATES ---
    if os.path.exists("/app/cert.pem") and os.path.exists("/app/key.pem"):
        logger.info("SSL Certificates found! Enabling WSS (Secure WebSocket).")
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain("/app/cert.pem", "/app/key.pem")
    else:
        logger.warning("No SSL certs found. Running in insecure WS mode.")

    logger.info(f"Starting Broadcast Server on 0.0.0.0:{port}")
    
    # Start server with SSL context (if it exists)
    async with websockets.serve(ws_handler, "0.0.0.0", port, ssl=ssl_context):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())