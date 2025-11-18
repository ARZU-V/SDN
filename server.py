
import asyncio
import websockets
import cv2
import base64
import logging
import numpy as np
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Object Detection Setup ---
# Model files (must be in the same folder)
PROTOTXT = "MobileNetSSD_deploy.prototxt.txt"
MODEL = "MobileNetSSD_deploy.caffemodel"

# List of object classes
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Load the Caffe model
try:
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
    logger.info("Successfully loaded object detection model.")
except cv2.error as e:
    logger.error(f"Error: Could not load model. Make sure '{PROTOTXT}' and '{MODEL}' are in the correct folder.")
    logger.error(e)
    exit()
# --- End of Object Detection Setup ---


# --- WebSocket Server (for Quest) ---
# Keep track of all connected Quest clients
clients = set()

async def register(websocket):
    """Adds a new Quest client to the set."""
    logger.info(f"New Quest client connected: {websocket.remote_address}")
    clients.add(websocket)

async def unregister(websocket):
    """Removes a Quest client from the set."""
    logger.info(f"Quest client disconnected: {websocket.remote_address}")
    clients.remove(websocket)

# FIX: Removed the 'path' argument which was causing the TypeError
async def stream_video(websocket):
    """Handles a new Quest WebSocket connection."""
    await register(websocket)
    try:
        await websocket.wait_closed()
    finally:
        await unregister(websocket)
# --- End of WebSocket Server ---


def detect_objects(frame):
    """Performs object detection on a single frame and draws boxes."""
    try:
        (h, w) = frame.shape[:2]
        # Preprocess the image for the neural network
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

        # Pass the blob through the network
        net.setInput(blob)
        detections = net.forward()

        # Loop over the detections
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Filter out weak detections (confidence > 0.5)
            if confidence > 0.5:
                idx = int(detections[0, 0, i, 1])
                if idx < len(CLASSES):
                    label = CLASSES[idx]
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # --- EDIT: Made box thicker (from 2 to 3) ---
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 3)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    # --- EDIT: Made text larger (0.5 to 0.8) and bolder (2 to 3) ---
                    cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
        return frame
    except Exception as e:
        logger.error(f"Error in detect_objects: {e}")
        return frame # Return original frame on error


async def receive_and_broadcast():
    """
    Connects to the phone's WebSocket server, receives frames,
    processes them, and broadcasts them to all Quest clients.
    """
    # !! IMPORTANT !!
    # Make sure this IP is correct for your PHONE.
    # This must be the IP of your phone on the Wi-Fi network.
    phone_ws_url = "ws://192.168.1.101:8080/video"
    
    # Variables for logging FPS
    last_log_time = time.time()
    frame_count = 0
    
    while True:
        try:
            logger.info(f"Connecting to phone at {phone_ws_url}...")
            async with websockets.connect(phone_ws_url, max_size=1_000_000) as websocket:
                logger.info("Successfully connected to phone.")
                
                # Reset counter on new connection
                last_log_time = time.time()
                frame_count = 0
                
                # Receive frames from the phone
                async for frame_data in websocket:
                    if not isinstance(frame_data, bytes):
                        continue

                    # --- Logging Logic ---
                    frame_count += 1
                    current_time = time.time()
                    if (current_time - last_log_time) >= 1.0: # Log every 1 second
                        fps = frame_count / (current_time - last_log_time)
                        logger.info(f"Receiving blobs from phone. FPS: {fps:.1f}, Last blob size: {len(frame_data)} bytes")
                        frame_count = 0
                        last_log_time = current_time
                    # --- End of Logic ---

                    # 1. Decode the binary data (from phone) to an OpenCV frame
                    nparr = np.frombuffer(frame_data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    if frame is None:
                        logger.warning("Could not decode frame, skipping.")
                        continue

                    # 2. Process the frame (Object Detection)
                    processed_frame = detect_objects(frame)

                    # 3. Encode the processed frame to JPEG
                    ret, buffer = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    if not ret:
                        logger.warning("Error encoding frame.")
                        continue

                    # 4. Convert to base64 string
                    jpg_as_text = base64.b64encode(buffer).decode('utf-8')

                    # 5. Broadcast to all connected Quest clients
                    if clients:
                        await asyncio.gather(
                            *[client.send(jpg_as_text) for client in clients],
                            return_exceptions=True
                        )

        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"Connection to phone closed: {e}. Retrying in 3s...")
            await asyncio.sleep(3)
        except Exception as e:
            logger.error(f"Error in receive_and_broadcast: {e}. Retrying in 3s...")
            await asyncio.sleep(3)


async def main():
    """Starts the WebSocket server (for Quest) and the client (for phone)."""
    
    # This is the server for the Meta Quest to connect to
    quest_server_host = "0.0.0.0"
    quest_server_port = 8765
    logger.info(f"Starting WebSocket server for Quest on ws://{quest_server_host}:{quest_server_port}")
    
    quest_server = websockets.serve(stream_video, quest_server_host, quest_server_port)

    # This task connects to the phone and broadcasts to the Quest
    broadcast_task = asyncio.create_task(receive_and_broadcast())

    await asyncio.gather(quest_server, broadcast_task)

if __name__ == "__main__":
    asyncio.run(main())