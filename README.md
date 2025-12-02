SDN-VR Command Center: Real-Time Cloud-Edge AI Surveillance

This project implements a Real-Time AI Surveillance System using Virtual Reality (VR) for monitoring and control. It decouples hardware from intelligence by using a smartphone as a "drone" camera and offloading heavy AI processing (Object Detection, Segmentation, Captioning) to an edge server (Laptop or Cloud VM) using Network Function Virtualization (NFV) and Software Defined Networking (SDN) principles.

🚀 Features

Real-Time Streaming: Low-latency video streaming from mobile/webcam to VR.

Dual AI Modes:

Segmentation: Real-time red overlay highlighting people with semi-transparency.

Object Detection: Bounding boxes identifying objects (Person, Chair, etc.).

Live Situation Analysis: Integration with FastVLM (Vision-Language Model) to generate real-time text captions of the scene (e.g., "A person sitting at a desk").

Immersive VR Interface:

Sci-Fi Command Center: Futuristic 3D environment (futuristic_room.glb).

Floating TV Screen: Interactive video display inside the VR room.

Teleportation: Move around the virtual environment.

Grabbable Screen: Grab and reposition the video screen in 3D space.

Hybrid Ingest: Automatically switches between Mobile Camera and Laptop Webcam if connection is lost.

Secure Access: Full HTTPS/WSS support for WebXR compatibility.

🏗️ Architecture (NFV & SDN)

The system is built as a Service Function Chain (SFC) of microservices running in Docker containers.

1. Virtual Network Functions (VNFs)

Ingest Service: Captures video (Phone/Webcam) → Pushes to Redis Queue.

Detection Service (Fast Path): Pulls frame → Runs YOLOv8 Nano → Pushes processed frame to Redis.

Captioning Service (Slow Path): Pulls frame → Runs FastVLM/BLIP → Updates Redis Key.

Broadcast Service: Pulls processed frame + latest caption → Sends via Secure WebSocket (wss://) to VR Client.

2. SDN Control Plane

Docker Networking: Defines the virtual topology.

Redis Message Bus: Acts as the programmable software switch, routing data between decoupled services.

Tailscale (Optional): Creates a virtual overlay network for connecting Cloud VMs to local devices.

🛠️ Prerequisites

Hardware:

PC/Laptop (Linux/Windows/Mac) - Acts as the Edge Server.

Meta Quest 2/3/Pro - VR Headset.

Android/iOS Phone - Acts as the "Drone" camera.

Software:

Docker Desktop (Installed & Running).

Python 3.10+ (For local helper scripts).

IP Webcam App (or similar) on Android.

📥 Setup & Installation

Clone the Repository:

git clone <repository_url>
cd sdnVR


Configure IP Addresses:

Phone IP: Open docker-compose.yml, find ingest service, and update PHONE_WS_URL with your phone's IP (e.g., ws://192.168.1.101:8080/video).

PC IP: Open index.html, find YOUR_PC_IP const (approx line 90), and ensure it matches your PC's LAN IP (e.g., 192.168.1.24). Note: The script tries to auto-detect this, but manual check is good.

Generate SSL Certificates:
Run this python script locally to create self-signed certificates for HTTPS support (Required for VR):

python run_secure.py


(Press Ctrl+C to stop it after it says "SECURE SERVER RUNNING").

▶️ How to Run (Quick Start)

1. Start the Backend (Docker)

This spins up Redis, AI Engines, and the Broadcast Server.

docker-compose up --build


Wait until you see logs like Starting WebSocket broadcast server and Detection Loop Started.

2. Start the Frontend (Secure Web Server)

In a separate terminal:

python run_secure.py


This hosts the VR interface on HTTPS port 4443.

3. Connect VR Headset

Put on Meta Quest.

Open Quest Browser.

Go to: https://YOUR_PC_IP:4443 (e.g., https://192.168.1.24:4443).

Accept Warning: Click "Advanced" -> "Proceed to ... (unsafe)".

Click the "ENTER VR" button at the bottom.

📱 Dynamic IP Configuration (Runtime)

If your phone's IP changes while the system is running, you don't need to restart Docker!

Run the helper script on your laptop:

python set_ip.py


Enter the new Phone IP.

The Ingest Service will instantly reconnect.

☁️ Running on Cloud VM (Optional)

To deploy on a Cloud VM (AWS/GCP/Oracle):

Install Tailscale on VM, Laptop, and Phone to create a flat network mesh.

Update Configs:

docker-compose.yml: Set PHONE_WS_URL to Phone's Tailscale IP. Remove privileged: true and devices sections (VMs have no webcam).

index.html: Set YOUR_PC_IP to VM's Tailscale IP.

Run: Same commands as above (docker-compose up --build).

Access: https://<VM_TAILSCALE_IP>:4443.

📊 Performance Monitoring

The system includes a built-in HUD (Heads-Up Display) visible in the VR video feed:

SYSTEM: Online status.

MODE: Current AI mode (Segmentation / Detection).

LATENCY: End-to-end processing time in milliseconds.

FPS: Real-time frames per second.

🐞 Troubleshooting

"WEBXR NEEDS HTTPS" Button:

You must access the site via https://. Use python run_secure.py.

Alternatively, enable chrome://flags -> Insecure origins treated as secure on the Quest.

Black Screen / No Video:

Check docker-compose logs ingest.

If Phone connection failed, check your Phone IP and ensure the streaming app is running.

If Switching to BACKUP WEBCAM, the system is using the laptop webcam.
