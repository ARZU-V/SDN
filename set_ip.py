import redis

# Connect to the local Redis (exposed by Docker)
try:
    r = redis.Redis(host='localhost', port=6379, db=0)
    r.ping()
except redis.ConnectionError:
    print("Error: Could not connect to Redis. Is Docker running?")
    exit()

print("--- SDN-VR CONFIGURATION CONTROLLER ---")
print("Current stored URL:", r.get("config:phone_url"))

# 1. Take User Input
ip = input("\nEnter the new Phone IP (e.g., 192.168.1.101): ").strip()

# 2. Validate minimal format
if not ip:
    print("Invalid input.")
    exit()

# 3. Construct WebSocket URL
# Assuming standard port 8080 and path /video based on your previous setups
ws_url = f"ws://{ip}:8080/video"

# 4. Push to Redis (The "SDN Controller" action)
r.set("config:phone_url", ws_url)

print(f"\nSUCCESS: Configuration updated!")
print(f"Target set to: {ws_url}")
print("The Ingest Service will switch to this stream immediately.")
print("---------------------------------------")