import http.server, ssl, os

# 1. Generate SSL certificates if they don't exist
if not os.path.exists("cert.pem") or not os.path.exists("key.pem"):
    print("Generating SSL certificates...")
    # This command generates a self-signed certificate valid for 365 days
    os.system("openssl req -new -x509 -keyout key.pem -out cert.pem -days 365 -nodes -subj '/CN=localhost'")

# 2. Configure the HTTPS Server on Port 4443
# We use 4443 to avoid clashing with the Docker HTTP server on 8000
server_address = ('0.0.0.0', 4443)
httpd = http.server.HTTPServer(server_address, http.server.SimpleHTTPRequestHandler)

# 3. Wrap the server with SSL
context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
context.load_cert_chain(certfile='cert.pem', keyfile='key.pem')
httpd.socket = context.wrap_socket(httpd.socket, server_side=True)

print("\n--------------------------------------------------")
print("SECURE SERVER RUNNING!")
print("1. Ensure Docker is running (for the video backend).")
print("2. Access on Quest at: https://192.168.1.24:4443")
print("--------------------------------------------------\n")

httpd.serve_forever()