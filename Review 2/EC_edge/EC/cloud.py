# cloud.py (HTTP listener)
from http.server import BaseHTTPRequestHandler, HTTPServer
import json, time
class CloudHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers["Content-Length"])
        body = self.rfile.read(length)
        data = json.loads(body)
        print("[CLOUD] Received:", data.get("device"), data.get("seq"))
        self.send_response(200); self.end_headers()
        self.wfile.write(b"OK")

server = HTTPServer(("0.0.0.0", 9000), CloudHandler)
print("[CLOUD] listening on :9000")
server.serve_forever()
