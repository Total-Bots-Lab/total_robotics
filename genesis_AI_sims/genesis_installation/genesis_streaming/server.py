
import os
import http.server
import socketserver
import threading

class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress server logs

os.chdir(r"C:\Users\Ritu\Project\total_robotics\genesis_AI_sims\genesis_installation\genesis_streaming")
with socketserver.TCPServer(("", 8090), QuietHandler) as httpd:
    print(f"Genesis Dashboard server running on port 8090")
    httpd.serve_forever()
