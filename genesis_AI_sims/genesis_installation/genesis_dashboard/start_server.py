
import http.server
import socketserver
import os
import webbrowser
import time

PORT = 8080
os.chdir(r"genesis_dashboard")

class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress log messages

try:
    with socketserver.TCPServer(("", PORT), QuietHandler) as httpd:
        print(f"Genesis Dashboard server running on http://localhost:{PORT}/dashboard.html")
        time.sleep(1)
        webbrowser.open(f"http://localhost:{PORT}/dashboard.html")
        httpd.serve_forever()
except Exception as e:
    print(f"Server error: {e}")
