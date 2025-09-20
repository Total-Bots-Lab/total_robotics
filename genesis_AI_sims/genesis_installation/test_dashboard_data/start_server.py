
import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
import webbrowser
import time
import threading

def start_server():
    os.chdir(r"test_dashboard_data")
    handler = SimpleHTTPRequestHandler
    httpd = HTTPServer(("localhost", 8081), handler)
    print(f"Dashboard server running at http://localhost:8081")
    httpd.serve_forever()

def open_browser():
    time.sleep(3)  # Wait for server to start
    try:
        webbrowser.open(f"http://localhost:8081/dashboard.html")
        print(f"Dashboard opened: http://localhost:8081/dashboard.html")
    except Exception as e:
        print(f"Could not open browser: {e}")

if __name__ == "__main__":
    # Start browser in a separate thread
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Start server (blocks)
    start_server()
