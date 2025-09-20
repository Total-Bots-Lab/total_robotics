"""
Enhanced TensorBoard with WebSocket Real-time Streaming
Provides real-time data streaming to web clients via WebSockets
"""

import asyncio
import websockets
import json
import threading
import time
import os
import subprocess
import webbrowser
from datetime import datetime
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import logging

class WebSocketTensorBoardManager:
    """
    Enhanced TensorBoard manager with WebSocket real-time streaming
    """
    
    def __init__(self, log_dir="tensorboard_logs", tb_port=6006, ws_port=8765, auto_start=True):
        self.log_dir = log_dir
        self.tb_port = tb_port
        self.ws_port = ws_port
        self.auto_start = auto_start
        
        # TensorBoard process
        self.tensorboard_process = None
        self.tb_running = False
        
        # WebSocket server
        self.ws_server = None
        self.ws_running = False
        self.connected_clients = set()
        
        # Data streaming
        self.stream_data = {
            "step_metrics": [],
            "episode_metrics": [],
            "real_time_data": {},
            "status": "initialized"
        }
        
        # Threading
        self.data_lock = threading.Lock()
        self.streaming_thread = None
        
        # TensorBoard Writer
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        if auto_start:
            self.start_all_services()
    
    def start_all_services(self):
        """Start both TensorBoard and WebSocket services"""
        print("🚀 Starting Enhanced TensorBoard with WebSocket streaming...")
        
        # Start TensorBoard
        self.start_tensorboard()
        
        # Start WebSocket server
        self.start_websocket_server()
        
        # Start data streaming thread
        self.start_data_streaming()
        
        print(f"📊 TensorBoard Dashboard: http://localhost:{self.tb_port}")
        print(f"🔌 WebSocket Streaming: ws://localhost:{self.ws_port}")
        print(f"🌐 Real-time Dashboard: http://localhost:{self.ws_port + 1}/dashboard")
        
        # Auto-open browser if requested
        if self.auto_start:
            time.sleep(2)
            try:
                webbrowser.open(f"http://localhost:{self.tb_port}")
                print("🌐 TensorBoard opened in browser")
            except:
                print("⚠️ Could not auto-open browser")
    
    def start_tensorboard(self):
        """Start TensorBoard process"""
        try:
            # Stop any existing TensorBoard
            self.stop_tensorboard(silent=True)
            
            # Start TensorBoard with WebSocket-friendly settings
            cmd = [
                "tensorboard",
                f"--logdir={self.log_dir}",
                f"--port={self.tb_port}",
                "--reload_interval=1",  # Faster reload for real-time feel
                "--max_reload_threads=1"
            ]
            
            self.tensorboard_process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
            )
            
            self.tb_running = True
            print(f"✅ TensorBoard started on port {self.tb_port}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start TensorBoard: {e}")
            return False
    
    def stop_tensorboard(self, silent=False):
        """Stop TensorBoard process"""
        if self.tensorboard_process and self.tensorboard_process.poll() is None:
            try:
                self.tensorboard_process.terminate()
                self.tensorboard_process.wait(timeout=5)
                self.tb_running = False
                if not silent:
                    print("✅ TensorBoard stopped")
            except:
                if self.tensorboard_process.poll() is None:
                    self.tensorboard_process.kill()
                    if not silent:
                        print("⚠️ TensorBoard force-killed")
    
    def start_websocket_server(self):
        """Start WebSocket server for real-time streaming"""
        async def websocket_handler(websocket, path):
            """Handle WebSocket connections"""
            print(f"📡 New WebSocket client connected from {websocket.remote_address}")
            self.connected_clients.add(websocket)
            
            try:
                # Send initial data
                await self.send_initial_data(websocket)
                
                # Keep connection alive and handle incoming messages
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        await self.handle_websocket_message(websocket, data)
                    except json.JSONDecodeError:
                        await websocket.send(json.dumps({
                            "error": "Invalid JSON format"
                        }))
                        
            except websockets.exceptions.ConnectionClosed:
                print("📡 WebSocket client disconnected")
            finally:
                self.connected_clients.discard(websocket)
        
        # Start WebSocket server in a separate thread
        def run_websocket_server():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            start_server = websockets.serve(
                websocket_handler, 
                "localhost", 
                self.ws_port,
                ping_interval=20,
                ping_timeout=10
            )
            
            print(f"🔌 WebSocket server starting on port {self.ws_port}")
            loop.run_until_complete(start_server)
            self.ws_running = True
            loop.run_forever()
        
        ws_thread = threading.Thread(target=run_websocket_server, daemon=True)
        ws_thread.start()
        time.sleep(1)  # Give server time to start
    
    async def send_initial_data(self, websocket):
        """Send initial data to newly connected client"""
        with self.data_lock:
            initial_data = {
                "type": "initial_data",
                "data": self.stream_data,
                "timestamp": datetime.now().isoformat()
            }
        
        await websocket.send(json.dumps(initial_data))
    
    async def handle_websocket_message(self, websocket, data):
        """Handle incoming WebSocket messages"""
        message_type = data.get("type", "unknown")
        
        if message_type == "ping":
            await websocket.send(json.dumps({"type": "pong"}))
        
        elif message_type == "request_data":
            # Send current data
            with self.data_lock:
                response = {
                    "type": "data_update",
                    "data": self.stream_data,
                    "timestamp": datetime.now().isoformat()
                }
            await websocket.send(json.dumps(response))
        
        elif message_type == "subscribe":
            # Client wants to subscribe to specific data streams
            data_types = data.get("data_types", ["all"])
            # Implementation for selective data streaming
            print(f"📡 Client subscribed to: {data_types}")
    
    def start_data_streaming(self):
        """Start background thread for real-time data streaming"""
        def streaming_loop():
            while self.ws_running or self.tb_running:
                try:
                    if self.connected_clients:
                        # Broadcast real-time data to all connected clients
                        asyncio.run(self.broadcast_real_time_data())
                    time.sleep(0.5)  # Stream at 2Hz
                except Exception as e:
                    print(f"⚠️ Streaming error: {e}")
                    time.sleep(1)
        
        self.streaming_thread = threading.Thread(target=streaming_loop, daemon=True)
        self.streaming_thread.start()
        print("📊 Real-time data streaming started")
    
    async def broadcast_real_time_data(self):
        """Broadcast current data to all connected WebSocket clients"""
        if not self.connected_clients:
            return
        
        with self.data_lock:
            # Prepare real-time data package
            real_time_package = {
                "type": "real_time_update",
                "data": {
                    "latest_metrics": self.stream_data.get("real_time_data", {}),
                    "step_count": len(self.stream_data.get("step_metrics", [])),
                    "episode_count": len(self.stream_data.get("episode_metrics", [])),
                    "status": self.stream_data.get("status", "running")
                },
                "timestamp": datetime.now().isoformat()
            }
        
        # Send to all connected clients
        disconnected_clients = set()
        for client in self.connected_clients:
            try:
                await client.send(json.dumps(real_time_package))
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)
        
        # Remove disconnected clients
        self.connected_clients -= disconnected_clients
    
    def log_step_metrics(self, step_data):
        """Log step-level metrics to both TensorBoard and WebSocket stream"""
        # Log to TensorBoard (existing functionality)
        step = step_data.get('step', 0)
        
        if 'reward' in step_data and step_data['reward'] is not None:
            self.writer.add_scalar('Step/Reward', step_data['reward'], step)
        
        if 'total_reward' in step_data and step_data['total_reward'] is not None:
            self.writer.add_scalar('Step/Total_Reward', step_data['total_reward'], step)
        
        if 'tracking_reward' in step_data and step_data['tracking_reward'] is not None:
            self.writer.add_scalar('Step/Tracking_Reward', step_data['tracking_reward'], step)
        
        # Add to WebSocket stream
        with self.data_lock:
            self.stream_data["step_metrics"].append({
                **step_data,
                "timestamp": datetime.now().isoformat()
            })
            
            # Update real-time data
            self.stream_data["real_time_data"] = {
                "current_step": step,
                "latest_reward": step_data.get('reward', 0),
                "latest_total_reward": step_data.get('total_reward', 0),
                "latest_tracking_reward": step_data.get('tracking_reward', 0)
            }
            
            # Keep only recent step data to prevent memory growth
            if len(self.stream_data["step_metrics"]) > 1000:
                self.stream_data["step_metrics"] = self.stream_data["step_metrics"][-500:]
    
    def log_episode_metrics(self, episode_data):
        """Log episode-level metrics to both TensorBoard and WebSocket stream"""
        episode = episode_data.get('episode', 0)
        
        # Log to TensorBoard (existing functionality)
        if 'total_reward' in episode_data:
            self.writer.add_scalar('Episode/Total_Reward', episode_data['total_reward'], episode)
        
        if 'tracking_reward' in episode_data:
            self.writer.add_scalar('Episode/Tracking_Reward', episode_data['tracking_reward'], episode)
        
        # Add to WebSocket stream
        with self.data_lock:
            self.stream_data["episode_metrics"].append({
                **episode_data,
                "timestamp": datetime.now().isoformat()
            })
            
            # Update real-time data
            self.stream_data["real_time_data"].update({
                "current_episode": episode,
                "latest_episode_reward": episode_data.get('total_reward', 0),
                "latest_episode_tracking": episode_data.get('tracking_reward', 0)
            })
    
    def create_websocket_dashboard(self):
        """Create a custom WebSocket-powered dashboard"""
        dashboard_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Genesis AI - Real-time Training Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #1a1a1a;
            color: #ffffff;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
        }}
        .status {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            margin-left: 10px;
        }}
        .status.connected {{
            background-color: #2ed573;
            color: #ffffff;
        }}
        .status.disconnected {{
            background-color: #ff4757;
            color: #ffffff;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background-color: #2d2d2d;
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #ff6b6b;
            text-align: center;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .metric-label {{
            color: #aaaaaa;
            font-size: 0.9em;
        }}
        .plot-container {{
            background-color: #2d2d2d;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        #connection-log {{
            background-color: #2d2d2d;
            padding: 15px;
            border-radius: 5px;
            max-height: 200px;
            overflow-y: auto;
            font-family: monospace;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 Genesis AI - Real-time Training Dashboard</h1>
        <p>WebSocket Connection: <span id="connection-status" class="status disconnected">Disconnected</span></p>
        <p>TensorBoard: <a href="http://localhost:{self.tb_port}" target="_blank">http://localhost:{self.tb_port}</a></p>
    </div>

    <div class="metrics-grid" id="metrics-grid">
        <div class="metric-card">
            <div class="metric-value" id="current-step">0</div>
            <div class="metric-label">Current Step</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" id="current-episode">0</div>
            <div class="metric-label">Current Episode</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" id="latest-reward">0.00</div>
            <div class="metric-label">Latest Reward</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" id="latest-total-reward">0.00</div>
            <div class="metric-label">Latest Total Reward</div>
        </div>
    </div>

    <div class="plot-container">
        <h3>📊 Real-time Step Rewards</h3>
        <div id="step-rewards-plot" style="width:100%;height:400px;"></div>
    </div>

    <div class="plot-container">
        <h3>📈 Episode Total Rewards</h3>
        <div id="episode-rewards-plot" style="width:100%;height:400px;"></div>
    </div>

    <div class="plot-container">
        <h3>🔌 Connection Log</h3>
        <div id="connection-log"></div>
    </div>

    <script>
        // WebSocket connection
        let socket;
        let stepRewardsData = [];
        let episodeRewardsData = [];
        
        function connectWebSocket() {{
            socket = new WebSocket('ws://localhost:{self.ws_port}');
            
            socket.onopen = function(event) {{
                console.log('WebSocket connected');
                document.getElementById('connection-status').textContent = 'Connected';
                document.getElementById('connection-status').className = 'status connected';
                addToLog('🟢 WebSocket connected');
                
                // Request initial data
                socket.send(JSON.stringify({{type: 'request_data'}}));
            }};
            
            socket.onmessage = function(event) {{
                const message = JSON.parse(event.data);
                handleWebSocketMessage(message);
            }};
            
            socket.onclose = function(event) {{
                console.log('WebSocket disconnected');
                document.getElementById('connection-status').textContent = 'Disconnected';
                document.getElementById('connection-status').className = 'status disconnected';
                addToLog('🔴 WebSocket disconnected - attempting reconnect...');
                
                // Attempt to reconnect after 3 seconds
                setTimeout(connectWebSocket, 3000);
            }};
            
            socket.onerror = function(error) {{
                console.error('WebSocket error:', error);
                addToLog('❌ WebSocket error: ' + error);
            }};
        }}
        
        function handleWebSocketMessage(message) {{
            switch(message.type) {{
                case 'initial_data':
                    handleInitialData(message.data);
                    break;
                case 'real_time_update':
                    handleRealTimeUpdate(message.data);
                    break;
                case 'data_update':
                    handleDataUpdate(message.data);
                    break;
            }}
        }}
        
        function handleInitialData(data) {{
            addToLog('📊 Received initial data');
            updateMetrics(data.real_time_data || {{}});
            updatePlots(data);
        }}
        
        function handleRealTimeUpdate(data) {{
            updateMetrics(data.latest_metrics || {{}});
            addToLog(`📊 Step ${{data.step_count}}, Episode ${{data.episode_count}}`);
        }}
        
        function handleDataUpdate(data) {{
            updatePlots(data);
        }}
        
        function updateMetrics(data) {{
            document.getElementById('current-step').textContent = data.current_step || 0;
            document.getElementById('current-episode').textContent = data.current_episode || 0;
            document.getElementById('latest-reward').textContent = (data.latest_reward || 0).toFixed(2);
            document.getElementById('latest-total-reward').textContent = (data.latest_total_reward || 0).toFixed(2);
        }}
        
        function updatePlots(data) {{
            // Update step rewards plot
            if (data.step_metrics) {{
                const stepData = data.step_metrics.slice(-100); // Last 100 steps
                const stepX = stepData.map(d => d.step || 0);
                const stepY = stepData.map(d => d.reward || 0);
                
                Plotly.newPlot('step-rewards-plot', [{{
                    x: stepX,
                    y: stepY,
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Step Rewards',
                    line: {{color: '#ff6b6b'}}
                }}], {{
                    title: 'Real-time Step Rewards',
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    font: {{color: 'white'}},
                    xaxis: {{gridcolor: '#444', title: 'Step'}},
                    yaxis: {{gridcolor: '#444', title: 'Reward'}}
                }});
            }}
            
            // Update episode rewards plot
            if (data.episode_metrics) {{
                const episodeData = data.episode_metrics;
                const episodeX = episodeData.map(d => d.episode || 0);
                const episodeY = episodeData.map(d => d.total_reward || 0);
                
                Plotly.newPlot('episode-rewards-plot', [{{
                    x: episodeX,
                    y: episodeY,
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Episode Total Rewards',
                    line: {{color: '#3742fa'}}
                }}], {{
                    title: 'Episode Total Rewards',
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    font: {{color: 'white'}},
                    xaxis: {{gridcolor: '#444', title: 'Episode'}},
                    yaxis: {{gridcolor: '#444', title: 'Total Reward'}}
                }});
            }}
        }}
        
        function addToLog(message) {{
            const log = document.getElementById('connection-log');
            const timestamp = new Date().toLocaleTimeString();
            log.innerHTML += `<div>[${{timestamp}}] ${{message}}</div>`;
            log.scrollTop = log.scrollHeight;
        }}
        
        // Send periodic ping to keep connection alive
        setInterval(() => {{
            if (socket && socket.readyState === WebSocket.OPEN) {{
                socket.send(JSON.stringify({{type: 'ping'}}));
            }}
        }}, 30000);
        
        // Initialize connection
        connectWebSocket();
    </script>
</body>
</html>
        """
        
        # Save dashboard HTML
        dashboard_path = os.path.join(self.log_dir, "websocket_dashboard.html")
        with open(dashboard_path, 'w', encoding='utf-8') as f:
            f.write(dashboard_html)
        
        print(f"🌐 WebSocket dashboard created: {dashboard_path}")
        return dashboard_path
    
    def finalize(self):
        """Clean shutdown of all services"""
        print("🛑 Shutting down WebSocket TensorBoard services...")
        
        # Update status
        with self.data_lock:
            self.stream_data["status"] = "completed"
        
        # Close TensorBoard writer
        self.writer.close()
        
        # Stop TensorBoard
        self.stop_tensorboard()
        
        # Stop WebSocket server
        self.ws_running = False
        
        print("✅ All services stopped")

# Integration example for your existing code
def integrate_websocket_tensorboard():
    """
    Example of how to integrate WebSocket TensorBoard with your existing code
    """
    # Replace your existing TensorBoardLogger initialization with:
    
    # OLD:
    # tensorboard_logger = TensorBoardLogger(auto_start=True)
    
    # NEW:
    tensorboard_logger = WebSocketTensorBoardManager(
        log_dir="tensorboard_logs",
        tb_port=6006,
        ws_port=8765,
        auto_start=True
    )
    
    # Create WebSocket dashboard
    dashboard_path = tensorboard_logger.create_websocket_dashboard()
    
    # Your existing logging calls work the same way:
    # tensorboard_logger.log_step_metrics(step_data)
    # tensorboard_logger.log_episode_metrics(episode_data)
    
    return tensorboard_logger

if __name__ == "__main__":
    # Test the WebSocket TensorBoard
    ws_tb = WebSocketTensorBoardManager()
    
    # Create dashboard
    dashboard_path = ws_tb.create_websocket_dashboard()
    
    # Simulate some training data
    import time
    for step in range(100):
        step_data = {
            'step': step,
            'reward': np.random.normal(0, 1),
            'total_reward': np.random.normal(10, 2),
            'tracking_reward': np.random.normal(5, 1)
        }
        ws_tb.log_step_metrics(step_data)
        time.sleep(0.1)
    
    print("✅ Test completed - check your browser!")
    input("Press Enter to stop...")
    ws_tb.finalize()
