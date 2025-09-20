"""
Simple WebSocket TensorBoard Integration
Drop-in replacement for standard TensorBoard with real-time WebSocket streaming
"""

import json
import threading
import time
import asyncio
import websockets
from datetime import datetime
import subprocess
import webbrowser
import os
from torch.utils.tensorboard import SummaryWriter

class SimpleTensorBoardWebSocket:
    """
    Simple WebSocket-enhanced TensorBoard that maintains compatibility
    with your existing TensorBoardLogger interface
    """
    
    def __init__(self, log_dir="tensorboard_logs", experiment_name=None, auto_start=True, ws_port=8765):
        # Initialize exactly like your existing TensorBoardLogger
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if experiment_name:
            self.log_dir = os.path.join(log_dir, f"{experiment_name}_{timestamp}")
        else:
            self.log_dir = os.path.join(log_dir, f"genesis_training_{timestamp}")
        
        # Standard TensorBoard setup
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        # WebSocket enhancement
        self.ws_port = ws_port
        self.ws_server = None
        self.connected_clients = set()
        self.real_time_data = {}
        self.data_lock = threading.Lock()
        
        # Standard TensorBoard manager (from your existing code)
        from genesis_installation.NewTest_v1_pure_env import AutoTensorBoardManager
        self.tb_manager = AutoTensorBoardManager(log_dir=log_dir, auto_open_browser=auto_start)
        
        # Episode metrics storage (from your existing code)
        self.global_step = 0
        self.episode_step = 0
        self.episode_metrics = {
            'total_rewards': [],
            'tracking_rewards': [],
            'position_errors': [],
            'episode_lengths': []
        }
        
        # Start services
        if auto_start:
            self.tb_manager.start_tensorboard()
            self.start_websocket_server()
        
        os.makedirs(self.log_dir, exist_ok=True)
        print(f"🔥 Enhanced TensorBoard with WebSocket streaming initialized")
        print(f"📊 TensorBoard: http://localhost:{self.tb_manager.port}")
        print(f"🔌 WebSocket: ws://localhost:{self.ws_port}")
    
    def start_websocket_server(self):
        """Start WebSocket server for real-time data streaming"""
        async def websocket_handler(websocket, path):
            print(f"📡 WebSocket client connected")
            self.connected_clients.add(websocket)
            
            try:
                # Send current data on connection
                with self.data_lock:
                    await websocket.send(json.dumps({
                        "type": "initial_data",
                        "data": self.real_time_data,
                        "timestamp": datetime.now().isoformat()
                    }))
                
                # Keep connection alive
                await websocket.wait_closed()
            except:
                pass
            finally:
                self.connected_clients.discard(websocket)
                print("📡 WebSocket client disconnected")
        
        def run_server():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            start_server = websockets.serve(websocket_handler, "localhost", self.ws_port)
            loop.run_until_complete(start_server)
            print(f"🔌 WebSocket server started on port {self.ws_port}")
            loop.run_forever()
        
        threading.Thread(target=run_server, daemon=True).start()
    
    async def broadcast_data(self, data):
        """Broadcast data to all connected WebSocket clients"""
        if not self.connected_clients:
            return
        
        message = json.dumps({
            "type": "real_time_update",
            "data": data,
            "timestamp": datetime.now().isoformat()
        })
        
        # Send to all clients
        disconnected = set()
        for client in self.connected_clients:
            try:
                await client.send(message)
            except:
                disconnected.add(client)
        
        # Remove disconnected clients
        self.connected_clients -= disconnected
    
    def log_step_metrics(self, step_data):
        """Enhanced step logging with WebSocket streaming"""
        # Standard TensorBoard logging (exactly like your existing code)
        step = step_data.get('step', self.global_step)
        
        if 'reward' in step_data and step_data['reward'] is not None:
            self.writer.add_scalar('Step/Reward', step_data['reward'], self.global_step)
        
        if 'total_reward' in step_data and step_data['total_reward'] is not None:
            self.writer.add_scalar('Step/Total_Reward', step_data['total_reward'], self.global_step)
        
        if 'tracking_reward' in step_data and step_data['tracking_reward'] is not None:
            self.writer.add_scalar('Step/Tracking_Reward', step_data['tracking_reward'], self.global_step)
        
        # WebSocket enhancement - real-time streaming
        with self.data_lock:
            self.real_time_data.update({
                "current_step": step,
                "latest_reward": step_data.get('reward', 0),
                "latest_total_reward": step_data.get('total_reward', 0),
                "latest_tracking_reward": step_data.get('tracking_reward', 0),
                "last_update": datetime.now().isoformat()
            })
        
        # Broadcast to WebSocket clients
        if self.connected_clients:
            asyncio.run_coroutine_threadsafe(
                self.broadcast_data(self.real_time_data),
                asyncio.get_event_loop()
            )
        
        self.global_step += 1
    
    def log_episode_metrics(self, episode_data):
        """Enhanced episode logging with WebSocket streaming"""
        # Standard TensorBoard logging (exactly like your existing code)
        episode = episode_data.get('episode', self.episode_step)
        
        if 'total_reward' in episode_data and episode_data['total_reward'] is not None:
            total_reward = episode_data['total_reward']
            self.writer.add_scalar('Episode/Total_Reward', total_reward, episode)
            self.episode_metrics['total_rewards'].append(total_reward)
        
        if 'tracking_reward' in episode_data and episode_data['tracking_reward'] is not None:
            tracking_reward = episode_data['tracking_reward']
            self.writer.add_scalar('Episode/Tracking_Reward', tracking_reward, episode)
            self.episode_metrics['tracking_rewards'].append(tracking_reward)
        
        # WebSocket enhancement
        with self.data_lock:
            self.real_time_data.update({
                "current_episode": episode,
                "latest_episode_reward": episode_data.get('total_reward', 0),
                "latest_episode_tracking": episode_data.get('tracking_reward', 0),
                "total_episodes": len(self.episode_metrics['total_rewards']),
                "episode_update": datetime.now().isoformat()
            })
        
        # Broadcast to WebSocket clients
        if self.connected_clients:
            asyncio.run_coroutine_threadsafe(
                self.broadcast_data(self.real_time_data),
                asyncio.get_event_loop()
            )
    
    # All other methods remain exactly the same as your existing TensorBoardLogger
    def log_hyperparameters(self, hparams):
        """Log hyperparameters to TensorBoard"""
        self.writer.add_hparams(hparams, {})
        print("📝 Hyperparameters logged to TensorBoard")
    
    def log_episode_start(self, episode):
        """Log episode start"""
        self.episode_step = episode
    
    def log_network_weights(self, model, step):
        """Log network weights and gradients"""
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.writer.add_histogram(f'Weights/{name}', param.data, step)
                self.writer.add_histogram(f'Gradients/{name}', param.grad.data, step)
    
    def finalize(self):
        """Close TensorBoard writer and stop services"""
        # Standard finalization (exactly like your existing code)
        if self.episode_metrics['total_rewards']:
            final_stats = {
                'Total Episodes': len(self.episode_metrics['total_rewards']),
                'Best Total Reward': max(self.episode_metrics['total_rewards']),
                'Final Total Reward': self.episode_metrics['total_rewards'][-1],
                'Average Total Reward': sum(self.episode_metrics['total_rewards']) / len(self.episode_metrics['total_rewards'])
            }
        else:
            final_stats = None
        
        self.writer.close()
        print("✅ TensorBoard logging completed")
        
        # Stop TensorBoard
        self.tb_manager.stop_tensorboard()
        print("📊 TensorBoard results saved")
        
        return final_stats

def create_websocket_dashboard_html(ws_port=8765, tb_port=6006):
    """Create a simple WebSocket dashboard HTML file"""
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Genesis AI - Real-time Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; background: #1a1a1a; color: white; margin: 20px; }}
        .header {{ text-align: center; padding: 20px; background: linear-gradient(45deg, #667eea, #764ba2); border-radius: 10px; }}
        .metrics {{ display: flex; justify-content: space-around; margin: 20px 0; }}
        .metric {{ background: #2d2d2d; padding: 20px; border-radius: 10px; text-align: center; }}
        .metric-value {{ font-size: 2em; font-weight: bold; }}
        .metric-label {{ color: #aaa; }}
        .status {{ padding: 5px 10px; border-radius: 15px; }}
        .connected {{ background: #2ed573; }}
        .disconnected {{ background: #ff4757; }}
        #log {{ background: #2d2d2d; padding: 15px; border-radius: 5px; height: 200px; overflow-y: auto; font-family: monospace; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 Genesis AI Real-time Dashboard</h1>
        <p>WebSocket: <span id="status" class="status disconnected">Disconnected</span></p>
        <p><a href="http://localhost:{tb_port}" target="_blank">Open TensorBoard</a></p>
    </div>
    
    <div class="metrics">
        <div class="metric">
            <div class="metric-value" id="step">0</div>
            <div class="metric-label">Current Step</div>
        </div>
        <div class="metric">
            <div class="metric-value" id="episode">0</div>
            <div class="metric-label">Current Episode</div>
        </div>
        <div class="metric">
            <div class="metric-value" id="reward">0.00</div>
            <div class="metric-label">Latest Reward</div>
        </div>
        <div class="metric">
            <div class="metric-value" id="total-reward">0.00</div>
            <div class="metric-label">Latest Total Reward</div>
        </div>
    </div>
    
    <h3>📊 Real-time Updates</h3>
    <div id="log"></div>
    
    <script>
        const socket = new WebSocket('ws://localhost:{ws_port}');
        const status = document.getElementById('status');
        const log = document.getElementById('log');
        
        socket.onopen = function() {{
            status.textContent = 'Connected';
            status.className = 'status connected';
            addLog('🟢 Connected to WebSocket');
        }};
        
        socket.onmessage = function(event) {{
            const message = JSON.parse(event.data);
            if (message.type === 'real_time_update' || message.type === 'initial_data') {{
                updateMetrics(message.data);
                addLog(`📊 Update: Step ${{message.data.current_step || 0}}, Episode ${{message.data.current_episode || 0}}`);
            }}
        }};
        
        socket.onclose = function() {{
            status.textContent = 'Disconnected';
            status.className = 'status disconnected';
            addLog('🔴 Disconnected from WebSocket');
        }};
        
        function updateMetrics(data) {{
            document.getElementById('step').textContent = data.current_step || 0;
            document.getElementById('episode').textContent = data.current_episode || 0;
            document.getElementById('reward').textContent = (data.latest_reward || 0).toFixed(2);
            document.getElementById('total-reward').textContent = (data.latest_total_reward || 0).toFixed(2);
        }}
        
        function addLog(message) {{
            const time = new Date().toLocaleTimeString();
            log.innerHTML += `<div>[${{time}}] ${{message}}</div>`;
            log.scrollTop = log.scrollHeight;
        }}
    </script>
</body>
</html>
    """
    
    dashboard_path = "websocket_dashboard.html"
    with open(dashboard_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"🌐 WebSocket dashboard created: {dashboard_path}")
    return dashboard_path
