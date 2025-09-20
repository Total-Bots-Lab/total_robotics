"""
Native Genesis Real-time Streaming Dashboard
Replaces TensorBoard with pure Genesis 3D visualization + lightweight web dashboard
"""

import json
import time
import threading
import os
from datetime import datetime
import numpy as np
from http.server import HTTPServer, SimpleHTTPRequestHandler
import socketserver
import webbrowser

class GenesisNativeStreamer:
    """
    Pure Genesis-based real-time streaming without TensorBoard
    Uses Genesis 3D viewer + lightweight web dashboard
    """
    
    def __init__(self, save_dir="genesis_streaming", port=8090):
        self.save_dir = save_dir
        self.port = port
        self.data_file = os.path.join(save_dir, "stream_data.json")
        self.is_streaming = False
        self.server_process = None
        
        # Data storage
        self.stream_data = {
            "current_episode": 0,
            "current_step": 0,
            "latest_metrics": {},
            "episode_history": [],
            "realtime_updates": [],
            "genesis_info": {
                "scene_active": False,
                "viewer_status": "initializing"
            }
        }
        
        # Create directories
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize streaming
        self._create_dashboard_html()
        self._start_web_server()
        
        print(f"🚀 Genesis Native Streamer initialized")
        print(f"🌐 Dashboard: http://localhost:{port}/dashboard.html")
        print(f"🎬 Genesis 3D Viewer: Primary visualization")
        print(f"📊 Web Metrics: Secondary dashboard")
    
    def _create_dashboard_html(self):
        """Create lightweight HTML dashboard"""
        dashboard_html = f'''<!DOCTYPE html>
<html>
<head>
    <title>Genesis Native Training Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {{
            margin: 0;
            padding: 20px;
            font-family: 'Segoe UI', Arial, sans-serif;
            background: linear-gradient(135deg, #1e3c72, #2a5298);
            color: white;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
        }}
        .dashboard-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            max-width: 1200px;
            margin: 0 auto;
        }}
        .card {{
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }}
        .metric {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 10px 0;
            padding: 10px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 5px;
        }}
        .metric-value {{
            font-weight: bold;
            font-size: 1.2em;
        }}
        .status-good {{ color: #4CAF50; }}
        .status-warning {{ color: #FF9800; }}
        .status-error {{ color: #F44336; }}
        .genesis-info {{
            background: linear-gradient(45deg, #667eea, #764ba2);
        }}
        .live-indicator {{
            display: inline-block;
            width: 10px;
            height: 10px;
            background: #4CAF50;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }}
        @keyframes pulse {{
            0% {{ opacity: 1; }}
            50% {{ opacity: 0.5; }}
            100% {{ opacity: 1; }}
        }}
        .chart-placeholder {{
            height: 200px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 5px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 15px 0;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 Genesis Native Training Dashboard</h1>
        <p><span class="live-indicator"></span> Live Training Session</p>
        <p>Primary View: Genesis 3D Viewer | Secondary Metrics: This Dashboard</p>
    </div>
    
    <div class="dashboard-grid">
        <!-- Genesis Status -->
        <div class="card genesis-info">
            <h3>🎬 Genesis 3D Viewer Status</h3>
            <div class="metric">
                <span>Scene Active:</span>
                <span id="scene-status" class="metric-value status-good">Active</span>
            </div>
            <div class="metric">
                <span>Viewer Mode:</span>
                <span id="viewer-mode" class="metric-value">Real-time 3D</span>
            </div>
            <div class="metric">
                <span>Visualization:</span>
                <span class="metric-value status-good">Native Genesis</span>
            </div>
        </div>
        
        <!-- Current Training -->
        <div class="card">
            <h3>📊 Current Training</h3>
            <div class="metric">
                <span>Episode:</span>
                <span id="current-episode" class="metric-value">0</span>
            </div>
            <div class="metric">
                <span>Step:</span>
                <span id="current-step" class="metric-value">0</span>
            </div>
            <div class="metric">
                <span>Total Reward:</span>
                <span id="total-reward" class="metric-value">0.000</span>
            </div>
            <div class="metric">
                <span>Position Error:</span>
                <span id="position-error" class="metric-value">0.000</span>
            </div>
        </div>
        
        <!-- Episode Statistics -->
        <div class="card">
            <h3>📈 Episode Statistics</h3>
            <div class="metric">
                <span>Best Reward:</span>
                <span id="best-reward" class="metric-value">0.000</span>
            </div>
            <div class="metric">
                <span>Average Reward:</span>
                <span id="avg-reward" class="metric-value">0.000</span>
            </div>
            <div class="metric">
                <span>Episodes Completed:</span>
                <span id="episodes-completed" class="metric-value">0</span>
            </div>
            <div class="metric">
                <span>Success Rate:</span>
                <span id="success-rate" class="metric-value">0%</span>
            </div>
        </div>
        
        <!-- Quick Metrics Chart -->
        <div class="card">
            <h3>📊 Quick Trend (Last 10 Episodes)</h3>
            <div class="chart-placeholder">
                <canvas id="quick-chart" width="280" height="180"></canvas>
            </div>
        </div>
        
        <!-- System Info -->
        <div class="card">
            <h3>⚙️ System Status</h3>
            <div class="metric">
                <span>Dashboard Type:</span>
                <span class="metric-value status-good">Genesis Native</span>
            </div>
            <div class="metric">
                <span>TensorBoard:</span>
                <span class="metric-value status-warning">Replaced</span>
            </div>
            <div class="metric">
                <span>Streaming:</span>
                <span id="streaming-status" class="metric-value status-good">Active</span>
            </div>
            <div class="metric">
                <span>Last Update:</span>
                <span id="last-update" class="metric-value">Loading...</span>
            </div>
        </div>
        
        <!-- Training Log -->
        <div class="card">
            <h3>📝 Recent Activity</h3>
            <div id="activity-log" style="height: 150px; overflow-y: auto; font-size: 0.9em;">
                <div>Genesis Native Dashboard initialized...</div>
            </div>
        </div>
    </div>

    <script>
        let streamData = {{}};
        let chart = null;
        
        function initChart() {{
            const canvas = document.getElementById('quick-chart');
            const ctx = canvas.getContext('2d');
            
            // Simple chart drawing function
            chart = {{
                canvas: canvas,
                ctx: ctx,
                data: [],
                draw: function() {{
                    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
                    if (this.data.length < 2) return;
                    
                    const padding = 20;
                    const width = this.canvas.width - 2 * padding;
                    const height = this.canvas.height - 2 * padding;
                    
                    // Draw grid
                    this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
                    this.ctx.lineWidth = 1;
                    for (let i = 0; i <= 5; i++) {{
                        const y = padding + (height / 5) * i;
                        this.ctx.beginPath();
                        this.ctx.moveTo(padding, y);
                        this.ctx.lineTo(this.canvas.width - padding, y);
                        this.ctx.stroke();
                    }}
                    
                    // Draw line
                    if (this.data.length > 1) {{
                        this.ctx.strokeStyle = '#4CAF50';
                        this.ctx.lineWidth = 2;
                        this.ctx.beginPath();
                        
                        const maxVal = Math.max(...this.data);
                        const minVal = Math.min(...this.data);
                        const range = maxVal - minVal || 1;
                        
                        for (let i = 0; i < this.data.length; i++) {{
                            const x = padding + (width / (this.data.length - 1)) * i;
                            const y = padding + height - ((this.data[i] - minVal) / range) * height;
                            
                            if (i === 0) {{
                                this.ctx.moveTo(x, y);
                            }} else {{
                                this.ctx.lineTo(x, y);
                            }}
                        }}
                        this.ctx.stroke();
                    }}
                }}
            }};
        }}
        
        function updateDashboard() {{
            fetch('stream_data.json')
                .then(response => response.json())
                .then(data => {{
                    streamData = data;
                    
                    // Update current training
                    document.getElementById('current-episode').textContent = data.current_episode || 0;
                    document.getElementById('current-step').textContent = data.current_step || 0;
                    
                    // Update latest metrics
                    if (data.latest_metrics) {{
                        document.getElementById('total-reward').textContent = 
                            (data.latest_metrics.total_reward || 0).toFixed(3);
                        document.getElementById('position-error').textContent = 
                            (data.latest_metrics.position_error || 0).toFixed(3);
                    }}
                    
                    // Update episode statistics
                    if (data.episode_history && data.episode_history.length > 0) {{
                        const rewards = data.episode_history.map(ep => ep.total_reward || 0);
                        const bestReward = Math.max(...rewards);
                        const avgReward = rewards.reduce((a, b) => a + b, 0) / rewards.length;
                        
                        document.getElementById('best-reward').textContent = bestReward.toFixed(3);
                        document.getElementById('avg-reward').textContent = avgReward.toFixed(3);
                        document.getElementById('episodes-completed').textContent = data.episode_history.length;
                        
                        // Update chart
                        if (chart) {{
                            chart.data = rewards.slice(-10); // Last 10 episodes
                            chart.draw();
                        }}
                    }}
                    
                    // Update Genesis status
                    document.getElementById('scene-status').textContent = 
                        data.genesis_info?.scene_active ? 'Active' : 'Inactive';
                    
                    // Update timestamp
                    document.getElementById('last-update').textContent = 
                        new Date().toLocaleTimeString();
                    
                    // Update activity log
                    if (data.realtime_updates && data.realtime_updates.length > 0) {{
                        const log = document.getElementById('activity-log');
                        const latest = data.realtime_updates.slice(-5); // Last 5 updates
                        log.innerHTML = latest.map(update => 
                            `<div>[${{new Date(update.timestamp).toLocaleTimeString()}}] ${{update.message}}</div>`
                        ).join('');
                        log.scrollTop = log.scrollHeight;
                    }}
                }})
                .catch(error => {{
                    console.log('Dashboard update pending...');
                    document.getElementById('streaming-status').textContent = 'Connecting...';
                    document.getElementById('streaming-status').className = 'metric-value status-warning';
                }});
        }}
        
        // Initialize
        initChart();
        updateDashboard();
        
        // Update every 2 seconds
        setInterval(updateDashboard, 2000);
        
        // Add activity
        setTimeout(() => {{
            const log = document.getElementById('activity-log');
            log.innerHTML += '<div>[' + new Date().toLocaleTimeString() + '] Dashboard ready for Genesis streaming</div>';
        }}, 1000);
    </script>
</body>
</html>'''
        
        dashboard_path = os.path.join(self.save_dir, "dashboard.html")
        with open(dashboard_path, 'w', encoding='utf-8') as f:
            f.write(dashboard_html)
        
        print(f"📄 Dashboard HTML created: {dashboard_path}")
    
    def _start_web_server(self):
        """Start lightweight web server"""
        try:
            import subprocess
            import sys
            
            # Create simple server script
            server_script = os.path.join(self.save_dir, "server.py")
            with open(server_script, 'w') as f:
                f.write(f'''
import os
import http.server
import socketserver
import threading

class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress server logs

os.chdir(r"{os.path.abspath(self.save_dir)}")
with socketserver.TCPServer(("", {self.port}), QuietHandler) as httpd:
    print(f"Genesis Dashboard server running on port {self.port}")
    httpd.serve_forever()
''')
            
            # Start server in background
            self.server_process = subprocess.Popen([
                sys.executable, server_script
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            time.sleep(1)
            webbrowser.open(f"http://localhost:{self.port}/dashboard.html")
            return True
            
        except Exception as e:
            print(f"⚠️ Could not start web server: {e}")
            return False
    
    def update_genesis_status(self, scene_active=True, viewer_status="active"):
        """Update Genesis viewer status"""
        self.stream_data["genesis_info"]["scene_active"] = scene_active
        self.stream_data["genesis_info"]["viewer_status"] = viewer_status
        self._save_data()
    
    def log_step_data(self, episode, step, total_reward, tracking_reward=None, position_error=None, action=None, actor_loss=None, critic_loss=None, noise_std=None):
        """Log step data - Enhanced to match TensorBoard functionality"""
        self.stream_data["current_episode"] = episode
        self.stream_data["current_step"] = step
        
        # Enhanced metrics tracking (same as TensorBoard)
        step_metrics = {
            "total_reward": total_reward,
            "tracking_reward": tracking_reward,
            "position_error": position_error,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add optional metrics if provided
        if action is not None:
            if isinstance(action, (list, tuple, np.ndarray)):
                step_metrics["action"] = list(action) if hasattr(action, '__iter__') else [action]
            else:
                step_metrics["action"] = action
                
        if actor_loss is not None:
            step_metrics["actor_loss"] = float(actor_loss)
            
        if critic_loss is not None:
            step_metrics["critic_loss"] = float(critic_loss)
            
        if noise_std is not None:
            step_metrics["exploration_noise"] = float(noise_std)
        
        self.stream_data["latest_metrics"] = step_metrics
        
        # Store step history for trending (same as TensorBoard scalars)
        if "step_history" not in self.stream_data:
            self.stream_data["step_history"] = []
        
        step_record = {
            "episode": episode,
            "step": step,
            **step_metrics
        }
        self.stream_data["step_history"].append(step_record)
        
        # Keep last 1000 steps for performance
        if len(self.stream_data["step_history"]) > 1000:
            self.stream_data["step_history"] = self.stream_data["step_history"][-1000:]
        
        # Add to realtime updates with detailed info (like TensorBoard logs)
        update_msg = f"Ep{episode} St{step}: R={total_reward:.3f}"
        if tracking_reward is not None:
            update_msg += f" TR={tracking_reward:.3f}"
        if position_error is not None:
            update_msg += f" PE={position_error:.4f}"
        if actor_loss is not None:
            update_msg += f" AL={actor_loss:.4f}"
        if critic_loss is not None:
            update_msg += f" CL={critic_loss:.4f}"
            
        self.stream_data["realtime_updates"].append({
            "message": update_msg,
            "timestamp": datetime.now().isoformat(),
            "type": "step_data"
        })
        
        # Keep only last 100 updates
        if len(self.stream_data["realtime_updates"]) > 100:
            self.stream_data["realtime_updates"] = self.stream_data["realtime_updates"][-100:]
        
        self._save_data()
    
    def log_episode_complete(self, episode, total_reward, episode_length=None, tracking_reward=None, avg_position_error=None, avg_actor_loss=None, avg_critic_loss=None, success_rate=None):
        """Log episode completion - Enhanced to match TensorBoard episode metrics"""
        episode_data = {
            "episode": episode,
            "total_reward": total_reward,
            "episode_length": episode_length,
            "tracking_reward": tracking_reward,
            "avg_position_error": avg_position_error,
            "avg_actor_loss": avg_actor_loss,
            "avg_critic_loss": avg_critic_loss,
            "success_rate": success_rate,
            "timestamp": datetime.now().isoformat()
        }
        
        # Remove None values for cleaner data
        episode_data = {k: v for k, v in episode_data.items() if v is not None}
        
        self.stream_data["episode_history"].append(episode_data)
        
        # Calculate running statistics (like TensorBoard scalars)
        if "episode_statistics" not in self.stream_data:
            self.stream_data["episode_statistics"] = {
                "total_episodes": 0,
                "best_reward": float('-inf'),
                "worst_reward": float('inf'),
                "avg_reward": 0,
                "recent_avg_reward": 0,  # Last 10 episodes
                "improvement_trend": 0
            }
        
        stats = self.stream_data["episode_statistics"]
        stats["total_episodes"] = len(self.stream_data["episode_history"])
        
        # Update best/worst rewards
        if total_reward > stats["best_reward"]:
            stats["best_reward"] = total_reward
        if total_reward < stats["worst_reward"]:
            stats["worst_reward"] = total_reward
        
        # Calculate average rewards
        all_rewards = [ep["total_reward"] for ep in self.stream_data["episode_history"]]
        stats["avg_reward"] = sum(all_rewards) / len(all_rewards)
        
        # Recent average (last 10 episodes)
        recent_rewards = all_rewards[-10:] if len(all_rewards) >= 10 else all_rewards
        stats["recent_avg_reward"] = sum(recent_rewards) / len(recent_rewards)
        
        # Improvement trend (compare first 10 vs last 10)
        if len(all_rewards) >= 20:
            first_10_avg = sum(all_rewards[:10]) / 10
            last_10_avg = sum(all_rewards[-10:]) / 10
            stats["improvement_trend"] = last_10_avg - first_10_avg
        
        # Add to realtime updates with comprehensive info
        update_msg = f"Episode {episode} COMPLETE: Reward={total_reward:.3f}"
        if episode_length:
            update_msg += f" Length={episode_length}"
        if tracking_reward is not None:
            update_msg += f" Tracking={tracking_reward:.3f}"
        if avg_position_error is not None:
            update_msg += f" AvgError={avg_position_error:.4f}"
            
        self.stream_data["realtime_updates"].append({
            "message": update_msg,
            "timestamp": datetime.now().isoformat(),
            "type": "episode_complete"
        })
        
        self._save_data()
        print(f"📊 Episode {episode} logged: Reward={total_reward:.3f} | Episodes completed: {stats['total_episodes']}")
    
    def log_hyperparameters(self, hparams):
        """Log hyperparameters (like TensorBoard hparams)"""
        if "hyperparameters" not in self.stream_data:
            self.stream_data["hyperparameters"] = {}
        
        self.stream_data["hyperparameters"].update(hparams)
        self._save_data()
        
        print(f"📝 Hyperparameters logged: {list(hparams.keys())}")
    
    def log_network_weights(self, model_name, weights_summary):
        """Log network weights summary (simplified version of TensorBoard histograms)"""
        if "network_weights" not in self.stream_data:
            self.stream_data["network_weights"] = {}
        
        self.stream_data["network_weights"][model_name] = {
            "summary": weights_summary,
            "timestamp": datetime.now().isoformat()
        }
        self._save_data()
        
        print(f"🧠 Network weights logged for {model_name}")
    
    def get_tensorboard_compatible_interface(self):
        """Returns object with TensorBoard-like interface for easy replacement"""
        class TensorBoardCompatibleInterface:
            def __init__(self, native_streamer):
                self.native = native_streamer
                self.global_step = 0
                self.current_episode = 0
            
            def log_episode_start(self, episode):
                """Start episode logging"""
                self.current_episode = episode
                self.native.update_genesis_status(scene_active=True, viewer_status=f"training_episode_{episode}")
                print(f"📊 Episode {episode} started")
            
            def log_step_metrics(self, step_data):
                """TensorBoard-compatible step logging"""
                episode = step_data.get('episode', 0)
                step = step_data.get('step', self.global_step)
                total_reward = step_data.get('total_reward', step_data.get('reward', 0))
                tracking_reward = step_data.get('tracking_reward')
                position_error = step_data.get('position_error')
                action = step_data.get('action')
                actor_loss = step_data.get('actor_loss', step_data.get('loss'))
                critic_loss = step_data.get('critic_loss')
                noise_std = step_data.get('noise_std', step_data.get('exploration_noise'))
                
                self.native.log_step_data(
                    episode=episode,
                    step=step, 
                    total_reward=total_reward,
                    tracking_reward=tracking_reward,
                    position_error=position_error,
                    action=action,
                    actor_loss=actor_loss,
                    critic_loss=critic_loss,
                    noise_std=noise_std
                )
                self.global_step += 1
            
            def log_episode_metrics(self, episode_data):
                """TensorBoard-compatible episode logging"""
                episode = episode_data.get('episode', 0)
                total_reward = episode_data.get('total_reward', 0)
                episode_length = episode_data.get('episode_length')
                tracking_reward = episode_data.get('tracking_reward')
                avg_position_error = episode_data.get('avg_position_error', episode_data.get('position_error'))
                avg_actor_loss = episode_data.get('avg_actor_loss', episode_data.get('actor_loss'))
                avg_critic_loss = episode_data.get('avg_critic_loss', episode_data.get('critic_loss'))
                success_rate = episode_data.get('success_rate')
                
                self.native.log_episode_complete(
                    episode=episode,
                    total_reward=total_reward,
                    episode_length=episode_length,
                    tracking_reward=tracking_reward,
                    avg_position_error=avg_position_error,
                    avg_actor_loss=avg_actor_loss,
                    avg_critic_loss=avg_critic_loss,
                    success_rate=success_rate
                )
            
            def log_hyperparameters(self, hparams):
                """TensorBoard-compatible hyperparameter logging"""
                self.native.log_hyperparameters(hparams)
            
            def log_network_weights(self, model, episode):
                """TensorBoard-compatible network weights logging"""
                try:
                    if hasattr(model, 'state_dict'):
                        # PyTorch model
                        state_dict = model.state_dict()
                        weights_summary = {}
                        for name, param in state_dict.items():
                            if hasattr(param, 'data'):
                                tensor_data = param.data.cpu().numpy() if hasattr(param.data, 'cpu') else param.data.numpy()
                                weights_summary[name] = {
                                    "mean": float(np.mean(tensor_data)),
                                    "std": float(np.std(tensor_data)),
                                    "min": float(np.min(tensor_data)),
                                    "max": float(np.max(tensor_data)),
                                    "shape": list(tensor_data.shape)
                                }
                        self.native.log_network_weights("actor_network", weights_summary)
                    else:
                        print(f"⚠️ Network weights logging: unsupported model type")
                except Exception as e:
                    print(f"⚠️ Network weights logging error: {e}")
            
            def finalize(self):
                """TensorBoard-compatible finalization"""
                self.native.finalize()
        
        return TensorBoardCompatibleInterface(self)
    
    def _save_data(self):
        """Save data to file for web dashboard"""
        try:
            with open(self.data_file, 'w') as f:
                json.dump(self.stream_data, f, indent=2)
        except Exception as e:
            print(f"⚠️ Data save error: {e}")
    
    def finalize(self):
        """Clean shutdown"""
        if self.server_process:
            try:
                self.server_process.terminate()
            except:
                pass
        
        # Save final data
        self.stream_data["genesis_info"]["viewer_status"] = "completed"
        self._save_data()
        
        print("✅ Genesis Native Streamer finalized")

# Enhanced TensorBoard-compatible interface for seamless replacement
class TensorBoardCompatibleInterface:
    """Complete TensorBoard replacement interface"""
    
    def __init__(self, native_streamer):
        self.native = native_streamer
        self.global_step = 0
        self.current_episode = 0
    
    def log_episode_start(self, episode):
        """Start episode logging"""
        self.current_episode = episode
        self.native.update_genesis_status(scene_active=True, viewer_status=f"training_episode_{episode}")
        print(f"📊 Episode {episode} started")
    
    def log_step_metrics(self, step_data):
        """TensorBoard-compatible step logging"""
        episode = step_data.get('episode', self.current_episode)
        step = step_data.get('step', self.global_step)
        total_reward = step_data.get('total_reward', step_data.get('reward', 0))
        tracking_reward = step_data.get('tracking_reward')
        position_error = step_data.get('position_error')
        action = step_data.get('action')
        actor_loss = step_data.get('actor_loss', step_data.get('loss'))
        critic_loss = step_data.get('critic_loss')
        noise_std = step_data.get('noise_std', step_data.get('exploration_noise'))
        
        self.native.log_step_data(
            episode=episode,
            step=step, 
            total_reward=total_reward,
            tracking_reward=tracking_reward,
            position_error=position_error,
            action=action,
            actor_loss=actor_loss,
            critic_loss=critic_loss,
            noise_std=noise_std
        )
        self.global_step += 1
    
    def log_episode_metrics(self, episode_data):
        """TensorBoard-compatible episode logging"""
        episode = episode_data.get('episode', self.current_episode)
        total_reward = episode_data.get('total_reward', 0)
        episode_length = episode_data.get('episode_length')
        tracking_reward = episode_data.get('tracking_reward')
        avg_position_error = episode_data.get('avg_position_error', episode_data.get('position_error'))
        avg_actor_loss = episode_data.get('avg_actor_loss', episode_data.get('actor_loss'))
        avg_critic_loss = episode_data.get('avg_critic_loss', episode_data.get('critic_loss'))
        success_rate = episode_data.get('success_rate')
        
        self.native.log_episode_complete(
            episode=episode,
            total_reward=total_reward,
            episode_length=episode_length,
            tracking_reward=tracking_reward,
            avg_position_error=avg_position_error,
            avg_actor_loss=avg_actor_loss,
            avg_critic_loss=avg_critic_loss,
            success_rate=success_rate
        )
    
    def log_hyperparameters(self, hparams):
        """TensorBoard-compatible hyperparameter logging"""
        self.native.log_hyperparameters(hparams)
    
    def log_network_weights(self, model, episode):
        """TensorBoard-compatible network weights logging"""
        try:
            if hasattr(model, 'state_dict'):
                # PyTorch model
                state_dict = model.state_dict()
                weights_summary = {}
                for name, param in state_dict.items():
                    if hasattr(param, 'data'):
                        tensor_data = param.data.cpu().numpy() if hasattr(param.data, 'cpu') else param.data.numpy()
                        weights_summary[name] = {
                            "mean": float(np.mean(tensor_data)),
                            "std": float(np.std(tensor_data)),
                            "min": float(np.min(tensor_data)),
                            "max": float(np.max(tensor_data)),
                            "shape": list(tensor_data.shape)
                        }
                self.native.log_network_weights("actor_network", weights_summary)
            else:
                print(f"⚠️ Network weights logging: unsupported model type")
        except Exception as e:
            print(f"⚠️ Network weights logging error: {e}")
    
    def finalize(self):
        """TensorBoard-compatible finalization"""
        self.native.finalize()

# Convenience function that returns the compatible interface directly
def create_genesis_native_tensorboard_replacement(save_dir="genesis_dashboard_logs", port=8090):
    """Create Genesis Native Dashboard with TensorBoard-compatible interface"""
    native_streamer = GenesisNativeStreamer(save_dir=save_dir, port=port)
    return native_streamer.get_tensorboard_compatible_interface()

# Usage example - replaces TensorBoard
def create_genesis_native_dashboard():
    """Create Genesis native dashboard (TensorBoard replacement)"""
    return GenesisNativeStreamer()

if __name__ == "__main__":
    # Test the native dashboard
    dashboard = create_genesis_native_dashboard()
    
    # Simulate some training data
    for episode in range(5):
        dashboard.update_genesis_status(scene_active=True)
        
        for step in range(10):
            total_reward = np.random.normal(0, 1)
            position_error = np.random.uniform(0, 0.1)
            
            dashboard.log_step_data(episode, step, total_reward, position_error=position_error)
            time.sleep(0.1)
        
        episode_reward = np.random.normal(0, 5)
        dashboard.log_episode_complete(episode, episode_reward, episode_length=10)
        time.sleep(1)
    
    print("🎉 Native Genesis dashboard test completed!")
    print("🌐 View dashboard at: http://localhost:8090/dashboard.html")
    time.sleep(30)  # Keep server running
    dashboard.finalize()
