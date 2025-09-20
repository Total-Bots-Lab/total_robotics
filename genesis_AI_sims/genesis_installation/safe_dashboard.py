"""
Safe Live Dashboard for Genesis Training
Uses separate process and file-based communication to avoid threading conflicts
"""

import json
import time
import os
import threading
import webbrowser
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from http.server import HTTPServer, SimpleHTTPRequestHandler
import socketserver
import subprocess
import sys

class SafeTrainingDashboard:
    """
    Safe dashboard that runs in separate process to avoid Genesis conflicts
    Uses file-based communication instead of threading
    """
    
    def __init__(self, save_dir="dashboard_data", port=8080):
        self.save_dir = save_dir
        self.port = port
        self.data_file = os.path.join(save_dir, "training_data.json")
        self.dashboard_html = os.path.join(save_dir, "dashboard.html")
        self.server_process = None
        
        # Create directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize data structure
        self.training_data = {
            "episodes": [],
            "step_data": [],
            "episode_summaries": [],
            "aggregated_stats": {
                "total_rewards": {"values": [], "min": [], "max": [], "mean": []},
                "tracking_rewards": {"values": [], "min": [], "max": [], "mean": []}
            },
            "metadata": {
                "start_time": datetime.now().isoformat(),
                "status": "running",
                "current_episode": 0,
                "total_episodes": 0
            }
        }
        
        self._save_data()
        self._create_dashboard_html()
        
    def log_step_data(self, episode, step, total_reward, tracking_reward, position_error=0.0):
        """Log step-level data safely"""
        step_entry = {
            "episode": episode,
            "step": step,
            "total_reward": total_reward,
            "tracking_reward": tracking_reward,
            "position_error": position_error,
            "timestamp": datetime.now().isoformat()
        }
        
        self.training_data["step_data"].append(step_entry)
        self._save_data()
        
    def log_episode_complete(self, episode, episode_rewards, episode_tracking_rewards):
        """Log episode completion with statistics"""
        # Calculate episode statistics
        episode_stats = {
            "episode": episode,
            "total_reward": {
                "sum": sum(episode_rewards),
                "min": min(episode_rewards) if episode_rewards else 0,
                "max": max(episode_rewards) if episode_rewards else 0,
                "mean": np.mean(episode_rewards) if episode_rewards else 0,
                "std": np.std(episode_rewards) if episode_rewards else 0,
                "count": len(episode_rewards)
            },
            "tracking_reward": {
                "sum": sum(episode_tracking_rewards),
                "min": min(episode_tracking_rewards) if episode_tracking_rewards else 0,
                "max": max(episode_tracking_rewards) if episode_tracking_rewards else 0,
                "mean": np.mean(episode_tracking_rewards) if episode_tracking_rewards else 0,
                "std": np.std(episode_tracking_rewards) if episode_tracking_rewards else 0,
                "count": len(episode_tracking_rewards)
            },
            "timestamp": datetime.now().isoformat()
        }
        
        self.training_data["episode_summaries"].append(episode_stats)
        self.training_data["metadata"]["current_episode"] = episode
        
        # Update aggregated statistics
        self._update_aggregated_stats()
        self._save_data()
        self._update_dashboard_plots()
        
    def _update_aggregated_stats(self):
        """Update aggregated statistics across all episodes"""
        if not self.training_data["episode_summaries"]:
            return
            
        # Extract all episode totals
        total_rewards = [ep["total_reward"]["sum"] for ep in self.training_data["episode_summaries"]]
        tracking_rewards = [ep["tracking_reward"]["sum"] for ep in self.training_data["episode_summaries"]]
        
        # Calculate rolling statistics
        self.training_data["aggregated_stats"]["total_rewards"]["values"] = total_rewards
        self.training_data["aggregated_stats"]["tracking_rewards"]["values"] = tracking_rewards
        
        # Calculate min, max, mean for each episode
        for i in range(len(total_rewards)):
            current_totals = total_rewards[:i+1]
            current_tracking = tracking_rewards[:i+1]
            
            # Update total rewards stats
            if len(self.training_data["aggregated_stats"]["total_rewards"]["min"]) <= i:
                self.training_data["aggregated_stats"]["total_rewards"]["min"].append(min(current_totals))
                self.training_data["aggregated_stats"]["total_rewards"]["max"].append(max(current_totals))
                self.training_data["aggregated_stats"]["total_rewards"]["mean"].append(np.mean(current_totals))
            else:
                self.training_data["aggregated_stats"]["total_rewards"]["min"][i] = min(current_totals)
                self.training_data["aggregated_stats"]["total_rewards"]["max"][i] = max(current_totals)
                self.training_data["aggregated_stats"]["total_rewards"]["mean"][i] = np.mean(current_totals)
                
            # Update tracking rewards stats
            if len(self.training_data["aggregated_stats"]["tracking_rewards"]["min"]) <= i:
                self.training_data["aggregated_stats"]["tracking_rewards"]["min"].append(min(current_tracking))
                self.training_data["aggregated_stats"]["tracking_rewards"]["max"].append(max(current_tracking))
                self.training_data["aggregated_stats"]["tracking_rewards"]["mean"].append(np.mean(current_tracking))
            else:
                self.training_data["aggregated_stats"]["tracking_rewards"]["min"][i] = min(current_tracking)
                self.training_data["aggregated_stats"]["tracking_rewards"]["max"][i] = max(current_tracking)
                self.training_data["aggregated_stats"]["tracking_rewards"]["mean"][i] = np.mean(current_tracking)
    
    def _save_data(self):
        """Save data to JSON file for dashboard"""
        try:
            with open(self.data_file, 'w') as f:
                json.dump(self.training_data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save dashboard data: {e}")
    
    def _update_dashboard_plots(self):
        """Generate updated plots for dashboard"""
        try:
            # Create plots directory
            plots_dir = os.path.join(self.save_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            
            # Plot 1: Episode Rewards with Min/Max/Mean
            if len(self.training_data["episode_summaries"]) > 0:
                self._create_episode_rewards_plot(plots_dir)
                self._create_step_wise_plot(plots_dir)
                self._create_aggregated_stats_plot(plots_dir)
                
        except Exception as e:
            print(f"Warning: Could not update dashboard plots: {e}")
    
    def _create_episode_rewards_plot(self, plots_dir):
        """Create episode-wise rewards plot"""
        episodes = [ep["episode"] for ep in self.training_data["episode_summaries"]]
        total_rewards = [ep["total_reward"]["sum"] for ep in self.training_data["episode_summaries"]]
        tracking_rewards = [ep["tracking_reward"]["sum"] for ep in self.training_data["episode_summaries"]]
        
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Total Rewards
        ax1.plot(episodes, total_rewards, 'o-', color='#ff6b6b', linewidth=2, markersize=6)
        ax1.set_title('Total Rewards per Episode', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.grid(True, alpha=0.3)
        
        # Tracking Rewards
        ax2.plot(episodes, tracking_rewards, 's-', color='#3742fa', linewidth=2, markersize=6)
        ax2.set_title('Tracking Rewards per Episode', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Tracking Reward')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'episode_rewards.png'), dpi=150, bbox_inches='tight', facecolor='black')
        plt.close()
    
    def _create_step_wise_plot(self, plots_dir):
        """Create step-wise rewards plot for current episode"""
        if not self.training_data["step_data"]:
            return
            
        # Get latest episode data
        current_episode = self.training_data["metadata"]["current_episode"]
        episode_steps = [s for s in self.training_data["step_data"] if s["episode"] == current_episode]
        
        if not episode_steps:
            return
            
        steps = [s["step"] for s in episode_steps]
        total_rewards = [s["total_reward"] for s in episode_steps]
        tracking_rewards = [s["tracking_reward"] for s in episode_steps]
        
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Total Rewards
        ax1.plot(steps, total_rewards, '-', color='#ff6b6b', linewidth=1, alpha=0.8)
        ax1.set_title(f'Step-wise Total Rewards - Episode {current_episode}', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Total Reward')
        ax1.grid(True, alpha=0.3)
        
        # Tracking Rewards
        ax2.plot(steps, tracking_rewards, '-', color='#3742fa', linewidth=1, alpha=0.8)
        ax2.set_title(f'Step-wise Tracking Rewards - Episode {current_episode}', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Tracking Reward')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'step_wise_rewards.png'), dpi=150, bbox_inches='tight', facecolor='black')
        plt.close()
    
    def _create_aggregated_stats_plot(self, plots_dir):
        """Create aggregated min/max/mean statistics plot"""
        agg_stats = self.training_data["aggregated_stats"]
        
        if not agg_stats["total_rewards"]["values"]:
            return
            
        episodes = list(range(1, len(agg_stats["total_rewards"]["values"]) + 1))
        
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Total Rewards Statistics
        ax1.plot(episodes, agg_stats["total_rewards"]["min"], '--', color='#ff4757', linewidth=2, label='Min', alpha=0.8)
        ax1.plot(episodes, agg_stats["total_rewards"]["mean"], '-', color='#ff6b6b', linewidth=3, label='Mean')
        ax1.plot(episodes, agg_stats["total_rewards"]["max"], '--', color='#ff9ff3', linewidth=2, label='Max', alpha=0.8)
        ax1.fill_between(episodes, agg_stats["total_rewards"]["min"], agg_stats["total_rewards"]["max"], alpha=0.2, color='#ff6b6b')
        ax1.set_title('Total Rewards - Min/Max/Mean Statistics', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Tracking Rewards Statistics
        ax2.plot(episodes, agg_stats["tracking_rewards"]["min"], '--', color='#1e3799', linewidth=2, label='Min', alpha=0.8)
        ax2.plot(episodes, agg_stats["tracking_rewards"]["mean"], '-', color='#3742fa', linewidth=3, label='Mean')
        ax2.plot(episodes, agg_stats["tracking_rewards"]["max"], '--', color='#70a1ff', linewidth=2, label='Max', alpha=0.8)
        ax2.fill_between(episodes, agg_stats["tracking_rewards"]["min"], agg_stats["tracking_rewards"]["max"], alpha=0.2, color='#3742fa')
        ax2.set_title('Tracking Rewards - Min/Max/Mean Statistics', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Tracking Reward')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'aggregated_stats.png'), dpi=150, bbox_inches='tight', facecolor='black')
        plt.close()
    
    def _create_dashboard_html(self):
        """Create HTML dashboard file with UTF-8 encoding"""
        html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Genesis Training Dashboard</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #1a1a1a;
            color: #ffffff;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .metric-card {
            background-color: #2d2d2d;
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #ff6b6b;
        }
        .metric-card.tracking {
            border-left-color: #3742fa;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .metric-label {
            color: #aaaaaa;
            font-size: 0.9em;
        }
        .plots-section {
            margin-top: 30px;
        }
        .plot-container {
            background-color: #2d2d2d;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .plot-container img {
            max-width: 100%;
            height: auto;
            border-radius: 5px;
        }
        .refresh-btn {
            background-color: #667eea;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            margin-bottom: 20px;
        }
        .refresh-btn:hover {
            background-color: #5a6fd8;
        }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-running {
            background-color: #2ed573;
            animation: pulse 2s infinite;
        }
        .status-completed {
            background-color: #ffa502;
        }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        .data-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            background-color: #2d2d2d;
        }
        .data-table th, .data-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #444;
        }
        .data-table th {
            background-color: #3d3d3d;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>&#128640; Genesis Training Dashboard</h1>
        <p><span id="status-indicator" class="status-indicator status-running"></span>Live Training Monitoring</p>
        <button class="refresh-btn" onclick="refreshDashboard()">&#128260; Refresh Data</button>
    </div>

    <div class="metrics-grid" id="metrics-grid">
        <!-- Metrics will be populated by JavaScript -->
    </div>

    <div class="plots-section">
        <div class="plot-container">
            <h3>&#128202; Episode Rewards Analysis</h3>
            <img id="episode-rewards-plot" src="plots/episode_rewards.png" alt="Episode Rewards" onerror="this.style.display='none'">
        </div>
        
        <div class="plot-container">
            <h3>&#128200; Step-wise Rewards (Current Episode)</h3>
            <img id="step-wise-plot" src="plots/step_wise_rewards.png" alt="Step-wise Rewards" onerror="this.style.display='none'">
        </div>
        
        <div class="plot-container">
            <h3>&#128202; Aggregated Statistics (Min/Max/Mean)</h3>
            <img id="aggregated-stats-plot" src="plots/aggregated_stats.png" alt="Aggregated Statistics" onerror="this.style.display='none'">
        </div>
    </div>

    <div class="plot-container">
        <h3>&#128203; Episode Summary Table</h3>
        <table class="data-table" id="episode-table">
            <thead>
                <tr>
                    <th>Episode</th>
                    <th>Total Reward (Sum)</th>
                    <th>Total Reward (Mean)</th>
                    <th>Tracking Reward (Sum)</th>
                    <th>Tracking Reward (Mean)</th>
                    <th>Steps</th>
                </tr>
            </thead>
            <tbody id="episode-table-body">
                <!-- Table rows will be populated by JavaScript -->
            </tbody>
        </table>
    </div>

    <script>
        function refreshDashboard() {
            fetch('training_data.json')
                .then(response => response.json())
                .then(data => updateDashboard(data))
                .catch(error => console.error('Error loading data:', error));
        }

        function updateDashboard(data) {
            updateMetrics(data);
            updatePlots();
            updateEpisodeTable(data);
            updateStatus(data);
        }

        function updateMetrics(data) {
            const metricsGrid = document.getElementById('metrics-grid');
            
            if (data.aggregated_stats.total_rewards.values.length === 0) {
                metricsGrid.innerHTML = '<p>No data available yet...</p>';
                return;
            }

            const totalRewards = data.aggregated_stats.total_rewards;
            const trackingRewards = data.aggregated_stats.tracking_rewards;
            const latestTotal = totalRewards.values[totalRewards.values.length - 1] || 0;
            const latestTracking = trackingRewards.values[trackingRewards.values.length - 1] || 0;

            metricsGrid.innerHTML = `
                <div class="metric-card">
                    <div class="metric-value">${latestTotal.toFixed(2)}</div>
                    <div class="metric-label">Latest Total Reward</div>
                </div>
                <div class="metric-card tracking">
                    <div class="metric-value">${latestTracking.toFixed(2)}</div>
                    <div class="metric-label">Latest Tracking Reward</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${(totalRewards.mean[totalRewards.mean.length - 1] || 0).toFixed(2)}</div>
                    <div class="metric-label">Mean Total Reward</div>
                </div>
                <div class="metric-card tracking">
                    <div class="metric-value">${(trackingRewards.mean[trackingRewards.mean.length - 1] || 0).toFixed(2)}</div>
                    <div class="metric-label">Mean Tracking Reward</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${data.metadata.current_episode}</div>
                    <div class="metric-label">Current Episode</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${data.step_data.length}</div>
                    <div class="metric-label">Total Steps</div>
                </div>
            `;
        }

        function updatePlots() {
            const timestamp = new Date().getTime();
            document.getElementById('episode-rewards-plot').src = `plots/episode_rewards.png?t=${timestamp}`;
            document.getElementById('step-wise-plot').src = `plots/step_wise_rewards.png?t=${timestamp}`;
            document.getElementById('aggregated-stats-plot').src = `plots/aggregated_stats.png?t=${timestamp}`;
        }

        function updateEpisodeTable(data) {
            const tableBody = document.getElementById('episode-table-body');
            tableBody.innerHTML = '';

            data.episode_summaries.forEach(episode => {
                const row = `
                    <tr>
                        <td>${episode.episode}</td>
                        <td>${episode.total_reward.sum.toFixed(2)}</td>
                        <td>${episode.total_reward.mean.toFixed(2)}</td>
                        <td>${episode.tracking_reward.sum.toFixed(2)}</td>
                        <td>${episode.tracking_reward.mean.toFixed(2)}</td>
                        <td>${episode.total_reward.count}</td>
                    </tr>
                `;
                tableBody.innerHTML += row;
            });
        }

        function updateStatus(data) {
            const statusIndicator = document.getElementById('status-indicator');
            if (data.metadata.status === 'completed') {
                statusIndicator.className = 'status-indicator status-completed';
            } else {
                statusIndicator.className = 'status-indicator status-running';
            }
        }

        // Auto-refresh every 5 seconds
        setInterval(refreshDashboard, 5000);
        
        // Initial load
        refreshDashboard();
    </script>
</body>
</html>
        """
        
        with open(self.dashboard_html, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def start_dashboard_server(self):
        """Start dashboard server using subprocess to avoid pickle issues"""
        try:
            # Create a simple Python script to run the server
            server_script = os.path.join(self.save_dir, "start_server.py")
            server_code = f"""
import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
import webbrowser
import time
import threading

def start_server():
    os.chdir(r"{self.save_dir}")
    handler = SimpleHTTPRequestHandler
    httpd = HTTPServer(("localhost", {self.port}), handler)
    print(f"Dashboard server running at http://localhost:{self.port}")
    httpd.serve_forever()

def open_browser():
    time.sleep(3)  # Wait for server to start
    try:
        webbrowser.open(f"http://localhost:{self.port}/dashboard.html")
        print(f"Dashboard opened: http://localhost:{self.port}/dashboard.html")
    except Exception as e:
        print(f"Could not open browser: {{e}}")

if __name__ == "__main__":
    # Start browser in a separate thread
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Start server (blocks)
    start_server()
"""
            
            # Write the server script
            with open(server_script, 'w', encoding='utf-8') as f:
                f.write(server_code)
            
            # Start server using subprocess
            import subprocess
            self.server_process = subprocess.Popen([
                sys.executable, server_script
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0)
            
            print(f"🌐 Dashboard server starting at http://localhost:{self.port}")
            print(f"🌐 Dashboard will auto-open at: http://localhost:{self.port}/dashboard.html")
            
            return True
            
        except Exception as e:
            print(f"⚠️ Could not start dashboard server: {e}")
            return False
    
    def finalize_dashboard(self):
        """Mark training as complete and generate final dashboard"""
        self.training_data["metadata"]["status"] = "completed"
        self.training_data["metadata"]["end_time"] = datetime.now().isoformat()
        
        # Generate final comprehensive plots
        self._update_dashboard_plots()
        self._save_data()
        
        print("📊 Dashboard finalized - all data saved")
        
        # Open final dashboard
        if not self.server_process:
            self.start_dashboard_server()
        
        return {
            "dashboard_url": f"http://localhost:{self.port}/dashboard.html",
            "data_file": self.data_file,
            "plots_directory": os.path.join(self.save_dir, "plots")
        }
    
    def stop_dashboard_server(self):
        """Safely stop dashboard server"""
        if self.server_process:
            try:
                # Terminate the subprocess
                if os.name == 'nt':  # Windows
                    import signal
                    self.server_process.send_signal(signal.CTRL_BREAK_EVENT)
                else:  # Unix/Linux
                    self.server_process.terminate()
                
                # Wait for process to end (with timeout)
                try:
                    self.server_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.server_process.kill()  # Force kill if it doesn't stop
                
                self.server_process = None
                print("📊 Dashboard server stopped")
            except Exception as e:
                print(f"⚠️ Error stopping dashboard server: {e}")
                # Force cleanup
                try:
                    self.server_process.kill()
                    self.server_process = None
                except:
                    pass
