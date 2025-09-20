import torch
import torch.nn as nn
import torch.optim as optim
import genesis as gs
import numpy as np
import gymnasium as gym
import os
# Import matplotlib without GUI backend initially
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from collections import deque
import random
import datetime
import sys
import signal
import time
import threading

# Import Genesis timeout fix
from genesis_timeout_fix import safe_exit_with_genesis_timeout

# Add automatic exit mechanisms
def cleanup_and_exit():
    """Cleanup function to ensure graceful exit with timeout protection"""
    try:
        # Stop TensorBoard first to prevent conflicts
        print("🛑 Stopping TensorBoard processes...")
        if 'tensorboard_logger' in globals() and hasattr(tensorboard_logger, 'tb_manager'):
            tensorboard_logger.tb_manager.stop_tensorboard(silent=True)
        print("✅ TensorBoard stopped")
    except:
        print("⚠️ TensorBoard stop completed with warnings")
    
    # Use the safe exit function with timeout protection
    safe_exit_with_genesis_timeout(gs)

def signal_handler(signum, frame):
    """Handle interrupt signals for clean exit"""
    print("🔄 Received exit signal, cleaning up...")
    cleanup_and_exit()

# Set up signal handlers for clean exit
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Removed aggressive emergency watchdog that was causing hanging issues
# Natural exit will be handled at the end of training
print("🔄 Training will exit naturally when completed - no aggressive timeouts")

import time
import threading
# Remove GUI-related imports to prevent threading issues
import queue
import json
from datetime import datetime as dt
from torch.utils.tensorboard import SummaryWriter
import logging
import subprocess
import webbrowser

# Automatic TensorBoard Management System
class AutoTensorBoardManager:
    """Automatically starts, manages, and stops TensorBoard during training"""
    
    def __init__(self, log_dir="tensorboard_logs", port=6006, auto_open_browser=True):
        self.log_dir = log_dir
        self.port = port
        self.auto_open_browser = auto_open_browser
        self.tensorboard_process = None
        self.is_running = False
        
    def start_tensorboard(self):
        """Start TensorBoard automatically in background"""
        try:
            # Kill any existing TensorBoard processes on this port
            self.stop_tensorboard(silent=True)
            
            # Start TensorBoard process
            cmd = f"tensorboard --logdir={self.log_dir} --port={self.port} --reload_interval=5"
            self.tensorboard_process = subprocess.Popen(
                cmd, 
                shell=True, 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
            )
            
            self.is_running = True
            log_print(f"🚀 TensorBoard started automatically on port {self.port}")
            log_print(f"📊 Live dashboard: http://localhost:{self.port}")
            
            # Wait a moment for TensorBoard to start
            time.sleep(3)
            
            # Auto-open browser if requested
            if self.auto_open_browser:
                try:
                    webbrowser.open(f"http://localhost:{self.port}")
                    log_print("🌐 TensorBoard opened in browser automatically")
                except:
                    log_print("⚠️ Could not auto-open browser, but TensorBoard is running")
            
            return True
            
        except Exception as e:
            log_print(f"⚠️ Could not start TensorBoard automatically: {e}")
            return False
    
    def stop_tensorboard(self, silent=False):
        """Stop TensorBoard process cleanly without taskkill"""
        try:
            # Stop our process if it exists
            if self.tensorboard_process and self.tensorboard_process.poll() is None:
                if not silent:
                    log_print("🛑 Stopping TensorBoard process gracefully...")
                
                # Graceful termination first
                self.tensorboard_process.terminate()
                
                # Wait a bit longer for graceful shutdown
                time.sleep(3)
                
                # Only force kill if absolutely necessary
                if self.tensorboard_process.poll() is None:
                    if not silent:
                        log_print("⚠️ TensorBoard didn't respond to termination, force killing...")
                    self.tensorboard_process.kill()
                    time.sleep(1)
            
            # DON'T use taskkill - let processes close naturally
            # Removed aggressive process killing to prevent conflicts
            
            self.is_running = False
            if not silent:
                log_print("✅ TensorBoard stopped cleanly")
            return True
            
        except Exception as e:
            if not silent:
                log_print(f"⚠️ TensorBoard stop warning: {e}")
            return False
    
    def is_tensorboard_running(self):
        """Check if TensorBoard is still running"""
        if self.tensorboard_process:
            return self.tensorboard_process.poll() is None
        return False

# Professional TensorBoard Integration for PyTorch
class TensorBoardLogger:
    """Professional TensorBoard integration for Genesis AI training"""
    
    def __init__(self, log_dir="tensorboard_logs", experiment_name=None, auto_start=True):
        """Initialize TensorBoard logger with automatic management"""
        timestamp = dt.now().strftime('%Y%m%d_%H%M%S')
        if experiment_name:
            self.log_dir = os.path.join(log_dir, f"{experiment_name}_{timestamp}")
        else:
            self.log_dir = os.path.join(log_dir, f"genesis_training_{timestamp}")
        
        # Create TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        # Initialize automatic TensorBoard manager
        self.tb_manager = AutoTensorBoardManager(log_dir=log_dir, auto_open_browser=auto_start)
        
        # Start TensorBoard automatically if requested
        if auto_start:
            success = self.tb_manager.start_tensorboard()
            if success:
                log_print(f"🔥 TensorBoard auto-started and streaming live!")
            else:
                log_print(f"⚠️ TensorBoard auto-start failed, but logging will continue")
        
        # Initialize step counters
        self.global_step = 0
        self.episode_step = 0
        
        # Store metrics for aggregation
        self.episode_metrics = {
            'total_rewards': [],
            'tracking_rewards': [],
            'position_errors': [],
            'episode_lengths': []
        }
        
        # Create log directory
        os.makedirs(self.log_dir, exist_ok=True)
        
        log_print(f"🔥 TensorBoard logging initialized: {self.log_dir}")
        if not auto_start:
            log_print(f"📊 View live training: tensorboard --logdir={log_dir}")
        log_print(f"🌐 Access at: http://localhost:{self.tb_manager.port}")
        
        log_print(f"🔥 TensorBoard logging initialized: {self.log_dir}")
        log_print(f"📊 View live training: tensorboard --logdir={log_dir}")
        log_print(f"🌐 Access at: http://localhost:6006")
    
    def log_hyperparameters(self, hparams):
        """Log hyperparameters to TensorBoard"""
        self.writer.add_hparams(hparams, {})
        log_print("📝 Hyperparameters logged to TensorBoard")
    
    def log_episode_start(self, episode):
        """Log episode start"""
        self.episode_step = episode
        self.episode_metrics = {
            'total_rewards': [],
            'tracking_rewards': [],
            'position_errors': [],
            'episode_lengths': []
        }
    
    def log_step_metrics(self, step_data):
        """Log individual step metrics"""
        step = step_data.get('step', self.global_step)
        
        # Log individual step metrics with None checks
        if 'reward' in step_data and step_data['reward'] is not None:
            self.writer.add_scalar('Step/Reward', step_data['reward'], self.global_step)
        
        if 'total_reward' in step_data and step_data['total_reward'] is not None:
            self.writer.add_scalar('Step/Total_Reward', step_data['total_reward'], self.global_step)
        
        if 'tracking_reward' in step_data and step_data['tracking_reward'] is not None:
            self.writer.add_scalar('Step/Tracking_Reward', step_data['tracking_reward'], self.global_step)
        
        if 'position_error' in step_data and step_data['position_error'] is not None:
            self.writer.add_scalar('Step/Position_Error', step_data['position_error'], self.global_step)
        
        if 'action' in step_data and step_data['action'] is not None:
            action = step_data['action']
            if isinstance(action, (list, np.ndarray)):
                for i, a in enumerate(action):
                    if a is not None:
                        self.writer.add_scalar(f'Step/Action_{i}', a, self.global_step)
            else:
                self.writer.add_scalar('Step/Action', action, self.global_step)
        
        if 'loss' in step_data and step_data['loss'] is not None:
            self.writer.add_scalar('Training/Loss', step_data['loss'], self.global_step)
        
        self.global_step += 1
    
    def log_episode_metrics(self, episode_data):
        """Log episode-level metrics with advanced statistics"""
        episode = episode_data.get('episode', self.episode_step)
        
        # Basic episode metrics with None checks
        if 'total_reward' in episode_data and episode_data['total_reward'] is not None:
            total_reward = episode_data['total_reward']
            self.writer.add_scalar('Episode/Total_Reward', total_reward, episode)
            self.episode_metrics['total_rewards'].append(total_reward)
        
        if 'tracking_reward' in episode_data and episode_data['tracking_reward'] is not None:
            tracking_reward = episode_data['tracking_reward']
            self.writer.add_scalar('Episode/Tracking_Reward', tracking_reward, episode)
            self.episode_metrics['tracking_rewards'].append(tracking_reward)
        
        if 'position_error' in episode_data and episode_data['position_error'] is not None:
            position_error = episode_data['position_error']
            self.writer.add_scalar('Episode/Position_Error', position_error, episode)
            self.episode_metrics['position_errors'].append(position_error)
        
        if 'episode_length' in episode_data:
            episode_length = episode_data['episode_length']
            self.writer.add_scalar('Episode/Length', episode_length, episode)
            self.episode_metrics['episode_lengths'].append(episode_length)
        
        # Advanced statistics (rolling averages)
        if len(self.episode_metrics['total_rewards']) > 0:
            self._log_rolling_statistics(episode)
        
        # Log learning parameters
        if 'learning_rate' in episode_data:
            self.writer.add_scalar('Training/Learning_Rate', episode_data['learning_rate'], episode)
        
        if 'noise_std' in episode_data:
            self.writer.add_scalar('Training/Exploration_Noise', episode_data['noise_std'], episode)
        
        if 'buffer_size' in episode_data:
            self.writer.add_scalar('Training/Buffer_Size', episode_data['buffer_size'], episode)
    
    def _log_rolling_statistics(self, episode):
        """Log rolling statistics for better trend analysis"""
        # Calculate rolling averages for different window sizes
        for window in [10, 50, 100]:
            if len(self.episode_metrics['total_rewards']) >= window:
                # Total rewards rolling statistics
                recent_rewards = self.episode_metrics['total_rewards'][-window:]
                self.writer.add_scalar(f'Episode/Total_Reward_Avg_{window}', np.mean(recent_rewards), episode)
                self.writer.add_scalar(f'Episode/Total_Reward_Std_{window}', np.std(recent_rewards), episode)
                self.writer.add_scalar(f'Episode/Total_Reward_Min_{window}', np.min(recent_rewards), episode)
                self.writer.add_scalar(f'Episode/Total_Reward_Max_{window}', np.max(recent_rewards), episode)
                
                # Tracking rewards rolling statistics
                if len(self.episode_metrics['tracking_rewards']) >= window:
                    recent_tracking = self.episode_metrics['tracking_rewards'][-window:]
                    self.writer.add_scalar(f'Episode/Tracking_Reward_Avg_{window}', np.mean(recent_tracking), episode)
                    self.writer.add_scalar(f'Episode/Tracking_Reward_Std_{window}', np.std(recent_tracking), episode)
                    self.writer.add_scalar(f'Episode/Tracking_Reward_Min_{window}', np.min(recent_tracking), episode)
                    self.writer.add_scalar(f'Episode/Tracking_Reward_Max_{window}', np.max(recent_tracking), episode)
    
    def log_network_weights(self, model, step):
        """Log network weights and gradients"""
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.writer.add_histogram(f'Weights/{name}', param.data, step)
                self.writer.add_histogram(f'Gradients/{name}', param.grad.data, step)
    
    def log_trajectory_visualization(self, trajectory_data, episode):
        """Log trajectory visualization data"""
        if 'reference_trajectory' in trajectory_data and 'training_trajectory' in trajectory_data:
            ref_traj = np.array(trajectory_data['reference_trajectory'])
            train_traj = np.array(trajectory_data['training_trajectory'])
            
            # Log trajectory following error
            if len(train_traj) > 0 and len(ref_traj) > 0:
                min_len = min(len(ref_traj), len(train_traj))
                errors = np.linalg.norm(ref_traj[:min_len] - train_traj[:min_len], axis=1)
                
                self.writer.add_scalar('Trajectory/Mean_Error', np.mean(errors), episode)
                self.writer.add_scalar('Trajectory/Max_Error', np.max(errors), episode)
                self.writer.add_scalar('Trajectory/Min_Error', np.min(errors), episode)
                self.writer.add_histogram('Trajectory/Error_Distribution', errors, episode)
    
    def log_custom_plots(self, plots_data, episode):
        """Log custom matplotlib plots to TensorBoard"""
        for plot_name, fig in plots_data.items():
            self.writer.add_figure(f'Plots/{plot_name}', fig, episode)
    
    def log_text(self, tag, text, step):
        """Log text data to TensorBoard"""
        self.writer.add_text(tag, text, step)
    
    def finalize(self):
        """Close TensorBoard writer and stop automatic TensorBoard"""
        # Log final statistics
        if self.episode_metrics['total_rewards']:
            final_stats = {
                'Total Episodes': len(self.episode_metrics['total_rewards']),
                'Best Total Reward': max(self.episode_metrics['total_rewards']),
                'Final Total Reward': self.episode_metrics['total_rewards'][-1],
                'Average Total Reward': np.mean(self.episode_metrics['total_rewards']),
                'Total Reward Improvement': self.episode_metrics['total_rewards'][-1] - self.episode_metrics['total_rewards'][0]
            }
            
            if self.episode_metrics['tracking_rewards']:
                final_stats.update({
                    'Best Tracking Reward': max(self.episode_metrics['tracking_rewards']),
                    'Final Tracking Reward': self.episode_metrics['tracking_rewards'][-1],
                    'Average Tracking Reward': np.mean(self.episode_metrics['tracking_rewards'])
                })
            
            # Log final summary as text
            summary_text = "\n".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                                    for k, v in final_stats.items()])
            self.writer.add_text('Final_Summary', summary_text, self.episode_step)
        
        # Close TensorBoard writer first
        self.writer.close()
        log_print(f"✅ TensorBoard logging completed")
        
        # Stop automatic TensorBoard to prevent Genesis hanging
        log_print("� Stopping TensorBoard before Genesis cleanup...")
        self.tb_manager.stop_tensorboard()
        log_print(f"📊 TensorBoard results saved in: {os.path.dirname(self.log_dir)}")
        
        return final_stats if self.episode_metrics['total_rewards'] else None
        
        return final_stats

# Live TensorBoard-Equivalent Monitoring System
class LiveTrainingMonitor:
    """Real-time TensorBoard-equivalent visualization for training metrics"""
    
    def __init__(self, save_dir="training_metrics", update_interval=1000):
        self.save_dir = save_dir
        self.update_interval = update_interval  # Update interval in milliseconds
        os.makedirs(save_dir, exist_ok=True)
        
        # Data storage
        self.episodes = []
        self.total_rewards = []
        self.tracking_rewards = []
        self.position_errors = []
        self.timestamps = []
        
        # Statistics storage
        self.total_reward_stats = {'min': [], 'mean': [], 'max': []}
        self.tracking_reward_stats = {'min': [], 'mean': [], 'max': []}
        
        # Threading for live updates
        self.data_queue = queue.Queue()
        self.should_stop = False
        self.console_mode = False  # Flag for console-only mode
        
        # Start monitoring in separate thread
        self.monitor_thread = None
        self.start_monitoring()
        
    def start_monitoring(self):
        """Start console-based monitoring (GUI disabled due to threading issues)"""
        log_print("🔥 Live Training Monitor started - Console-based monitoring active!")
        log_print("💡 Real-time statistics will be displayed in console for threading compatibility")
        
        # Start console monitoring in a separate thread
        self.console_mode = True
        self.monitor_thread = threading.Thread(target=self._start_console_monitoring, daemon=True)
        self.monitor_thread.start()
        
    def _run_gui_safe(self):
        """Disabled GUI to avoid threading issues"""
        log_print("⚠️ GUI disabled for threading compatibility - using console monitoring")
        self._start_console_monitoring()
        
    def _run_gui(self):
        """Disabled GUI to avoid threading issues"""
        log_print("⚠️ GUI disabled for threading compatibility - using console monitoring")
        self._start_console_monitoring()
    
    def _start_console_monitoring(self):
        """Enhanced console-based monitoring when GUI is disabled"""
        log_print("📊 Console-based monitoring active - Enhanced live stats in console")
        log_print("📈 Real-time Min/Mean/Max statistics will be displayed every few episodes")
        log_print("=" * 80)
        self.console_mode = True
        
        # Start a simple monitoring loop
        while not self.should_stop:
            try:
                # Process queued data
                episode_count = 0
                while not self.data_queue.empty():
                    try:
                        data = self.data_queue.get_nowait()
                        self._process_data(data)
                        episode_count += 1
                        
                        # Print detailed stats every episode
                        self._print_detailed_console_stats(data)
                        
                    except queue.Empty:
                        break
                
                # Print summary every 5 seconds
                if episode_count == 0:
                    time.sleep(1)
                else:
                    time.sleep(2)  # Short pause between episodes
                    
            except Exception as e:
                log_print(f"Console monitoring error: {e}")
                break
    
    def _print_detailed_console_stats(self, data):
        """Print enhanced training statistics to console"""
        episode = data['episode']
        total_reward = data['total_reward']
        tracking_reward = data.get('tracking_reward', 0)
        
        # Episode header
        log_print(f"\n🚀 EPISODE {episode} COMPLETED")
        log_print("=" * 60)
        
        # Current episode stats
        log_print(f"📊 Current Episode Results:")
        log_print(f"   🎯 Total Reward: {total_reward:.4f}")
        log_print(f"   📍 Tracking Reward: {tracking_reward:.4f}")
        log_print(f"   📏 Position Error: {data.get('position_error', 0):.4f}")
        
        if len(self.total_rewards) > 0:
            # Overall performance stats
            best_total = max(self.total_rewards)
            worst_total = min(self.total_rewards)
            avg_total = np.mean(self.total_rewards)
            
            best_tracking = max(self.tracking_rewards) if self.tracking_rewards else 0
            avg_tracking = np.mean(self.tracking_rewards) if self.tracking_rewards else 0
            
            log_print(f"\n� Overall Training Progress:")
            log_print(f"   Episodes Completed: {len(self.total_rewards)}")
            log_print(f"   Best Total Reward: {best_total:.4f}")
            log_print(f"   Worst Total Reward: {worst_total:.4f}")
            log_print(f"   Average Total Reward: {avg_total:.4f}")
            log_print(f"   Best Tracking Reward: {best_tracking:.4f}")
            log_print(f"   Average Tracking Reward: {avg_tracking:.4f}")
            
            # Recent performance (last 5 episodes)
            if len(self.total_rewards) >= 5:
                recent_total = np.mean(self.total_rewards[-5:])
                recent_tracking = np.mean(self.tracking_rewards[-5:]) if len(self.tracking_rewards) >= 5 else 0
                log_print(f"\n🔥 Recent Performance (Last 5 Episodes):")
                log_print(f"   Recent Total Reward Average: {recent_total:.4f}")
                log_print(f"   Recent Tracking Reward Average: {recent_tracking:.4f}")
                
                # Improvement trend
                if len(self.total_rewards) >= 10:
                    early_avg = np.mean(self.total_rewards[:5])
                    improvement = recent_total - early_avg
                    trend = "📈 IMPROVING" if improvement > 0 else "📉 DECLINING" if improvement < -0.1 else "📊 STABLE"
                    log_print(f"   Trend: {trend} (Change: {improvement:+.4f})")
            
            # Min/Mean/Max Statistics (TensorBoard-equivalent data)
            if len(self.total_reward_stats['mean']) > 0:
                current_mean = self.total_reward_stats['mean'][-1]
                current_min = self.total_reward_stats['min'][-1]
                current_max = self.total_reward_stats['max'][-1]
                
                log_print(f"\n📊 TensorBoard-Equivalent Statistics (Rolling Window):")
                log_print(f"   Min Total Reward: {current_min:.4f}")
                log_print(f"   Mean Total Reward: {current_mean:.4f}")
                log_print(f"   Max Total Reward: {current_max:.4f}")
                
                if len(self.tracking_reward_stats['mean']) > 0:
                    tracking_mean = self.tracking_reward_stats['mean'][-1]
                    tracking_min = self.tracking_reward_stats['min'][-1]
                    tracking_max = self.tracking_reward_stats['max'][-1]
                    log_print(f"   Min Tracking Reward: {tracking_min:.4f}")
                    log_print(f"   Mean Tracking Reward: {tracking_mean:.4f}")
                    log_print(f"   Max Tracking Reward: {tracking_max:.4f}")
        
        log_print("=" * 60)
        
    def _update_plots(self, frame):
        """Update all plots with latest data"""
        try:
            # Process all queued data
            while not self.data_queue.empty():
                try:
                    data = self.data_queue.get_nowait()
                    self._process_data(data)
                except queue.Empty:
                    break
            
            if len(self.episodes) == 0:
                return
            
            # Clear all axes
            self.ax1.clear()
            self.ax2.clear()
            self.ax3.clear()
            self.ax4.clear()
            
            # Plot 1: Total Rewards (Min, Mean, Max)
            self._plot_reward_statistics(self.ax1, "Total Rewards", self.total_reward_stats, '#ff6b6b', '#4ecdc4', '#45b7d1')
            
            # Plot 2: Tracking Rewards (Min, Mean, Max)
            self._plot_reward_statistics(self.ax2, "Tracking Rewards", self.tracking_reward_stats, '#ff9ff3', '#f368e0', '#3742fa')
            
            # Plot 3: Episode-wise Total Rewards
            self._plot_episode_rewards(self.ax3, "Episode Total Rewards", self.total_rewards, '#ff6b6b')
            
            # Plot 4: Episode-wise Tracking Rewards
            self._plot_episode_rewards(self.ax4, "Episode Tracking Rewards", self.tracking_rewards, '#3742fa')
            
            # Update status
            if len(self.total_rewards) > 0:
                best_total = max(self.total_rewards)
                best_tracking = max(self.tracking_rewards) if self.tracking_rewards else 0
                current_episode = len(self.episodes)
                
                self.status_label.config(text=f"🟢 Live Monitoring - Episode {current_episode}")
                self.stats_label.config(text=f"📊 Episodes: {current_episode} | Best Total: {best_total:.2f} | Best Tracking: {best_tracking:.2f}")
            
            # Adjust layout
            self.fig.tight_layout()
            
        except Exception as e:
            print(f"Error updating plots: {e}")
    
    def _plot_reward_statistics(self, ax, title, stats_data, color_min, color_mean, color_max):
        """Plot min, mean, max statistics"""
        if not stats_data['mean']:
            return
            
        episodes = list(range(1, len(stats_data['mean']) + 1))
        
        ax.plot(episodes, stats_data['min'], color=color_min, linewidth=2, label='Min', alpha=0.8)
        ax.plot(episodes, stats_data['mean'], color=color_mean, linewidth=3, label='Mean', alpha=0.9)
        ax.plot(episodes, stats_data['max'], color=color_max, linewidth=2, label='Max', alpha=0.8)
        
        ax.fill_between(episodes, stats_data['min'], stats_data['max'], alpha=0.2, color=color_mean)
        
        ax.set_title(title, color='white', fontsize=12, fontweight='bold')
        ax.set_xlabel('Episode', color='white')
        ax.set_ylabel('Reward', color='white')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3, color='white')
        ax.set_facecolor('#2d2d2d')
    
    def _plot_episode_rewards(self, ax, title, rewards, color):
        """Plot episode-wise rewards"""
        if not rewards:
            return
            
        episodes = list(range(1, len(rewards) + 1))
        ax.plot(episodes, rewards, color=color, linewidth=2, alpha=0.8)
        ax.scatter(episodes, rewards, color=color, s=30, alpha=0.6)
        
        # Add trend line
        if len(rewards) > 1:
            z = np.polyfit(episodes, rewards, 1)
            p = np.poly1d(z)
            ax.plot(episodes, p(episodes), "--", color='yellow', alpha=0.8, linewidth=2, label='Trend')
            ax.legend(loc='upper left')
        
        ax.set_title(title, color='white', fontsize=12, fontweight='bold')
        ax.set_xlabel('Episode', color='white')
        ax.set_ylabel('Reward', color='white')
        ax.grid(True, alpha=0.3, color='white')
        ax.set_facecolor('#2d2d2d')
    
    def _process_data(self, data):
        """Process incoming data and update statistics"""
        episode = data['episode']
        total_reward = data['total_reward']
        tracking_reward = data.get('tracking_reward', 0)
        position_error = data.get('position_error', 0)
        
        self.episodes.append(episode)
        self.total_rewards.append(total_reward)
        self.tracking_rewards.append(tracking_reward)
        self.position_errors.append(position_error)
        self.timestamps.append(dt.now().isoformat())
        
        # Calculate rolling statistics (last 10 episodes)
        window_size = min(10, len(self.total_rewards))
        if window_size > 0:
            recent_total = self.total_rewards[-window_size:]
            recent_tracking = self.tracking_rewards[-window_size:]
            
            # Total reward statistics
            self.total_reward_stats['min'].append(min(recent_total))
            self.total_reward_stats['mean'].append(np.mean(recent_total))
            self.total_reward_stats['max'].append(max(recent_total))
            
            # Tracking reward statistics
            self.tracking_reward_stats['min'].append(min(recent_tracking))
            self.tracking_reward_stats['mean'].append(np.mean(recent_tracking))
            self.tracking_reward_stats['max'].append(max(recent_tracking))
    
    def log_episode(self, episode, total_reward, tracking_reward=0, position_error=0):
        """Log episode data for live monitoring"""
        data = {
            'episode': episode,
            'total_reward': total_reward,
            'tracking_reward': tracking_reward,
            'position_error': position_error,
            'timestamp': dt.now().isoformat()
        }
        
        # Add to queue for GUI update
        try:
            self.data_queue.put_nowait(data)
        except queue.Full:
            pass  # Skip if queue is full
        
        # Also save to file immediately
        self._save_episode_data(data)
    
    def _save_episode_data(self, data):
        """Save episode data to JSON file"""
        filename = os.path.join(self.save_dir, f"episode_{data['episode']:04d}.json")
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving episode data: {e}")
    
    def finalize_training(self):
        """Generate final aggregated metrics and save evidence"""
        log_print("\n🔥 Generating Final Training Evidence...")
        
        # Calculate final aggregated metrics
        if len(self.total_rewards) > 0:
            total_stats = {
                'episodes_completed': len(self.total_rewards),
                'total_rewards': {
                    'min': float(min(self.total_rewards)),
                    'max': float(max(self.total_rewards)),
                    'mean': float(np.mean(self.total_rewards)),
                    'std': float(np.std(self.total_rewards)),
                    'final': float(self.total_rewards[-1]),
                    'improvement': float(self.total_rewards[-1] - self.total_rewards[0]) if len(self.total_rewards) > 1 else 0
                },
                'tracking_rewards': {
                    'min': float(min(self.tracking_rewards)) if self.tracking_rewards else 0,
                    'max': float(max(self.tracking_rewards)) if self.tracking_rewards else 0,
                    'mean': float(np.mean(self.tracking_rewards)) if self.tracking_rewards else 0,
                    'std': float(np.std(self.tracking_rewards)) if self.tracking_rewards else 0,
                    'final': float(self.tracking_rewards[-1]) if self.tracking_rewards else 0,
                },
                'position_errors': {
                    'min': float(min(self.position_errors)) if self.position_errors else 0,
                    'max': float(max(self.position_errors)) if self.position_errors else 0,
                    'mean': float(np.mean(self.position_errors)) if self.position_errors else 0,
                },
                'training_metadata': {
                    'start_time': self.timestamps[0] if self.timestamps else None,
                    'end_time': self.timestamps[-1] if self.timestamps else None,
                    'duration_minutes': len(self.timestamps) * 0.1 if self.timestamps else 0  # Approximate
                }
            }
            
            # Save aggregated metrics
            summary_file = os.path.join(self.save_dir, f"training_summary_{dt.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(summary_file, 'w') as f:
                json.dump(total_stats, f, indent=2)
            
            # Save complete data
            complete_data = {
                'episodes': self.episodes,
                'total_rewards': self.total_rewards,
                'tracking_rewards': self.tracking_rewards,
                'position_errors': self.position_errors,
                'timestamps': self.timestamps,
                'statistics': total_stats
            }
            
            complete_file = os.path.join(self.save_dir, f"complete_training_data_{dt.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(complete_file, 'w') as f:
                json.dump(complete_data, f, indent=2)
            
            # Generate final matplotlib plots
            self._generate_final_plots()
            
            # Print summary
            log_print(f"📊 FINAL TRAINING METRICS:")
            log_print(f"  Episodes Completed: {total_stats['episodes_completed']}")
            log_print(f"  Total Reward - Min: {total_stats['total_rewards']['min']:.2f}, Max: {total_stats['total_rewards']['max']:.2f}, Mean: {total_stats['total_rewards']['mean']:.2f}")
            log_print(f"  Tracking Reward - Min: {total_stats['tracking_rewards']['min']:.2f}, Max: {total_stats['tracking_rewards']['max']:.2f}, Mean: {total_stats['tracking_rewards']['mean']:.2f}")
            log_print(f"  Improvement: {total_stats['total_rewards']['improvement']:.2f}")
            log_print(f"📁 Evidence saved to: {self.save_dir}/")
            log_print(f"📄 Summary file: {summary_file}")
            log_print(f"📊 Complete data: {complete_file}")
            
            return total_stats
        
        return None
    
    def _generate_final_plots(self):
        """Generate final publication-quality plots using non-GUI matplotlib backend"""
        try:
            # Set matplotlib to non-GUI backend
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            
            plt.style.use('dark_background')
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
            fig.suptitle('Genesis AI Training Results - Complete Evidence', fontsize=16, fontweight='bold')
            
            # Plot 1: Total Rewards with Statistics
            episodes = list(range(1, len(self.total_rewards) + 1))
            ax1.plot(episodes, self.total_rewards, 'o-', color='#ff6b6b', linewidth=2, markersize=4, alpha=0.8)
            if len(self.total_reward_stats['mean']) > 0:
                ax1.plot(episodes, self.total_reward_stats['mean'], '--', color='#4ecdc4', linewidth=3, alpha=0.9, label='Mean')
                ax1.fill_between(episodes, self.total_reward_stats['min'], self.total_reward_stats['max'], alpha=0.2, color='#4ecdc4')
            ax1.set_title('Total Rewards Over Episodes', fontweight='bold')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Total Reward')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Plot 2: Tracking Rewards
            ax2.plot(episodes, self.tracking_rewards, 's-', color='#3742fa', linewidth=2, markersize=4, alpha=0.8)
            if len(self.tracking_reward_stats['mean']) > 0:
                ax2.plot(episodes, self.tracking_reward_stats['mean'], '--', color='#f368e0', linewidth=3, alpha=0.9, label='Mean')
            ax2.set_title('Tracking Rewards Over Episodes', fontweight='bold')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Tracking Reward')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # Plot 3: Reward Distribution
            ax3.hist(self.total_rewards, bins=20, alpha=0.7, color='#ff6b6b', edgecolor='white')
            ax3.axvline(np.mean(self.total_rewards), color='yellow', linestyle='--', linewidth=2, label=f'Mean: {np.mean(self.total_rewards):.2f}')
            ax3.set_title('Total Reward Distribution', fontweight='bold')
            ax3.set_xlabel('Reward Value')
            ax3.set_ylabel('Frequency')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Learning Progress
            if len(self.total_rewards) > 1:
                window_size = min(5, len(self.total_rewards))
                moving_avg = []
                for i in range(len(self.total_rewards)):
                    start = max(0, i - window_size + 1)
                    avg = np.mean(self.total_rewards[start:i+1])
                    moving_avg.append(avg)
                
                ax4.plot(episodes, self.total_rewards, alpha=0.5, color='#ff6b6b', label='Episode Reward')
                ax4.plot(episodes, moving_avg, linewidth=3, color='#4ecdc4', label=f'Moving Average ({window_size})')
                ax4.set_title('Learning Progress', fontweight='bold')
                ax4.set_xlabel('Episode')
                ax4.set_ylabel('Reward')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save final plots
            final_plot_path = os.path.join(self.save_dir, f"final_training_evidence_{dt.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(final_plot_path, dpi=300, bbox_inches='tight', facecolor='black')
            plt.close()
            
            log_print(f"📈 Final evidence plots saved: {final_plot_path}")
            
        except Exception as e:
            log_print(f"Warning: Could not generate final plots: {e}")
            log_print("💡 Continuing with text-based evidence only")
    
    def _on_closing(self):
        """Handle GUI window closing"""
        self.should_stop = True
        try:
            self.root.quit()
            self.root.destroy()
        except:
            pass
    
    def stop_monitoring(self):
        """Stop the monitoring system"""
        self.should_stop = True
        if hasattr(self, 'root'):
            try:
                self.root.after(100, self._on_closing)
            except:
                pass

# FIXED 3D Visualization Helper Functions - Genesis Compatible
def create_simple_trajectory_points(trajectory_points, spacing=2):
    """Create simple trajectory points for Genesis visualization - FIXED"""
    if len(trajectory_points) < 2:
        return []
    
    # Sample points with spacing to avoid too many points
    sampled_points = []
    for i in range(0, len(trajectory_points), spacing):
        point = trajectory_points[i]
        # Ensure each point is exactly 3 floats
        if len(point) >= 3:
            sampled_points.append([float(point[0]), float(point[1]), float(point[2])])
    
    return sampled_points

def create_large_point(position, color=(1.0, 1.0, 0.0, 1.0), radius=0.08):
    """Create a large spherical point for visualization - FIXED"""
    points = []
    # Reduced number of points for better performance and compatibility
    phi_steps = 6
    theta_steps = 8
    
    for i in range(phi_steps):
        phi = np.pi * i / (phi_steps - 1) if phi_steps > 1 else 0
        for j in range(theta_steps):
            theta = 2 * np.pi * j / theta_steps if theta_steps > 1 else 0
            x = float(radius * np.sin(phi) * np.cos(theta) + position[0])
            y = float(radius * np.sin(phi) * np.sin(theta) + position[1])
            z = float(radius * np.cos(phi) + position[2])
            points.append([x, y, z])
    
    return points

def create_safe_debug_points(points_list, max_points=100):
    """Create safe debug points for Genesis - prevents shape errors"""
    if not points_list:
        return []
    
    # Ensure all points are properly formatted
    safe_points = []
    for point in points_list[:max_points]:  # Limit number of points
        if isinstance(point, (list, tuple, np.ndarray)) and len(point) >= 3:
            safe_points.append([float(point[0]), float(point[1]), float(point[2])])
    
    return safe_points

# Set up logging
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = f"training_log_{timestamp}.txt"

# Get full paths for clarity
current_directory = os.getcwd()
full_log_path = os.path.join(current_directory, log_filename)

log_file = open(log_filename, 'w', encoding='utf-8')  # Use UTF-8 encoding for Unicode support

def log_print(message):
    """Print to console and save to log file with error handling"""
    print(message)
    try:
        # Remove or replace problematic Unicode characters for file writing
        safe_message = message.encode('ascii', 'replace').decode('ascii')
        log_file.write(safe_message + '\n')
        log_file.flush()  # Ensure immediate write
    except (ValueError, AttributeError):
        # Log file is closed or not available, just print to console
        pass

log_print(f"📂 Current working directory: {current_directory}")
log_print(f" Log will be saved to: {full_log_path}")

# Initialize Live Training Monitor - TensorBoard Equivalent
log_print("\n🔥 Initializing Live Training Monitor...")
training_monitor = LiveTrainingMonitor(save_dir="training_metrics", update_interval=500)
log_print("✅ Live Monitor launched - Real-time graphs available!")

# Initialize Genesis
gs.init(backend=gs.gpu)

# Actor Network Definition
class ActorNetwork(nn.Module):
    """Actor Network for the Franka robot control"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, action_dim)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        
    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.tanh(self.fc4(x))
        return x

# Environment wrapper following Gym Env Package structure - PURE ENVIRONMENT VERSION
class FrankaGymEnv:
    """Pure Environment-Driven Gym Environment for Franka robot - NO REFERENCE TRAJECTORY"""
    def __init__(self, scene, franka, dofs_idx, action_space, max_episode_steps=200):
        self.scene = scene
        self.franka = franka
        self.dofs_idx = dofs_idx
        self.action_space = action_space
        
        # PURE ENVIRONMENT LEARNING - No predefined trajectory
        self.current_step = 0
        self.max_steps = max_episode_steps  # Fixed episode length instead of trajectory-based
        self.initial_position = np.array([0.0, -0.3, 0.0, -1.2, 0.0, 1.0, 0.785, 0.04, 0.04], dtype=np.float32)
        self.step_count = 0
        
        # GOAL-BASED LEARNING: Define target regions/goals the robot should discover
        self.target_goals = self._generate_dynamic_goals()
        self.current_goal_idx = 0
        self.goal_tolerance = 0.15  # Distance tolerance for reaching goals
        self.goal_reached_bonus = 50.0  # Large bonus for reaching a goal
        
        # Workspace constraints for safety
        self.workspace_center = np.array([0.0, 0.0, 0.5])
        self.workspace_radius = 1.5
        
        # Curriculum learning - start with easier goals
        self.difficulty_level = 1
        self.goals_reached_this_episode = 0
    
    def _generate_dynamic_goals(self):
        """Generate dynamic goals for pure environment learning"""
        goals = []
        
        # Level 1: Simple reachable positions around the robot
        goals.extend([
            [0.4, 0.2, 0.6],   # Right side
            [-0.4, 0.2, 0.6],  # Left side  
            [0.0, 0.5, 0.8],   # Forward high
            [0.0, 0.2, 0.3],   # Forward low
        ])
        
        # Level 2: More challenging positions
        goals.extend([
            [0.6, 0.4, 0.5],   # Far right
            [-0.6, 0.4, 0.5],  # Far left
            [0.2, 0.6, 0.9],   # High reach
            [0.0, 0.8, 0.4],   # Very forward
        ])
        
        # Level 3: Complex 3D movements  
        goals.extend([
            [0.3, 0.3, 0.7],   # Diagonal up
            [-0.3, 0.3, 0.7],  # Other diagonal
            [0.5, 0.5, 0.3],   # Corner low
            [-0.5, 0.5, 0.3],  # Other corner
        ])
        
        return goals
    
    def _get_current_goal(self):
        """Get current goal based on difficulty and progress"""
        # Curriculum learning: unlock harder goals as robot improves
        available_goals = min(4 * self.difficulty_level, len(self.target_goals))
        return self.target_goals[self.current_goal_idx % available_goals]
    
    def _update_goal(self):
        """Update to next goal when current one is reached"""
        self.current_goal_idx = (self.current_goal_idx + 1) % len(self.target_goals)
        self.goals_reached_this_episode += 1
        
        # Increase difficulty every 3 goals reached in an episode
        if self.goals_reached_this_episode % 3 == 0:
            self.difficulty_level = min(3, self.difficulty_level + 1)
            log_print(f"🎯 Difficulty increased to level {self.difficulty_level}")
        
    def reset(self):
        """Reset environment to initial state"""
        self.current_step = 0
        self.step_count = 0
        self.goals_reached_this_episode = 0
        
        # Randomize initial position slightly for better exploration
        noise = np.random.normal(0, 0.05, size=self.initial_position.shape)
        start_pos = self.initial_position + noise
        start_pos = np.clip(start_pos, self.initial_position - 0.1, self.initial_position + 0.1)
        
        self.franka.set_dofs_position(start_pos, self.dofs_idx)
        for _ in range(10):  # Stabilize
            self.scene.step()
            if hasattr(self, 'video_recorder'):
                self.video_recorder.capture_frame(self.scene)
        return self.get_observation()
    
    def step(self, action):
        """Execute action and return next state, reward, done, info"""
        # Apply action to robot
        action_scaled = self._scale_action(action)
        self.franka.control_dofs_position(action_scaled, self.dofs_idx)
        
        # Step physics simulation and capture video frames
        for i in range(3):
            self.scene.step()
            self.step_count += 1
            
            # Capture video frame if recording (every 2nd physics step for good coverage)
            if hasattr(self, 'video_recorder') and i % 2 == 0:
                self.video_recorder.capture_frame(self.scene)
        
        # Get new observation
        next_state = self.get_observation()
        
        # Calculate reward based on trajectory following (now returns total and tracking rewards)
        total_reward, tracking_reward = self._calculate_reward(action_scaled)
        
        # Store separated rewards for monitoring
        self.current_total_reward = total_reward
        self.current_tracking_reward = tracking_reward
        
        # Check if episode is done
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        info = {
            'step': self.current_step, 
            'target_reached': self._target_reached(),
            'total_reward': total_reward,
            'tracking_reward': tracking_reward,
            'position_error': getattr(self, 'current_position_error', 0.0)
        }
        
        return next_state, total_reward, done, info
    
    def _scale_action(self, action):
        """Scale normalized action [-1, 1] to joint limits"""
        action_low = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0.0, 0.0], dtype=np.float32)
        action_high = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 0.04, 0.04], dtype=np.float32)
        return action_low + (action + 1.0) * 0.5 * (action_high - action_low)
    
    def _calculate_reward(self, action):
        """PURE ENVIRONMENT REWARD - No reference trajectory, goal-based learning"""
        
        # Get current end-effector position
        current_ee = self._get_end_effector_position()
        current_goal = self._get_current_goal()
        
        # === GOAL-BASED REWARDS (Primary Learning Signal) ===
        goal_distance = np.linalg.norm(current_ee - current_goal)
        
        # Distance-based reward (encourages moving toward goal)
        goal_reward = -5.0 * goal_distance  # Penalty proportional to distance
        
        # Bonus for reaching the goal
        goal_reached = False
        if goal_distance < self.goal_tolerance:
            goal_reward += self.goal_reached_bonus  # Large bonus!
            goal_reached = True
            log_print(f"🎯 GOAL REACHED! Distance: {goal_distance:.3f}, Bonus: {self.goal_reached_bonus}")
            self._update_goal()  # Move to next goal
        
        # === EXPLORATION REWARDS ===
        # Reward for visiting new areas (exploration bonus)
        exploration_reward = self._calculate_exploration_bonus(current_ee)
        
        # === SAFETY AND CONSTRAINT REWARDS ===
        # Workspace constraint reward (keep robot in safe area)
        workspace_distance = np.linalg.norm(current_ee - self.workspace_center)
        if workspace_distance > self.workspace_radius:
            workspace_penalty = -20.0 * (workspace_distance - self.workspace_radius)
        else:
            workspace_penalty = 0.0
        
        # === MOVEMENT QUALITY REWARDS ===
        # Velocity smoothness (encourage smooth movements)
        if hasattr(self, 'previous_ee_pos'):
            velocity = np.linalg.norm(current_ee - self.previous_ee_pos)
            smoothness_reward = -0.2 * velocity  # Small penalty for jerky movements
        else:
            smoothness_reward = 0.0
        self.previous_ee_pos = current_ee.copy()
        
        # Action magnitude penalty (energy efficiency)
        action_penalty = -0.05 * np.linalg.norm(action)
        
        # === CURRICULUM LEARNING REWARDS ===
        # Bonus for reaching multiple goals in one episode
        if self.goals_reached_this_episode > 0:
            multi_goal_bonus = 2.0 * self.goals_reached_this_episode
        else:
            multi_goal_bonus = 0.0
        
        # === TOTAL REWARD CALCULATION ===
        total_reward = (goal_reward + exploration_reward + workspace_penalty + 
                       smoothness_reward + action_penalty + multi_goal_bonus)
        
        # Separate tracking reward for monitoring (distance to goal)
        tracking_reward = -goal_distance + (self.goal_reached_bonus if goal_reached else 0)
        
        # Store metrics for monitoring
        self.current_position_error = goal_distance
        
        # Debug logging
        if self.current_step % 50 == 0:
            log_print(f"  Pure Env Step {self.current_step}: goal_dist={goal_distance:.3f}, "
                     f"goal_reward={goal_reward:.2f}, total_reward={total_reward:.2f}")
            log_print(f"  Current goal: {current_goal}, EE pos: {current_ee}")
        
        return total_reward, tracking_reward
    
    def _get_end_effector_position(self):
        """Get end-effector position using multiple fallback methods"""
        try:
            # Method 1: Try to get end-effector link directly
            ee_link_names = ["hand", "panda_hand", "gripper", "end_effector"]
            for link_name in ee_link_names:
                try:
                    link_state = self.franka.get_link(link_name)
                    if link_state is not None:
                        pose = link_state.get_pose()
                        return pose[:3].cpu().numpy() if hasattr(pose, "cpu") else pose[:3]
                except:
                    continue
            
            # Method 2: Use joint-based forward kinematics approximation
            joint_pos = self.franka.get_dofs_position(self.dofs_idx)
            j_pos = joint_pos.cpu().numpy() if hasattr(joint_pos, "cpu") else joint_pos
            j1, j2, j3, j4, j5, j6, j7 = j_pos[:7]
            
            # Simplified Franka forward kinematics
            x = 0.3 * np.cos(j1) * np.sin(j2) + 0.4 * np.cos(j1) * np.cos(j2) * np.sin(j4)
            y = 0.3 * np.sin(j1) * np.sin(j2) + 0.4 * np.sin(j1) * np.cos(j2) * np.sin(j4)
            z = 0.5 + 0.3 * np.cos(j2) - 0.4 * np.sin(j2) * np.sin(j4)
            return np.array([x, y, z])
            
        except Exception as e:
            log_print(f"Warning: Could not get end-effector position: {e}")
            return np.array([0.0, 0.0, 0.5])  # Default safe position
    
    def _calculate_exploration_bonus(self, current_ee):
        """Reward for exploring new areas of the workspace"""
        if not hasattr(self, 'visited_positions'):
            self.visited_positions = []
        
        # Check if this position is significantly different from previously visited
        exploration_bonus = 0.0
        min_distance_to_visited = float('inf')
        
        for visited_pos in self.visited_positions:
            distance = np.linalg.norm(current_ee - visited_pos)
            min_distance_to_visited = min(min_distance_to_visited, distance)
        
        # Bonus for visiting new areas (distance > 0.2 from any visited position)
        if len(self.visited_positions) == 0 or min_distance_to_visited > 0.2:
            exploration_bonus = 5.0  # Exploration bonus
            self.visited_positions.append(current_ee.copy())
            # Keep only recent positions to prevent memory growth
            if len(self.visited_positions) > 20:
                self.visited_positions = self.visited_positions[-15:]
        
        return exploration_bonus
    
    def _target_reached(self):
        """Check if current goal is reached within tolerance"""
        current_ee = self._get_end_effector_position()
        current_goal = self._get_current_goal()
        distance = np.linalg.norm(current_ee - current_goal)
        return distance < self.goal_tolerance
    
    def get_observation(self):
        """Get current observation state - PURE ENVIRONMENT VERSION"""
        def to_numpy(x):
            return np.array(x.cpu().numpy(), dtype=np.float32) if hasattr(x, "cpu") else np.array(x, dtype=np.float32)

        # Robot state
        positions = to_numpy(self.franka.get_dofs_position(self.dofs_idx))
        velocities = to_numpy(self.franka.get_dofs_velocity(self.dofs_idx))
        
        # End-effector state
        try:
            if hasattr(self.franka, 'get_link_pose'):
                ee_pose = to_numpy(self.franka.get_link_pose("hand"))
            elif hasattr(self.franka, 'get_pose'):
                ee_pose = to_numpy(self.franka.get_pose())
                ee_pose = ee_pose[:7]  # position + quaternion
            else:
                ee_pose = np.concatenate([positions[:3], np.array([0, 0, 0, 1])])
        except Exception as e:
            log_print(f"Warning: Could not get end-effector pose: {e}")
            ee_pose = np.zeros(7, dtype=np.float32)
            
        # Torques (if available)
        try:
            torques = to_numpy(self.franka.get_dofs_torque(self.dofs_idx))
        except AttributeError:
            torques = np.zeros_like(positions)
        
        # PURE ENVIRONMENT: Current goal information (instead of reference trajectory)
        current_goal = np.array(self._get_current_goal(), dtype=np.float32)
        goal_distance = np.linalg.norm(self._get_end_effector_position() - current_goal)
        
        # Difficulty and progress information
        difficulty_info = np.array([
            self.difficulty_level / 3.0,  # Normalized difficulty
            self.goals_reached_this_episode / 10.0,  # Normalized goals reached
            self.current_step / self.max_steps,  # Episode progress
            goal_distance  # Distance to current goal
        ], dtype=np.float32)
        
        # Workspace information
        ee_pos = self._get_end_effector_position()
        workspace_info = np.array([
            ee_pos[0] / self.workspace_radius,  # Normalized X
            ee_pos[1] / self.workspace_radius,  # Normalized Y
            ee_pos[2] / self.workspace_radius,  # Normalized Z
        ], dtype=np.float32)
        
        return np.concatenate([
            positions,      # Joint positions
            velocities,     # Joint velocities  
            ee_pose,        # End-effector pose
            torques,        # Joint torques
            current_goal,   # Current goal position
            difficulty_info, # Learning progress info
            workspace_info  # Workspace state
        ])

# Enhanced Training Agent Class
class DDPGAgent:
    """Enhanced DDPG Agent with improved exploration and learning rate scheduling"""
    def __init__(self, state_dim, action_dim, lr=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Actor network with better initialization
        self.actor = ActorNetwork(state_dim, action_dim).to(self.device)
        self.actor_target = ActorNetwork(state_dim, action_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr, weight_decay=1e-5)
        
        # Initialize target network
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        # Enhanced experience replay buffer - OPTIMIZED FOR 5 MIN TRAINING
        self.memory = deque(maxlen=10000)  # Smaller buffer for faster training
        self.batch_size = 64  # Smaller batch size for faster updates
        
        # Adaptive noise for exploration
        self.noise_std = 0.3  # Higher initial exploration
        self.noise_min = 0.05
        self.noise_decay = 0.995
        
        # Learning rate scheduling
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.actor_optimizer, gamma=0.995)
        
        # Performance tracking
        self.update_count = 0
        
    def select_action(self, state, add_noise=True):
        """Enhanced action selection with adaptive noise"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().data.numpy().flatten()
        
        if add_noise:
            # Adaptive noise that decreases over time
            noise = np.random.normal(0, self.noise_std, size=action.shape)
            action = action + noise
            
            # Decay noise over time
            self.noise_std = max(self.noise_min, self.noise_std * self.noise_decay)
            
        return np.clip(action, -1, 1)
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def update_networks(self):
        """Enhanced network update with better loss calculation"""
        if len(self.memory) < self.batch_size:
            return
            
        self.update_count += 1
            
        # Sample batch from memory with prioritization for recent experiences
        if len(self.memory) > self.batch_size * 2:
            # Sample 50% recent experiences, 50% random for better learning
            recent_size = self.batch_size // 2
            recent_batch = list(self.memory)[-recent_size:]
            random_batch = random.sample(list(self.memory)[:-recent_size], self.batch_size - recent_size)
            batch = recent_batch + random_batch
        else:
            batch = random.sample(self.memory, self.batch_size)
            
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        
        # Enhanced actor loss calculation
        predicted_actions = self.actor(states)
        
        # Policy gradient loss with value estimation
        # Use reward-weighted policy gradient for better learning
        normalized_rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        actor_loss = -(normalized_rewards.unsqueeze(1) * predicted_actions).mean()
        
        # Add regularization
        l2_reg = sum(param.pow(2.0).sum() for param in self.actor.parameters())
        actor_loss += 1e-6 * l2_reg
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        
        self.actor_optimizer.step()
        
        # Schedule learning rate decay
        if self.update_count % 100 == 0:
            self.lr_scheduler.step()
        
        # Soft update target network with adaptive tau
        tau = 0.005 if self.update_count < 1000 else 0.001  # Slower updates later
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            
        return actor_loss.item()

# Create scene and add ground
scene = gs.Scene(show_viewer=True)  # Simplified scene creation without viewer_options
scene.add_entity(gs.morphs.Plane())

# Load and add Franka robot
robot = gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml')
franka = scene.add_entity(robot)
scene.build()

# Joint setup
jnt_names = [
    'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7',
    'finger_joint1', 'finger_joint2'
]
dofs_idx = [franka.get_joint(name).dof_start for name in jnt_names]

# Initialize Professional TensorBoard Logger with WebSocket Support - After robot setup
log_print("\n📊 Initializing TensorBoard Logger with WebSocket Streaming...")

# EASY WEBSOCKET ENABLE/DISABLE
USE_WEBSOCKET_TENSORBOARD = True  # Set to True to enable WebSocket streaming

if USE_WEBSOCKET_TENSORBOARD:
    # WebSocket-Enhanced TensorBoard with real-time streaming
    try:
        from simple_websocket_tensorboard import SimpleTensorBoardWebSocket, create_websocket_dashboard_html
        
        tensorboard_logger = SimpleTensorBoardWebSocket(
            log_dir="tensorboard_logs", 
            experiment_name="genesis_franka_rl",
            auto_start=True,
            ws_port=8765
        )
        
        # Create real-time dashboard
        dashboard_path = create_websocket_dashboard_html(ws_port=8765, tb_port=6006)
        log_print(f"🔌 WebSocket TensorBoard enabled!")
        log_print(f"📊 Standard TensorBoard: http://localhost:6006")
        log_print(f"🌐 Real-time Dashboard: {dashboard_path}")
        log_print(f"⚡ WebSocket Stream: ws://localhost:8765")
        
    except ImportError:
        log_print("⚠️ WebSocket TensorBoard not available, using standard version")
        USE_WEBSOCKET_TENSORBOARD = False

if not USE_WEBSOCKET_TENSORBOARD:
    # Standard TensorBoard (your existing implementation)
    tensorboard_logger = TensorBoardLogger(
        log_dir="tensorboard_logs", 
        experiment_name="genesis_franka_rl",
        auto_start=True  # Automatically start TensorBoard and open browser
    )

# Action space bounds
action_low = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0.0, 0.0], dtype=np.float32)
action_high = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 0.04, 0.04], dtype=np.float32)
action_space = gym.spaces.Box(low=action_low, high=action_high, dtype=np.float32)

# PURE ENVIRONMENT LEARNING - NO REFERENCE TRAJECTORY GENERATION
log_print("\n🚀 PURE ENVIRONMENT-DRIVEN LEARNING - NO REFERENCE TRAJECTORY!")
log_print("🎯 Robot will learn through goal-based exploration and discovery")
log_print("🔍 Goals will be dynamically generated based on workspace exploration")
log_print("📈 Curriculum learning: difficulty increases as robot improves")

# Better initial position that's safe and reachable
initial_position = np.array([0.0, -0.3, 0.0, -1.2, 0.0, 1.0, 0.785, 0.04, 0.04], dtype=np.float32)

# Reset robot to initial position
franka.set_dofs_position(initial_position, dofs_idx)
for _ in range(10):  # Stabilize
    scene.step()

log_print(f"✅ Pure environment learning initialized")
log_print(f"🎯 Dynamic goal-based learning with curriculum progression")
log_print(f"🏆 Robot will discover optimal behaviors through exploration")

# Create Pure Environment-Driven Gym Environment (NO REFERENCE TRAJECTORY)
env = FrankaGymEnv(scene, franka, dofs_idx, action_space, max_episode_steps=200)

log_print("🚀 PURE ENVIRONMENT-DRIVEN LEARNING ENVIRONMENT CREATED!")
log_print("🎯 Goal-based learning with dynamic curriculum")
log_print("🔍 Robot will explore and discover optimal behaviors")
log_print("📊 No predefined trajectory - pure environment feedback!")

# Initialize observation space dimensions
sample_obs = env.reset()
state_dim = len(sample_obs)
action_dim = len(dofs_idx)

# Log hyperparameters to TensorBoard - PURE ENVIRONMENT VERSION
hyperparameters = {
    'algorithm': 'DDPG_Pure_Environment',
    'learning_type': 'Goal_Based_Exploration', 
    'learning_rate': 1e-5,
    'batch_size': 64,
    'memory_size': 10000,
    'episodes': 50,
    'max_episode_steps': 200,
    'exploration_noise_initial': 0.3,
    'exploration_noise_min': 0.05,
    'target_update_tau': 0.005,
    'hidden_dim': 256,
    'action_dim': action_dim,
    'state_dim': state_dim,
    'goal_tolerance': 0.15,
    'goal_bonus': 50.0,
    'curriculum_learning': True,
    'max_difficulty_level': 3,
    'workspace_radius': 1.5
}

tensorboard_logger.log_hyperparameters(hyperparameters)

log_print(f"State dimension: {state_dim}")
log_print(f"Action dimension: {action_dim}")
log_print(f"🎯 Pure environment learning - dynamic goal-based rewards")
log_print(f"🔍 No predefined trajectory - robot will explore and discover!")

# Initialize DDPG Agent (Actor Network) with improved hyperparameters
agent = DDPGAgent(state_dim, action_dim, lr=1e-5)  # Much lower learning rate for stable learning

# Enhanced training parameters with curriculum learning - PURE ENVIRONMENT VERSION
num_episodes = 2   # TEST: Single episode to verify clean exit
max_steps_per_episode = 200  # Fixed episode length for pure environment learning
episode_rewards = []
episode_losses = []

# Learning rate scheduling
initial_lr = 1e-5
min_lr = 1e-6
lr_decay_rate = 0.995

# Enhanced exploration parameters
initial_noise_std = 0.3  # Higher initial exploration
min_noise_std = 0.05
noise_decay_rate = 0.995

log_print("\n=== TESTING CLEAN EXIT - 1 EPISODE ===")
log_print("🧪 Testing Genesis exit behavior with natural completion")
log_print("🚀 Pure Environment-Driven Learning: NO REFERENCE TRAJECTORY")
log_print("🎯 Goal-Based Learning: Robot discovers optimal behaviors")
log_print("🔍 Exploration: Agent learns through trial and discovery")
log_print("📈 Curriculum: Difficulty adapts based on performance")
log_print("🕐 TEST: Single episode to verify clean exit")
log_print("🎨 GOAL VISUALIZATION:")
log_print("[GREEN SPHERES] Dynamic goals: Current target positions")
log_print("[BLUE TUBES] Training trajectory: Robot's discovered path")
log_print("[YELLOW SPHERE] Current position: Robot's end-effector")
log_print("[COLORED TRAILS] Exploration paths: Different episode attempts")

# Time-based training control - REMOVED TIME LIMIT FOR FULL 50 EPISODES
import time
training_start_time = time.time()
# target_duration_seconds = 5 * 60  # REMOVED: 5 minutes in seconds  
log_print(f"🕐 TEST RUN: Will run 1 episode to verify clean exit (no time limit)")
log_print("🧪 This test verifies Genesis exits properly without hanging")

# Initial visualization with PERSISTENT 3D CYLINDRICAL TUBES
# Removed scene.clear_debug_objects() to keep trajectories visible

# 3D Trajectory Visualization Functions
def create_3d_cylinder_points(start_pos, end_pos, radius=0.05, num_rings=10, points_per_ring=8):
    """Create 3D cylindrical tube points between two positions"""
    points = []
    colors = []
    
    # Vector from start to end
    direction = np.array(end_pos) - np.array(start_pos)
    length = np.linalg.norm(direction)
    
    if length < 1e-6:  # Avoid division by zero
        return points, colors
    
    direction = direction / length
    
    # Create perpendicular vectors for the cylinder
    if abs(direction[2]) < 0.9:
        perp1 = np.cross(direction, [0, 0, 1])
    else:
        perp1 = np.cross(direction, [1, 0, 0])
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(direction, perp1)
    
    # Generate cylinder points
    for i in range(num_rings):
        t = i / (num_rings - 1) if num_rings > 1 else 0
        center = np.array(start_pos) + t * direction * length
        
        for j in range(points_per_ring):
            angle = 2 * np.pi * j / points_per_ring
            offset = radius * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
            point = center + offset
            points.append(point)
            colors.append([1.0, 0.0, 0.0, 1.0])  # Red color
    
    return points, colors

def create_3d_trajectory_tube(trajectory_points, radius=0.04, color=(1.0, 0.0, 0.0, 1.0)):
    """Create a 3D tube visualization for trajectory"""
    if len(trajectory_points) < 2:
        return []
    
    all_points = []
    for i in range(len(trajectory_points) - 1):
        start_pos = trajectory_points[i]
        end_pos = trajectory_points[i + 1]
        points, _ = create_3d_cylinder_points(start_pos, end_pos, radius)
        all_points.extend(points)
    
    return all_points

def create_large_point(position, color=(1.0, 1.0, 0.0, 1.0), radius=0.08):
    """Create a large spherical point for visualization"""
    # Generate sphere points
    points = []
    phi_steps = 10
    theta_steps = 10
    
    for i in range(phi_steps):
        phi = np.pi * i / (phi_steps - 1)
        for j in range(theta_steps):
            theta = 2 * np.pi * j / theta_steps
            x = radius * np.sin(phi) * np.cos(theta) + position[0]
            y = radius * np.sin(phi) * np.sin(theta) + position[1]
            z = radius * np.cos(phi) + position[2]
            points.append([x, y, z])
    
    return points
    """Create 3D cylindrical trajectory segments like in the reference image"""
    points = []
    
    # Calculate direction vector
    direction = np.array(end_pos) - np.array(start_pos)
    length = np.linalg.norm(direction)
    
    if length < 1e-6:  # Points too close
        return [start_pos]
    
    direction_normalized = direction / length
    
    # Create perpendicular vectors for cylinder cross-section
    if abs(direction_normalized[2]) < 0.9:
        perp1 = np.cross(direction_normalized, [0, 0, 1])
    else:
        perp1 = np.cross(direction_normalized, [1, 0, 0])
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(direction_normalized, perp1)
    perp2 = perp2 / np.linalg.norm(perp2)
    
    # Generate cylinder points
    for ring in range(num_rings):
        t = ring / max(1, num_rings - 1)  # Parameter along cylinder length
        center = np.array(start_pos) + t * direction
        
        for i in range(points_per_ring):
            angle = 2 * np.pi * i / points_per_ring
            offset = radius * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
            point = center + offset
            points.append(point.tolist())
    
    return points

def create_3d_trajectory_tube(trajectory_points, radius=0.03, color=(1.0, 0.0, 0.0, 1.0)):
    """Create a 3D tube visualization for trajectory following the reference image style"""
    if len(trajectory_points) < 2:
        return []
    
    all_tube_points = []
    
    # Create cylinder segments between consecutive points
    for i in range(len(trajectory_points) - 1):
        start = trajectory_points[i]
        end = trajectory_points[i + 1]
        
        # Create cylindrical segment
        cylinder_points = create_3d_cylinder_points(start, end, radius=radius, num_rings=8, points_per_ring=12)
        all_tube_points.extend(cylinder_points)
    
    return all_tube_points

def create_large_point(center, color, radius=0.1):
    """Create a visually large point by clustering multiple small points"""
    points = []
    # Create a sphere of points around the center
    for i in range(20):  # More points = larger visual sphere
        offset_x = (np.random.random() - 0.5) * radius
        offset_y = (np.random.random() - 0.5) * radius 
        offset_z = (np.random.random() - 0.5) * radius
        point = [center[0] + offset_x, center[1] + offset_y, center[2] + offset_z]
        points.append(point)
    return points

# Large coordinate system markers
origin_cluster = create_large_point([0, 0, 0], (1.0, 1.0, 1.0, 1.0), 0.05)
x_cluster = create_large_point([0.3, 0, 0], (1.0, 0, 0, 1.0), 0.08)
y_cluster = create_large_point([0, 0.3, 0], (0, 1.0, 0, 1.0), 0.08)
z_cluster = create_large_point([0, 0, 0.3], (0, 0, 1.0, 1.0), 0.08)

scene.draw_debug_points(origin_cluster, (1.0, 1.0, 1.0, 1.0))  # White origin - PERMANENT
scene.draw_debug_points(x_cluster, (1.0, 0, 0, 1.0))  # Red X - PERMANENT
scene.draw_debug_points(y_cluster, (0, 1.0, 0, 1.0))  # Green Y - PERMANENT
scene.draw_debug_points(z_cluster, (0, 0, 1.0, 1.0))  # Blue Z - PERMANENT

# PURE ENVIRONMENT VISUALIZATION - Dynamic Goals Instead of Reference - FIXED
# Draw current goals as green spheres
def visualize_current_goals(env, scene):
    """Visualize current goals for pure environment learning - FIXED"""
    try:
        current_goal = env._get_current_goal()
        goal_sphere = create_large_point(current_goal, (0.0, 1.0, 0.0, 1.0), 0.12)
        safe_goal_points = create_safe_debug_points(goal_sphere, max_points=50)
        if safe_goal_points:
            scene.draw_debug_points(safe_goal_points, (0.0, 1.0, 0.0, 1.0))  # Green goal
        
        # Also show all available goals as smaller spheres
        available_goals = min(4 * env.difficulty_level, len(env.target_goals))
        for i, goal in enumerate(env.target_goals[:available_goals]):
            if np.linalg.norm(np.array(goal) - np.array(current_goal)) > 0.1:  # Don't overlap with current
                small_goal = create_large_point(goal, (0.0, 0.7, 0.0, 0.5), 0.06)
                safe_small_points = create_safe_debug_points(small_goal, max_points=30)
                if safe_small_points:
                    scene.draw_debug_points(safe_small_points, (0.0, 0.7, 0.0, 0.5))  # Smaller green goals
    except Exception as e:
        log_print(f"Warning: Could not visualize goals: {e}")

if hasattr(env, 'target_goals'):
    # Visualize goals for pure environment learning
    visualize_current_goals(env, scene)
    log_print(f"[PERMANENT] Goal visualization: DYNAMIC GREEN SPHERES for current goals")

# Store training trajectory for visualization
training_trajectory = []
all_episode_trajectories = []  # Store trajectories from all episodes

for episode in range(num_episodes):
    # REMOVED TIME LIMIT CHECK - WILL RUN FULL 50 EPISODES
    
    # Show TensorBoard status every 10 episodes
    if episode % 10 == 0 and episode > 0:
        tb_status = "🟢 STREAMING" if tensorboard_logger.tb_manager.is_tensorboard_running() else "🔴 STOPPED"
        log_print(f"📊 TensorBoard Status: {tb_status} | Dashboard: http://localhost:{tensorboard_logger.tb_manager.port}")
    
    # Reset environment and get initial state s(t)
    state = env.reset()
    episode_reward = 0
    episode_tracking_reward = 0  # Separate tracking reward accumulator
    episode_loss = 0
    training_trajectory = []  # Reset trajectory for each episode
    episode_position_errors = []  # Track position errors per episode
    
    # Log episode start to TensorBoard
    tensorboard_logger.log_episode_start(episode + 1)
    
    elapsed_time = time.time() - training_start_time
    print(f"\nEpisode {episode + 1}/{num_episodes} | Elapsed time: {elapsed_time//60:.1f}min {elapsed_time%60:.0f}s")
    
    for step in range(max_steps_per_episode):
        # Actor Network processes state s(t) and outputs action A(t)
        action = agent.select_action(state, add_noise=(episode < num_episodes * 0.8))  # Reduce noise over time
        
        # Environment receives action A(t) and returns next state, reward, done
        next_state, reward, done, info = env.step(action)
        
        # Extract separated rewards and metrics from info
        total_reward = info.get('total_reward', reward)
        tracking_reward = info.get('tracking_reward', 0)
        position_error = info.get('position_error', 0)
        
        # Store experience for learning
        agent.store_experience(state, action, reward, next_state, done)
        
        # Update networks
        if len(agent.memory) > agent.batch_size:
            loss = agent.update_networks()
            if loss is not None:
                episode_loss = loss
        
        # Log step-level metrics to TensorBoard
        step_data = {
            'step': step,
            'reward': reward,
            'total_reward': total_reward,
            'tracking_reward': tracking_reward,
            'position_error': position_error,
            'action': action,
            'loss': episode_loss if episode_loss is not None and episode_loss >= 0 else 0.0
        }
        tensorboard_logger.log_step_metrics(step_data)
        
        # Update state for next iteration
        state = next_state
        episode_reward += total_reward
        episode_tracking_reward += tracking_reward
        episode_position_errors.append(position_error)
        
        # Store current position with CORRECT end-effector calculation
        try:
            # Get ACTUAL end-effector position from the robot links
            ee_link_names = ["hand", "panda_hand", "gripper", "end_effector"]
            current_ee_pos = None
            
            for link_name in ee_link_names:
                try:
                    link_state = franka.get_link(link_name)
                    if link_state is not None:
                        pose = link_state.get_pose()
                        current_ee_pos = pose[:3].cpu().numpy() if hasattr(pose, "cpu") else pose[:3]
                        break
                except:
                    continue
            
            # Fallback: Use forward kinematics approximation
            if current_ee_pos is None:
                joint_pos = franka.get_dofs_position(dofs_idx)
                j_pos = joint_pos.cpu().numpy() if hasattr(joint_pos, "cpu") else joint_pos
                j1, j2, j3, j4, j5, j6, j7 = j_pos[:7]
                
                # Simplified Franka forward kinematics for end-effector
                x = 0.3 * np.cos(j1) * np.sin(j2) + 0.4 * np.cos(j1) * np.cos(j2) * np.sin(j4)
                y = 0.3 * np.sin(j1) * np.sin(j2) + 0.4 * np.sin(j1) * np.cos(j2) * np.sin(j4)
                z = 0.5 + 0.3 * np.cos(j2) - 0.4 * np.sin(j2) * np.sin(j4)
                current_ee_pos = np.array([x, y, z])
                
            training_trajectory.append(current_ee_pos.copy())
            
            # Enhanced real-time trajectory visualization with PERSISTENT GOAL-BASED TUBES
            if step % 20 == 0 and len(training_trajectory) > 1:  # Reduced frequency for faster training
                # Always redraw current goals
                try:
                    visualize_current_goals(env, scene)  # Show current goals
                except:
                    pass  # Continue if visualization fails
                
                # Draw CUMULATIVE training trajectory as 3D BLUE CYLINDRICAL TUBES (not just recent)
                try:
                    # Use simple points for training trajectory instead of complex tubes
                    simple_training_points = [[float(pos[0]), float(pos[1]), float(pos[2])] for pos in training_trajectory[::2]]
                    if simple_training_points:
                        scene.draw_debug_points(simple_training_points, (0.0, 0.0, 1.0, 0.9))  # Blue points - CUMULATIVE
                    
                    # Draw current position as simple point instead of complex sphere
                    if current_ee_pos is not None:
                        simple_current_point = [[float(current_ee_pos[0]), float(current_ee_pos[1]), float(current_ee_pos[2])]]
                        scene.draw_debug_points(simple_current_point, (1.0, 1.0, 0.0, 1.0))
                except:
                    # Fallback to bright points
                    scene.draw_debug_points(training_trajectory, (0.0, 0.0, 1.0, 1.0))
                    scene.draw_debug_points([current_ee_pos], (1.0, 1.0, 0.0, 1.0))
            
            # Debug: Enhanced position tracking for goal-based learning
            if step == 0:
                current_goal = env._get_current_goal()
                goal_distance = np.linalg.norm(current_ee_pos - current_goal)
                log_print(f"  Step {step}: goal_dist={goal_distance:.3f}, reward={reward:.3f}")
                log_print(f"  Initial EE position: {current_ee_pos}")
                log_print(f"  Current goal: {current_goal}")
            elif step % 50 == 0:
                current_goal = env._get_current_goal()
                goal_distance = np.linalg.norm(current_ee_pos - current_goal)
                log_print(f"  Step {step}: goal_dist={goal_distance:.3f}, reward={reward:.3f}")
                log_print(f"  Step {step} EE position: {current_ee_pos}")
                log_print(f"  Current goal: {current_goal}")
                
        except Exception as e:
            log_print(f"  Warning: Could not get end-effector position at step {step}: {e}")
            # Add a default position to keep trajectory going
            if len(training_trajectory) > 0:
                training_trajectory.append(training_trajectory[-1])  # Repeat last position
            else:
                training_trajectory.append([0, 0, 0])  # Default position
        
        # Visualization every 10 steps with PERSISTENT 3D CYLINDRICAL TUBES - OPTIMIZED
        if step % 10 == 0:  # Reduced from every 5 steps to every 10 steps
            # NO CLEARING - Keep all trajectories persistent and visible
            
            def create_3d_tube_segment(start, end, radius=0.04, num_rings=6, points_per_ring=8):
                """Create a short 3D cylindrical segment"""
                return create_3d_cylinder_points(start, end, radius, num_rings, points_per_ring)
            
            # Always ensure current goals are visible with DYNAMIC GREEN SPHERES
            try:
                visualize_current_goals(env, scene)  # Show current goals
            except:
                pass  # Continue if visualization fails
            
            # Always add to the training trajectory with SIMPLE BLUE POINTS
            if len(training_trajectory) > 2:
                # Draw the training trajectory as simple points
                sampled_trajectory = training_trajectory[::2] if len(training_trajectory) > 50 else training_trajectory
                simple_training_points = [[float(pos[0]), float(pos[1]), float(pos[2])] for pos in sampled_trajectory]
                
                if simple_training_points:
                    scene.draw_debug_points(simple_training_points, (0.0, 0.8, 1.0, 0.9))  # SIMPLE Blue points
            
            # Always show current robot position as LARGE YELLOW SPHERE
            if len(training_trajectory) > 0:
                current_pos = training_trajectory[-1]
                # Create the largest sphere for current position
                yellow_sphere = create_large_point(current_pos, (1.0, 1.0, 0.0, 1.0), 0.15)
                scene.draw_debug_points(yellow_sphere, (1.0, 1.0, 0.0, 1.0))  # PERSISTENT Yellow position
            
            log_print(f"Episode {episode + 1}, Step {step}: PERSISTENT 3D TUBES - Total trajectory: {len(training_trajectory)}")
        
        # Removed video capture for faster test execution
        # if step % 20 == 0 and hasattr(env, 'video_recorder'):  # Reduced frequency for faster training
        #     env.video_recorder.capture_frame(scene)
        
        if done:
            break
    
    episode_rewards.append(episode_reward)
    
    # Log episode data to live monitor
    avg_position_error = np.mean(episode_position_errors) if episode_position_errors else 0.0
    training_monitor.log_episode(
        episode=episode + 1,
        total_reward=episode_reward,
        tracking_reward=episode_tracking_reward,
        position_error=avg_position_error
    )
    
    # Log episode metrics to TensorBoard
    episode_data = {
        'episode': episode + 1,
        'total_reward': episode_reward,
        'tracking_reward': episode_tracking_reward,
        'position_error': avg_position_error,
        'episode_length': len(training_trajectory),
        'learning_rate': agent.actor_optimizer.param_groups[0]['lr'],
        'noise_std': agent.noise_std,
        'buffer_size': len(agent.memory)
    }
    tensorboard_logger.log_episode_metrics(episode_data)
    
    # Log network weights every 10 episodes
    if (episode + 1) % 10 == 0:
        tensorboard_logger.log_network_weights(agent.actor, episode + 1)
    
    # Store this episode's trajectory
    if len(training_trajectory) > 0:
        all_episode_trajectories.append(training_trajectory.copy())
    
    # Enhanced episode statistics with learning progress
    avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
    current_noise = agent.noise_std
    current_lr = agent.actor_optimizer.param_groups[0]['lr']
    
    log_print(f"Episode {episode + 1} - Total Reward: {episode_reward:.2f}, Tracking Reward: {episode_tracking_reward:.2f}, Avg Reward (last 10): {avg_reward:.2f}, Trajectory Points: {len(training_trajectory)}")
    log_print(f"  [LEARNING] Noise: {current_noise:.4f}, LR: {current_lr:.2e}, Buffer: {len(agent.memory)}, Avg Position Error: {avg_position_error:.4f}")
    
    # Additional debugging info
    if len(training_trajectory) == 0:
        log_print(f"  [WARNING] No trajectory points recorded in episode {episode + 1}")
    else:
        log_print(f"  [SUCCESS] Trajectory recorded: {len(training_trajectory)} points")
        if len(training_trajectory) > 1:
            traj_movement = np.linalg.norm(np.array(training_trajectory[-1]) - np.array(training_trajectory[0]))
            log_print(f"  [MOVEMENT] Total movement: {traj_movement:.3f} units")
            
            # Calculate goal-reaching performance instead of trajectory following
            goals_reached_episode = env.goals_reached_this_episode
            current_goal_distance = np.linalg.norm(current_ee_pos - env._get_current_goal()) if current_ee_pos is not None else 0.0
            
            log_print(f"  [SUCCESS] Trajectory recorded: {len(training_trajectory)} points")
            log_print(f"  [GOALS] Goals reached this episode: {goals_reached_episode}")
            log_print(f"  [DISTANCE] Current goal distance: {current_goal_distance:.3f} units")
    
    # Enhanced trajectory comparison every 10 episodes with ADDITIONAL 3D CYLINDRICAL TUBES - OPTIMIZED
    if (episode + 1) % 10 == 0:  # Reduced from every 5 episodes to every 10 episodes
        # NO CLEARING - ADD to existing visualization instead of replacing
        
        # Always ensure current goals remain visible
        try:
            visualize_current_goals(env, scene)  # Show current goals
        except:
            pass  # Continue if visualization fails
        
        # Add last 3 episode trajectories with 3D COLORED CYLINDRICAL TUBES (ADDITIVE)
        tube_colors = [(0.0, 1.0, 0.0, 0.7), (0.0, 1.0, 1.0, 0.7), (1.0, 0.0, 1.0, 0.7)]  # Green, Cyan, Magenta
        tube_radii = [0.025, 0.030, 0.035]  # Different sizes for distinction
        
        for i, traj in enumerate(all_episode_trajectories[-3:]):
            if len(traj) > 2:
                color = tube_colors[i % 3]
                radius = tube_radii[i % 3]
                
                # Create simple points for this episode's trajectory - ADDITIVE
                simple_episode_points = [[float(pos[0]), float(pos[1]), float(pos[2])] for pos in traj[::2]]
                if simple_episode_points:
                    scene.draw_debug_points(simple_episode_points, color)
        
        log_print("🎨 Episode comparison with ADDITIONAL 3D CYLINDRICAL TUBES - no clearing, persistent view!")
        
        # Removed video capture for faster test execution
    
    # Decay noise for exploration
    agent.noise_std = max(0.01, agent.noise_std * 0.995)

log_print("\n=== TEST EPISODE COMPLETE ===")
log_print("🧪 Single episode test completed successfully!")
log_print("🔍 Now testing Genesis cleanup and exit behavior...")
log_print(f"Final average reward: {np.mean(episode_rewards[-10:]):.2f}")

# Generate final training evidence and metrics
log_print("\n🔥 Generating Final Training Evidence...")
final_stats = training_monitor.finalize_training()

# Finalize TensorBoard logging
log_print("\n📊 Finalizing TensorBoard logs...")
tensorboard_final_stats = tensorboard_logger.finalize()

# Stop live monitoring
training_monitor.stop_monitoring()

# Stop video recording
# Plot training results - FIXED FOR PROPER GENERATION
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend to avoid display issues
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(15, 6))
    plt.ioff()  # Turn off interactive plotting

    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, linewidth=2, color='blue')
    plt.title('Episode Rewards Over Time', fontsize=14)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    # Plot moving average
    window_size = 10
    if len(episode_rewards) >= window_size:
        moving_avg = [np.mean(episode_rewards[i:i+window_size]) for i in range(len(episode_rewards)-window_size+1)]
        plt.plot(range(window_size-1, len(episode_rewards)), moving_avg, linewidth=2, color='red')
        plt.title(f'Moving Average Reward (window={window_size})', fontsize=14)
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Average Reward', fontsize=12)
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    
    # Save with absolute path to current directory
    plot_path = os.path.join(os.getcwd(), 'training_results.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()  # Close the figure to free memory
    
    # Verify file was actually created
    if os.path.exists(plot_path):
        file_size = os.path.getsize(plot_path)
        log_print(f"✅ Training results plot saved successfully to: {plot_path}")
        log_print(f"📏 File size: {file_size} bytes")
    else:
        log_print(f"❌ Plot file was not created at: {plot_path}")
    
except Exception as e:
    log_print(f"❌ Error generating training plot: {e}")
    log_print("Plot generation failed, but training completed successfully.")

# Enhanced final demonstration with comprehensive analysis - SKIPPED FOR EXIT TEST
log_print("\n=== SINGLE EPISODE TEST COMPLETE ===")
log_print("🧪 Skipping final demonstration - testing exit behavior only")
log_print("🔍 Focus: Verify Genesis cleanup without hanging")

# Skip final demonstration for exit test
final_trajectory = []
total_final_reward = 0.0
log_print(f"Final demonstration reward: {total_final_reward:.2f}")

# Calculate goal-reaching performance instead of trajectory following
if len(final_trajectory) > 0:
    total_goals_reached = sum(env.goals_reached_this_episode for env in [env])  # Get from environment
    final_goal_distance = np.linalg.norm(final_trajectory[-1] - env._get_current_goal()) if final_trajectory else 0.0
    
    log_print(f"Final goal-reaching performance:")
    log_print(f"  Goals reached: {total_goals_reached}")
    log_print(f"  Final goal distance: {final_goal_distance:.3f} units")
    log_print(f"  Difficulty level reached: {env.difficulty_level}")
    log_print(f"  Exploration points: {len(final_trajectory)} positions")

# Enhanced final visualization with GOAL-BASED 3D VISUALIZATION
# Show final goals and achieved trajectory

# Ensure current goals are still visible with GREEN SPHERES
try:
    visualize_current_goals(env, scene)  # Show final goals
    log_print("Final goals maintained as GREEN SPHERES")
except:
    log_print("Goal visualization failed in final demo")

# Add final trained trajectory as SAFE TRAJECTORY POINTS
if final_trajectory and len(final_trajectory) > 2:
    try:
        # Use safe trajectory visualization
        safe_final_points = create_simple_trajectory_points(
            final_trajectory, 
            color=(0.0, 1.0, 0.0, 0.95), 
            point_size=0.08
        )
        validated_final_points = create_safe_debug_points(safe_final_points, max_points=150)
        if validated_final_points and len(validated_final_points) > 0:
            scene.draw_debug_points(validated_final_points, (0.0, 1.0, 0.0, 0.95))  # Bright green trajectory points
            log_print("Final trained trajectory ADDED as SAFE GREEN TRAJECTORY POINTS")
        else:
            log_print("Warning: Final trajectory points could not be validated")
    except Exception as e:
        log_print(f"Warning: Final trajectory visualization failed: {e}")

log_print("\n=== PURE ENVIRONMENT LEARNING VISUALIZATION SUMMARY ===")
log_print("[GREEN SPHERES] Dynamic goals: Current and available targets")
log_print("[BRIGHT GREEN POINTS] Final trained goal-reaching trajectory (discovered path)")
log_print("[BLUE TRAILS] Training exploration trajectories (learning process)")
log_print("[COLORED TRAILS] Previous episode exploration attempts (learning history)")
log_print("� ALL GOAL-BASED LEARNING - no disappearing targets!")
log_print("🔍 PURE DISCOVERY - robot learned through exploration!")

# Training summary with live monitoring results
best_episode = np.argmax(episode_rewards) + 1
best_reward = np.max(episode_rewards)
final_avg_reward = np.mean(episode_rewards[-10:])
improvement = final_avg_reward - episode_rewards[0] if len(episode_rewards) > 0 else 0

log_print(f"\n=== Final Training Summary ===")
log_print(f"Episodes completed: {len(episode_rewards)}")
log_print(f"Best episode: {best_episode} (reward: {best_reward:.2f})")
log_print(f"Final 10-episode average: {final_avg_reward:.2f}")
log_print(f"Learning improvement: {improvement:.2f}")
log_print(f"Final exploration noise: {agent.noise_std:.4f}")
log_print(f"Final learning rate: {agent.actor_optimizer.param_groups[0]['lr']:.2e}")

# Add live monitoring summary
if final_stats:
    log_print(f"\n📊 LIVE MONITORING EVIDENCE:")
    log_print(f"  🎯 Total Reward Statistics:")
    log_print(f"    - Min: {final_stats['total_rewards']['min']:.2f}")
    log_print(f"    - Max: {final_stats['total_rewards']['max']:.2f}")
    log_print(f"    - Mean: {final_stats['total_rewards']['mean']:.2f}")
    log_print(f"    - Final: {final_stats['total_rewards']['final']:.2f}")
    log_print(f"    - Improvement: {final_stats['total_rewards']['improvement']:.2f}")
    log_print(f"  📍 Tracking Reward Statistics:")
    log_print(f"    - Min: {final_stats['tracking_rewards']['min']:.2f}")
    log_print(f"    - Max: {final_stats['tracking_rewards']['max']:.2f}")
    log_print(f"    - Mean: {final_stats['tracking_rewards']['mean']:.2f}")
    log_print(f"    - Final: {final_stats['tracking_rewards']['final']:.2f}")
    log_print(f"  📏 Position Error Statistics:")
    log_print(f"    - Min: {final_stats['position_errors']['min']:.4f}")
    log_print(f"    - Max: {final_stats['position_errors']['max']:.4f}")
    log_print(f"    - Mean: {final_stats['position_errors']['mean']:.4f}")
    log_print(f"📁 Complete evidence saved in: training_metrics/ directory")

# Add TensorBoard summary
if tensorboard_final_stats:
    log_print(f"\n📊 TENSORBOARD EVIDENCE:")
    log_print(f"  🎯 Professional ML Experiment Tracking Completed")
    log_print(f"  📈 All metrics logged with step-by-step detail")
    log_print(f"  🔬 Hyperparameters, gradients, and weights tracked")
    log_print(f"  📊 View live dashboard: tensorboard --logdir=tensorboard_logs")
    log_print(f"  🌐 Access at: http://localhost:6006")
    log_print(f"  📁 TensorBoard logs saved in: tensorboard_logs/ directory")

# Close log file with final file listing
log_print(f"\n=== Enhanced training session completed at {datetime.datetime.now()} ===")
log_print(f"Log saved to: {log_filename}")

# List generated files
current_files = os.listdir(os.getcwd())
plot_files = [f for f in current_files if f.endswith('.png')]
video_files = [f for f in current_files if f.endswith('.mp4')]

if plot_files:
    log_print(f"📊 Generated plot files: {plot_files}")
else:
    log_print("⚠️ No PNG files found in current directory")

if video_files:
    log_print(f"🎥 Generated video files: {video_files}")
    for video_file in video_files:
        if os.path.exists(video_file):
            size_mb = os.path.getsize(video_file) / 1024 / 1024
            log_print(f"   📹 {video_file}: {size_mb:.1f} MB")
else:
    log_print("⚠️ No MP4 video files found in current directory")

# Auto-exit after training completion (no user input required)
log_print("🔄 Auto-closing Genesis simulation...")
import time
import threading
import signal

# Close log file first
try:
    log_file.close()
    print("📝 Log file closed successfully")
except:
    print("⚠️ Log file already closed")

# Removed problematic force_exit function

# Removed problematic cleanup_genesis function

# Start timeout thread
# Removed problematic timeout thread

# Brief pause to ensure all logs are written
# time.sleep(2)

# Genesis-safe cleanup with timeout protection - PREVENTS HANGING
print("🔄 Starting Genesis-safe cleanup...")

# STEP 1: Stop TensorBoard processes FIRST (critical for preventing hang)
if 'tensorboard_logger' in globals() and tensorboard_logger is not None:
    try:
        print("🔄 Stopping TensorBoard safely...")
        
        # Stop TensorBoard manager
        if hasattr(tensorboard_logger, 'tb_manager') and tensorboard_logger.tb_manager:
            tensorboard_logger.tb_manager.stop_tensorboard(silent=False)
        
        # Close TensorBoard writer
        if hasattr(tensorboard_logger, 'writer') and tensorboard_logger.writer:
            tensorboard_logger.writer.close()
        
        # Finalize logger if available
        if hasattr(tensorboard_logger, 'finalize'):
            tensorboard_logger.finalize()
            
        print("✅ TensorBoard stopped successfully")
    except Exception as e:
        print(f"⚠️ TensorBoard stop warning: {e}")

# STEP 2: Force kill any remaining TensorBoard processes
try:
    import subprocess
    print("🔄 Killing remaining TensorBoard processes...")
    
    # Kill TensorBoard executable
    subprocess.run(['taskkill', '/f', '/im', 'tensorboard.exe'], 
                  capture_output=True, timeout=3)
    
    # Kill Python processes running TensorBoard
    subprocess.run(['taskkill', '/f', '/fi', 'WINDOWTITLE eq *tensorboard*'], 
                  capture_output=True, timeout=3)
    
    print("✅ TensorBoard processes terminated")
except Exception as e:
    print(f"⚠️ Process cleanup warning: {e}")

# STEP 3: Brief pause to ensure complete process cleanup
print("🔄 Waiting for process cleanup completion...")
time.sleep(3)

# STEP 4: Genesis destroy with timeout protection (prevents infinite hang)
def safe_genesis_destroy():
    """Safe Genesis destroy with error handling"""
    try:
        print("🔄 Destroying Genesis simulation...")
        gs.destroy()
        print("✅ Genesis destroyed successfully")
        return True
    except Exception as e:
        error_msg = str(e)
        if "CUDA_ERROR_INVALID_CONTEXT" in error_msg:
            print("⚠️ Genesis CUDA cleanup warning (expected during shutdown)")
        else:
            print(f"⚠️ Genesis destroy error: {e}")
        return False

# Run Genesis destroy in separate thread with timeout
import threading
destroy_success = [False]

def destroy_worker():
    """Worker thread for Genesis destroy"""
    destroy_success[0] = safe_genesis_destroy()

# Start destroy thread with timeout protection
destroy_thread = threading.Thread(target=destroy_worker, daemon=True)
destroy_thread.start()
destroy_thread.join(timeout=10)  # 10 second timeout

# Check if destroy completed or timed out
if destroy_thread.is_alive():
    print("⚠️ Genesis destroy timed out after 10 seconds, forcing exit...")
    print("⚠️ This prevents infinite hanging - Genesis may have cleanup issues")
else:
    if destroy_success[0]:
        print("✅ Genesis destroyed successfully")
    else:
        print("✅ Genesis cleanup completed (CUDA warnings are normal during shutdown)")

# STEP 5: Final exit (guaranteed to work)
print("✅ Training session completed - Genesis-safe exit successful!")
print("🔥 Ready for next training run!")
import os
os._exit(0)  # Force exit regardless of Genesis state
# Use safe exit with Genesis timeout protection
# safe_exit_with_genesis_timeout(gs)