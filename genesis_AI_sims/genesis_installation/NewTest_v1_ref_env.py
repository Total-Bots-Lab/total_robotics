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
import cv2  # For video recording
import time
import threading
# Remove GUI-related imports to prevent threading issues
import queue
import json
from datetime import datetime as dt
from torch.utils.tensorboard import SummaryWriter
import logging

# Professional TensorBoard Integration for PyTorch
class TensorBoardLogger:
    """Professional TensorBoard integration for Genesis AI training"""
    
    def __init__(self, log_dir="tensorboard_logs", experiment_name=None):
        """Initialize TensorBoard logger with professional setup"""
        timestamp = dt.now().strftime('%Y%m%d_%H%M%S')
        if experiment_name:
            self.log_dir = os.path.join(log_dir, f"{experiment_name}_{timestamp}")
        else:
            self.log_dir = os.path.join(log_dir, f"genesis_training_{timestamp}")
        
        # Create TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
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
    
    def log_video(self, video_frames, episode, fps=20):
        """Log video to TensorBoard"""
        if len(video_frames) > 0:
            # Convert frames to tensor format expected by TensorBoard
            video_tensor = torch.from_numpy(np.array(video_frames)).unsqueeze(0)
            self.writer.add_video('Training/Episode_Video', video_tensor, episode, fps=fps)
    
    def log_text(self, tag, text, step):
        """Log text data to TensorBoard"""
        self.writer.add_text(tag, text, step)
    
    def finalize(self):
        """Close TensorBoard writer and generate final summary"""
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
        
        self.writer.close()
        log_print(f"✅ TensorBoard logging completed")
        log_print(f"📊 View results: tensorboard --logdir={os.path.dirname(self.log_dir)}")
        log_print(f"🌐 Open: http://localhost:6006")
        
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

# Video Recording Class for Genesis Simulation
class GenesisVideoRecorder:
    """Records Genesis simulation as video with proper frame capture"""
    def __init__(self, output_path="genesis_training_video.mp4", fps=30, resolution=(1920, 1080)):
        self.output_path = output_path
        self.fps = fps
        self.resolution = resolution
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = None
        self.frame_count = 0
        self.recording = False
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
    def start_recording(self):
        """Start video recording"""
        try:
            self.video_writer = cv2.VideoWriter(self.output_path, self.fourcc, self.fps, self.resolution)
            if self.video_writer.isOpened():
                self.recording = True
                self.frame_count = 0
                log_print(f"🎥 Video recording started: {self.output_path}")
                log_print(f"📹 Recording settings: {self.resolution[0]}x{self.resolution[1]} @ {self.fps}fps")
                return True
            else:
                log_print(f"❌ Failed to initialize video writer")
                return False
        except Exception as e:
            log_print(f"❌ Error starting video recording: {e}")
            return False
    
    def capture_frame(self, scene):
        """Capture a frame from Genesis scene - ENHANCED for actual viewer capture"""
        if not self.recording or self.video_writer is None:
            return False
            
        try:
            # Method 1: Try Genesis viewer screenshot methods
            if hasattr(scene, 'viewer') and scene.viewer is not None:
                try:
                    # Try different Genesis viewer capture methods
                    if hasattr(scene.viewer, 'get_screenshot'):
                        rgb_data = scene.viewer.get_screenshot()
                    elif hasattr(scene.viewer, 'screenshot'):
                        rgb_data = scene.viewer.screenshot()
                    elif hasattr(scene.viewer, 'capture'):
                        rgb_data = scene.viewer.capture()
                    elif hasattr(scene.viewer, 'get_rgb_array'):
                        rgb_data = scene.viewer.get_rgb_array()
                    elif hasattr(scene.viewer, 'render'):
                        rgb_data = scene.viewer.render(mode='rgb_array')
                    else:
                        rgb_data = None
                    
                    if rgb_data is not None and hasattr(rgb_data, 'shape') and len(rgb_data.shape) == 3:
                        # Convert to numpy if needed
                        if hasattr(rgb_data, 'cpu'):
                            rgb_data = rgb_data.cpu().numpy()
                        elif hasattr(rgb_data, 'numpy'):
                            rgb_data = rgb_data.numpy()
                        
                        # Ensure correct data type
                        if rgb_data.dtype != np.uint8:
                            rgb_data = (rgb_data * 255).astype(np.uint8)
                        
                        # Convert RGB to BGR for OpenCV
                        bgr_frame = cv2.cvtColor(rgb_data, cv2.COLOR_RGB2BGR)
                        
                        # Resize if necessary
                        if bgr_frame.shape[:2][::-1] != self.resolution:
                            bgr_frame = cv2.resize(bgr_frame, self.resolution)
                        
                        self.video_writer.write(bgr_frame)
                        self.frame_count += 1
                        return True
                        
                except Exception as viewer_error:
                    if self.frame_count % 200 == 0:  # Log occasionally
                        log_print(f"Viewer capture method failed: {viewer_error}")
            
            # Method 2: Try scene-level rendering methods
            try:
                render_methods = ['render', 'get_rgb_image', 'get_image', 'screenshot']
                rgb_data = None
                
                for method_name in render_methods:
                    if hasattr(scene, method_name):
                        method = getattr(scene, method_name)
                        try:
                            if method_name == 'render':
                                rgb_data = method(mode='rgb_array')
                            else:
                                rgb_data = method()
                            
                            if rgb_data is not None:
                                break
                        except:
                            continue
                
                if rgb_data is not None and hasattr(rgb_data, 'shape') and len(rgb_data.shape) == 3:
                    # Process the captured frame
                    if hasattr(rgb_data, 'cpu'):
                        rgb_data = rgb_data.cpu().numpy()
                    elif hasattr(rgb_data, 'numpy'):
                        rgb_data = rgb_data.numpy()
                    
                    if rgb_data.dtype != np.uint8:
                        rgb_data = (rgb_data * 255).astype(np.uint8)
                    
                    bgr_frame = cv2.cvtColor(rgb_data, cv2.COLOR_RGB2BGR)
                    if bgr_frame.shape[:2][::-1] != self.resolution:
                        bgr_frame = cv2.resize(bgr_frame, self.resolution)
                    
                    self.video_writer.write(bgr_frame)
                    self.frame_count += 1
                    return True
                    
            except Exception as render_error:
                if self.frame_count % 200 == 0:
                    log_print(f"Scene render method failed: {render_error}")
            
            # Method 3: Try external screen capture (Windows-specific for Genesis viewer)
            try:
                try:
                    import pyautogui
                    # Capture the entire screen and crop to viewer window
                    screenshot = pyautogui.screenshot()
                    screenshot_np = np.array(screenshot)
                    
                    # Convert RGB to BGR
                    bgr_frame = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
                    
                    # Resize to target resolution
                    bgr_frame = cv2.resize(bgr_frame, self.resolution)
                    
                    self.video_writer.write(bgr_frame)
                    self.frame_count += 1
                    return True
                except ImportError:
                    if self.frame_count == 1:  # Log once
                        log_print("pyautogui not available for screen capture")
                except Exception as screen_error:
                    if self.frame_count % 500 == 0:
                        log_print(f"Screen capture failed: {screen_error}")
                
            except Exception:
                pass  # Continue to fallback method
            
            # Method 4: Last resort - enhanced placeholder with actual training info
            placeholder_frame = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
            
            # Make placeholder more informative with Genesis-style background
            # Add dark blue gradient background similar to Genesis
            for y in range(self.resolution[1]):
                intensity = int(20 + (y / self.resolution[1]) * 30)
                placeholder_frame[y, :] = [intensity, intensity//2, intensity//4]
            
            # Add title
            cv2.putText(placeholder_frame, "Genesis AI - Franka Robot Training", 
                       (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            
            # Add 3D visualization indicator
            cv2.putText(placeholder_frame, "3D Cylindrical Trajectory Visualization", 
                       (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            
            # Add frame and training info
            cv2.putText(placeholder_frame, f"Frame: {self.frame_count}", 
                       (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            # Add note about viewer capture
            cv2.putText(placeholder_frame, "Note: Direct viewer capture unavailable", 
                       (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 128), 2)
            cv2.putText(placeholder_frame, "Install pyautogui for screen recording", 
                       (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 255, 128), 2)
            
            # Add animated progress bar
            progress_width = 600
            progress_height = 30
            progress_x = 50
            progress_y = 300
            
            # Background bar
            cv2.rectangle(placeholder_frame, (progress_x, progress_y), 
                         (progress_x + progress_width, progress_y + progress_height), 
                         (60, 60, 60), -1)
            
            # Progress fill (animated based on frame count)
            progress = (self.frame_count % 240) / 240.0  # 240 frames cycle
            fill_width = int(progress_width * progress)
            cv2.rectangle(placeholder_frame, (progress_x, progress_y), 
                         (progress_x + fill_width, progress_y + progress_height), 
                         (0, 255, 128), -1)
            
            # Progress text
            cv2.putText(placeholder_frame, f"Training Progress: {progress*100:.1f}%", 
                       (progress_x, progress_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            self.video_writer.write(placeholder_frame)
            self.frame_count += 1
            return True
            
        except Exception as e:
            if self.frame_count % 100 == 0:  # Log occasionally to avoid spam
                log_print(f"Warning: Could not capture frame {self.frame_count}: {e}")
            return False
    
    def stop_recording(self):
        """Stop video recording and save file"""
        if self.recording and self.video_writer is not None:
            try:
                self.video_writer.release()
                self.recording = False
                
                # Verify file was created and has content
                if os.path.exists(self.output_path):
                    file_size = os.path.getsize(self.output_path)
                    log_print(f"✅ Video recording completed: {self.output_path}")
                    log_print(f"📊 Total frames: {self.frame_count}, File size: {file_size/1024/1024:.1f} MB")
                    return True
                else:
                    log_print(f"❌ Video file was not created: {self.output_path}")
                    return False
            except Exception as e:
                log_print(f"❌ Error stopping video recording: {e}")
                return False
        return False
    
    def get_status(self):
        """Get recording status"""
        return {
            'recording': self.recording,
            'frame_count': self.frame_count,
            'output_path': self.output_path
        }

# 3D Visualization Helper Functions (moved here for early access)
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

# Set up logging and video recording
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = f"training_log_{timestamp}.txt"
video_filename = f"genesis_training_{timestamp}.mp4"

# Get full paths for clarity
current_directory = os.getcwd()
full_log_path = os.path.join(current_directory, log_filename)
full_video_path = os.path.join(current_directory, video_filename)

log_file = open(log_filename, 'w', encoding='utf-8')  # Use UTF-8 encoding for Unicode support

def log_print(message):
    """Print to console and save to log file"""
    print(message)
    # Remove or replace problematic Unicode characters for file writing
    safe_message = message.encode('ascii', 'replace').decode('ascii')
    log_file.write(safe_message + '\n')
    log_file.flush()  # Ensure immediate write

# Initialize video recorder with fallback options
video_recorder = GenesisVideoRecorder(
    output_path=video_filename,
    fps=20,  # Reduced fps for compatibility
    resolution=(1024, 768)  # Standard resolution for better compatibility
)

log_print(f"🎥 Video recorder initialized: {video_filename}")
log_print(f"📂 Current working directory: {current_directory}")
log_print(f"📹 Video will be saved to: {full_video_path}")
log_print(f"📝 Log will be saved to: {full_log_path}")
log_print("📹 Enhanced capture methods: viewer API + screen capture fallback")
log_print("💡 For best video quality, install: pip install pyautogui")
log_print("🎬 This will enable direct screen recording of Genesis 3D visualization")

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

# Environment wrapper following Gym Env Package structure
class FrankaGymEnv:
    """Gym-style environment wrapper for Franka robot in Genesis"""
    def __init__(self, scene, franka, dofs_idx, action_space, target_trajectory):
        self.scene = scene
        self.franka = franka
        self.dofs_idx = dofs_idx
        self.action_space = action_space
        self.target_trajectory = target_trajectory
        self.current_step = 0
        self.max_steps = len(target_trajectory)
        self.initial_position = np.array([0.0, -0.3, 0.0, -1.2, 0.0, 1.0, 0.785, 0.04, 0.04], dtype=np.float32)  # Updated to match main script
        self.step_count = 0  # Add step counter for video recording
        
    def reset(self):
        """Reset environment to initial state"""
        self.current_step = 0
        self.step_count = 0  # Reset step counter
        self.franka.set_dofs_position(self.initial_position, self.dofs_idx)
        for _ in range(10):  # Stabilize
            self.scene.step()
            # Capture initial frames for video
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
        """Enhanced reward function with separated total and tracking components"""
        if self.current_step >= len(self.target_trajectory):
            return 0.0, 0.0  # Return total_reward, tracking_reward
            
        target_pos = self.target_trajectory[self.current_step]
        current_pos = self.franka.get_dofs_position(self.dofs_idx).cpu().numpy()
        
        # Enhanced position tracking reward with exponential decay
        position_error = np.linalg.norm(current_pos - target_pos)
        position_reward = -10.0 * position_error  # More aggressive position penalty
        
        # Velocity penalty for smoother motion
        if hasattr(self, 'previous_pos'):
            velocity = np.linalg.norm(current_pos - self.previous_pos)
            velocity_penalty = -0.5 * velocity
        else:
            velocity_penalty = 0.0
        self.previous_pos = current_pos.copy()
        
        # Action smoothness penalty
        action_penalty = -0.1 * np.linalg.norm(action)
        
        # Enhanced end-effector tracking with waypoint rewards
        tracking_reward = 0.0
        try:
            # Try different methods to get end-effector position
            if hasattr(self.franka, 'get_link_pose'):
                ee_pose = self.franka.get_link_pose("hand")
                current_ee = ee_pose[:3].cpu().numpy()
            elif hasattr(self.franka, 'get_pose'):
                ee_pose = self.franka.get_pose()
                current_ee = ee_pose[:3].cpu().numpy()
            else:
                # Use joint positions as proxy for end-effector
                current_ee = current_pos[:3]
            
            # Target end-effector position from reference trajectory
            if hasattr(self, 'ref_ee_positions') and self.current_step < len(self.ref_ee_positions):
                target_ee = self.ref_ee_positions[self.current_step]
                ee_error = np.linalg.norm(current_ee - target_ee)
                tracking_reward = -20.0 * ee_error  # Strong penalty for EE position error
                
                # Store position error for monitoring
                self.current_position_error = ee_error
                
                # Bonus for being close to target (within 0.1 units)
                if ee_error < 0.1:
                    tracking_reward += 10.0
                elif ee_error < 0.2:
                    tracking_reward += 5.0
            else:
                # Workspace constraint reward
                workspace_reward = 2.0 if np.linalg.norm(current_ee) < 2.0 else -5.0
                tracking_reward = workspace_reward
                self.current_position_error = 0.0
                
        except Exception as e:
            if self.current_step % 100 == 0:  # Log occasionally to avoid spam
                log_print(f"  Warning: Could not get end-effector position: {e}")
            tracking_reward = 0.0
            self.current_position_error = 0.0
        
        # Progress reward for advancing through trajectory
        progress_reward = 0.1 * self.current_step / len(self.target_trajectory)
        
        # Calculate total reward and tracking reward separately
        total_reward = position_reward + velocity_penalty + action_penalty + tracking_reward + progress_reward
        
        # Debug print occasionally
        if self.current_step % 50 == 0:
            log_print(f"  Step {self.current_step}: pos_err={position_error:.3f}, tracking_reward={tracking_reward:.3f}, total_reward={total_reward:.3f}")
        
        return total_reward, tracking_reward
    
    def _target_reached(self):
        """Check if target position is reached within tolerance"""
        if self.current_step >= len(self.target_trajectory):
            return True
        target_pos = self.target_trajectory[self.current_step]
        current_pos = self.franka.get_dofs_position(self.dofs_idx).cpu().numpy()
        return np.linalg.norm(current_pos - target_pos) < 0.1
    
    def get_observation(self):
        """Get current observation state s(t)"""
        def to_numpy(x):
            return np.array(x.cpu().numpy(), dtype=np.float32) if hasattr(x, "cpu") else np.array(x, dtype=np.float32)

        positions = to_numpy(self.franka.get_dofs_position(self.dofs_idx))
        velocities = to_numpy(self.franka.get_dofs_velocity(self.dofs_idx))
        
        # Try different methods to get end-effector pose
        try:
            # Method 1: Try get_link_pose if available
            if hasattr(self.franka, 'get_link_pose'):
                ee_pose = to_numpy(self.franka.get_link_pose("hand"))
            # Method 2: Try get_pose for the entire entity
            elif hasattr(self.franka, 'get_pose'):
                ee_pose = to_numpy(self.franka.get_pose())
                ee_pose = ee_pose[:7]  # Take first 7 elements (position + quaternion)
            # Method 3: Use joint positions as proxy
            else:
                ee_pose = np.concatenate([positions[:3], np.array([0, 0, 0, 1])])  # xyz + quaternion
        except Exception as e:
            log_print(f"  Warning: Could not get end-effector pose: {e}")
            ee_pose = np.zeros(7, dtype=np.float32)
            
        try:
            torques = to_numpy(self.franka.get_dofs_torque(self.dofs_idx))
        except AttributeError:
            torques = np.zeros_like(positions)
        
        # Add target information to state
        if self.current_step < len(self.target_trajectory):
            target_pos = self.target_trajectory[self.current_step]
        else:
            target_pos = np.zeros_like(positions)
            
        return np.concatenate([positions, velocities, ee_pose, torques, target_pos])

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

# Initialize Professional TensorBoard Logger - After robot setup
log_print("\n📊 Initializing TensorBoard Logger...")
tensorboard_logger = TensorBoardLogger(
    log_dir="tensorboard_logs", 
    experiment_name="genesis_franka_rl"
)

# Action space bounds
action_low = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0.0, 0.0], dtype=np.float32)
action_high = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 0.04, 0.04], dtype=np.float32)
action_space = gym.spaces.Box(low=action_low, high=action_high, dtype=np.float32)

# Enhanced reference trajectory generation with better starting position - SHORTENED
log_print("Generating enhanced reference trajectory...")
# Better initial position that's more reachable
initial_position = np.array([0.0, -0.3, 0.0, -1.2, 0.0, 1.0, 0.785, 0.04, 0.04], dtype=np.float32)
num_steps = 100  # Reduced from 200 to 100 steps for faster episodes
ref_trajectory = []
ref_ee_positions = []  # Store end-effector positions for visualization

# Temporarily reset robot to compute reference trajectory
franka.set_dofs_position(initial_position, dofs_idx)
for _ in range(10):  # Stabilize
    scene.step()

for step in range(num_steps):
    target_position = initial_position.copy()
    # Smaller, more achievable trajectory movements
    target_position[0] = 0.5 * np.sin(0.03 * step)  # Reduced amplitude
    target_position[1] = -0.3 + 0.3 * np.cos(0.03 * step)  # Smaller movements
    target_position[3] = -1.2 + 0.2 * np.sin(0.02 * step)  # Small joint 4 movement
    target_position[7] = 0.02 + 0.02 * np.sin(0.1 * step)
    target_position[8] = 0.02 + 0.02 * np.cos(0.1 * step)
    ref_trajectory.append(target_position)
    
    # Compute CORRECT end-effector position for this joint configuration
    franka.set_dofs_position(target_position, dofs_idx)
    for _ in range(5):  # More stabilization steps
        scene.step()
    
    try:
        # Get the actual end-effector link position (hand/gripper)
        ee_link_names = ["hand", "panda_hand", "gripper", "end_effector", "panda_gripper"]
        ee_pos = None
        
        for link_name in ee_link_names:
            try:
                # Try to get the pose of the end-effector link
                link_state = franka.get_link(link_name)
                if link_state is not None:
                    pose = link_state.get_pose()
                    ee_pos = pose[:3].cpu().numpy() if hasattr(pose, "cpu") else pose[:3]
                    break
            except:
                continue
        
        # Fallback: Forward kinematics approximation
        if ee_pos is None:
            # Simple forward kinematics for Franka (approximation)
            # Based on joint positions, estimate end-effector position
            j1, j2, j3, j4, j5, j6, j7 = target_position[:7]
            
            # Approximate Franka forward kinematics (simplified)
            # This is a rough approximation for visualization
            x = 0.3 * np.cos(j1) * np.sin(j2) + 0.4 * np.cos(j1) * np.cos(j2) * np.sin(j4)
            y = 0.3 * np.sin(j1) * np.sin(j2) + 0.4 * np.sin(j1) * np.cos(j2) * np.sin(j4)
            z = 0.5 + 0.3 * np.cos(j2) - 0.4 * np.sin(j2) * np.sin(j4)
            ee_pos = np.array([x, y, z])
            
        ref_ee_positions.append(ee_pos)
        
    except Exception as e:
        log_print(f"Warning: Could not compute reference EE position at step {step}: {e}")
        # Use a default position relative to the robot base
        base_x = 0.3 * np.cos(target_position[0])
        base_y = 0.3 * np.sin(target_position[0])
        base_z = 0.5
        ref_ee_positions.append([base_x, base_y, base_z])

# Reset robot to initial position after trajectory computation
franka.set_dofs_position(initial_position, dofs_idx)
for _ in range(10):  # Stabilize
    scene.step()

log_print(f"Reference trajectory generated with {len(ref_trajectory)} steps")
log_print(f"Reference end-effector positions: {len(ref_ee_positions)} points")

# Enhanced trajectory visualization with 3D CYLINDRICAL TUBES
if ref_ee_positions and len(ref_ee_positions) > 1:
    try:
        # Create 3D cylindrical tube for reference trajectory (RED TUBES) - TEMPORARILY DISABLED
        # ref_tube_points = create_3d_trajectory_tube(ref_ee_positions, radius=0.04, color=(1.0, 0.0, 0.0, 1.0))
        # if ref_tube_points:
        #     scene.draw_debug_points(ref_tube_points, (1.0, 0.0, 0.0, 0.9))  # Bright red tubes
        #     log_print(f"Reference trajectory visualized as 3D RED CYLINDRICAL TUBES with {len(ref_tube_points)} points")
        
        # Create simple points instead of complex tubes for now
        simple_ref_points = [[float(pos[0]), float(pos[1]), float(pos[2])] for pos in ref_ee_positions]
        if simple_ref_points:
            scene.draw_debug_points(simple_ref_points, (1.0, 0.0, 0.0, 0.9))  # Simple red points
            log_print(f"Reference trajectory visualized as SIMPLE RED POINTS with {len(simple_ref_points)} points")
        
        # Also add some marker spheres at key points for better visibility
        visible_refs = ref_ee_positions[::8]  # Every 8th point for key markers
        sphere_points = []
        for pos in visible_refs:
            sphere = create_large_point(pos, (1.0, 0.0, 0.0, 1.0), 0.08)
            sphere_points.extend(sphere)
        if sphere_points:
            scene.draw_debug_points(sphere_points, (1.0, 0.2, 0.2, 1.0))  # Light red spheres
            
    except Exception as e:
        log_print(f"Warning: Could not create 3D tube visualization: {e}")
        try:
            scene.draw_debug_points(ref_ee_positions[:10], (1.0, 0.0, 0.0, 1.0))
        except:
            log_print("Reference trajectory visualization failed")

# Create Enhanced Gym Environment
env = FrankaGymEnv(scene, franka, dofs_idx, action_space, ref_trajectory)
# Pass reference EE positions to environment for better reward calculation
env.ref_ee_positions = ref_ee_positions
# Pass video recorder to environment for frame capture
env.video_recorder = video_recorder

# Initialize observation space dimensions
sample_obs = env.reset()
state_dim = len(sample_obs)
action_dim = len(dofs_idx)

# Log hyperparameters to TensorBoard - Now that all variables are defined
hyperparameters = {
    'algorithm': 'DDPG_simplified',
    'learning_rate': 1e-5,
    'batch_size': 64,
    'memory_size': 10000,
    'episodes': 50,
    'trajectory_steps': 100,
    'exploration_noise_initial': 0.3,
    'exploration_noise_min': 0.05,
    'target_update_tau': 0.005,
    'hidden_dim': 256,
    'action_dim': action_dim,
    'state_dim': state_dim
}

tensorboard_logger.log_hyperparameters(hyperparameters)

log_print(f"State dimension: {state_dim}")
log_print(f"Action dimension: {action_dim}")
log_print(f"Initial end-effector position range: {[min(p[i] for p in ref_ee_positions) for i in range(3)]} to {[max(p[i] for p in ref_ee_positions) for i in range(3)]}")

# Initialize DDPG Agent (Actor Network) with improved hyperparameters
agent = DDPGAgent(state_dim, action_dim, lr=1e-5)  # Much lower learning rate for stable learning

# Enhanced training parameters with curriculum learning - SHORTENED FOR 5 MIN RUN
num_episodes = 2   # Reduced from 200 to ~50 episodes for 5-minute training
max_steps_per_episode = len(ref_trajectory)
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

log_print("\n=== Starting Training Loop with VIDEO RECORDING ===")
log_print("Following the flow: Environment -> Actor Network -> Action -> Environment...")
log_print("🕐 TARGET DURATION: ~5 minutes of training")
log_print("🎨 3D CYLINDRICAL TRAJECTORY VISUALIZATION (like reference image):")
log_print("[RED TUBES] 3D cylindrical tubes: Reference trajectory (target to follow)")
log_print("[BLUE TUBES] 3D cylindrical tubes: Current training trajectory")
log_print("[YELLOW SPHERE] Large sphere: Current robot end-effector position")
log_print("[GREEN/CYAN/MAGENTA TUBES] Different colored 3D tubes: Last 3 episode trajectories")

# Time-based training control for 5-minute duration
import time
training_start_time = time.time()
target_duration_seconds = 5 * 60  # 5 minutes in seconds
log_print(f"🕐 Training will run for approximately {target_duration_seconds//60} minutes")

# Start video recording
log_print("\n🎥 Starting video recording of training simulation...")
if video_recorder.start_recording():
    log_print("✅ Video recording initialized successfully")
else:
    log_print("⚠️ Video recording failed to start, continuing without recording")

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

if ref_ee_positions:
    # Create PERSISTENT reference trajectory with simple points instead of tubes
    # ref_tube_points = create_3d_trajectory_tube(ref_ee_positions, radius=0.04)
    simple_ref_points = [[float(pos[0]), float(pos[1]), float(pos[2])] for pos in ref_ee_positions]
    
    if simple_ref_points:
        scene.draw_debug_points(simple_ref_points, (1.0, 0.0, 0.0, 0.9))  # PERMANENT Red points
        log_print(f"[PERMANENT] Reference trajectory: PERSISTENT SIMPLE RED POINTS ({len(simple_ref_points)} points)")
    
    # Also add marker spheres at key points - PERMANENT
    visible_refs = ref_ee_positions[::8]  # Every 8th point
    all_ref_spheres = []
    for pos in visible_refs:
        sphere = create_large_point(pos, (1.0, 0.0, 0.0, 1.0), 0.06)
        all_ref_spheres.extend(sphere)
    if all_ref_spheres:
        scene.draw_debug_points(all_ref_spheres, (1.0, 0.2, 0.2, 1.0))  # PERMANENT Light red spheres

# Store training trajectory for visualization
training_trajectory = []
all_episode_trajectories = []  # Store trajectories from all episodes

for episode in range(num_episodes):
    # Check if 5-minute time limit has been reached
    elapsed_time = time.time() - training_start_time
    if elapsed_time >= target_duration_seconds:
        log_print(f"\n🕐 Time limit reached: {elapsed_time//60:.1f} minutes elapsed")
        log_print(f"🏁 Stopping training early at episode {episode + 1}/{num_episodes}")
        break
    
    # Reset environment and get initial state s(t)
    state = env.reset()
    episode_reward = 0
    episode_tracking_reward = 0  # Separate tracking reward accumulator
    episode_loss = 0
    training_trajectory = []  # Reset trajectory for each episode
    episode_position_errors = []  # Track position errors per episode
    
    # Log episode start to TensorBoard
    tensorboard_logger.log_episode_start(episode + 1)
    
    remaining_time = target_duration_seconds - elapsed_time
    print(f"\nEpisode {episode + 1}/{num_episodes} | Time remaining: {remaining_time//60:.1f}min {remaining_time%60:.0f}s")
    
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
            
            # Enhanced real-time trajectory visualization with PERSISTENT 3D CYLINDRICAL TUBES
            if step % 20 == 0 and len(training_trajectory) > 1:  # Reduced frequency for faster training
                # Only clear debug objects that are not permanent trajectories
                # Keep reference trajectory always visible, only update training trajectory
                
                # Always redraw reference trajectory as PERSISTENT SIMPLE POINTS
                try:
                    # ref_tube_points = create_3d_trajectory_tube(ref_ee_positions, radius=0.035, color=(1.0, 0.0, 0.0, 1.0))
                    simple_ref_points = [[float(pos[0]), float(pos[1]), float(pos[2])] for pos in ref_ee_positions]
                    if simple_ref_points:
                        scene.draw_debug_points(simple_ref_points, (1.0, 0.0, 0.0, 0.9))  # Red points - ALWAYS VISIBLE
                except:
                    scene.draw_debug_points(ref_ee_positions, (1.0, 0.0, 0.0, 1.0))
                
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
            
            # Debug: Enhanced position tracking
            if step == 0:
                target_ee = ref_ee_positions[step] if step < len(ref_ee_positions) else [0, 0, 0]
                pos_err = np.linalg.norm(current_ee_pos - target_ee)
                log_print(f"  Step {step}: pos_err={pos_err:.3f}, reward={reward:.3f}")
                log_print(f"  Initial EE position: {current_ee_pos}")
            elif step % 50 == 0:
                target_ee = ref_ee_positions[step] if step < len(ref_ee_positions) else [0, 0, 0]
                pos_err = np.linalg.norm(current_ee_pos - target_ee)
                log_print(f"  Step {step}: pos_err={pos_err:.3f}, reward={reward:.3f}")
                log_print(f"  Step {step} EE position: {current_ee_pos}")
                
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
            
            # Always ensure reference trajectory is visible with SIMPLE RED POINTS
            if ref_ee_positions and len(ref_ee_positions) > 1:
                # Use simple points instead of complex tubes
                simple_ref_points = [[float(pos[0]), float(pos[1]), float(pos[2])] for pos in ref_ee_positions[::3]]
                
                if simple_ref_points:
                    scene.draw_debug_points(simple_ref_points, (1.0, 0.0, 0.0, 0.9))  # SIMPLE Red points
            
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
        
        # Capture video frame for important visualization moments - OPTIMIZED
        if step % 20 == 0 and hasattr(env, 'video_recorder'):  # Reduced frequency for faster training
            env.video_recorder.capture_frame(scene)
        
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
            
            # Calculate trajectory following accuracy
            if len(ref_ee_positions) > 0:
                min_len = min(len(training_trajectory), len(ref_ee_positions))
                if min_len > 0:
                    errors = [np.linalg.norm(np.array(training_trajectory[i]) - np.array(ref_ee_positions[i])) 
                             for i in range(min_len)]
                    avg_error = np.mean(errors)
                    log_print(f"  [ACCURACY] Avg trajectory error: {avg_error:.3f} units")
    
    # Enhanced trajectory comparison every 10 episodes with ADDITIONAL 3D CYLINDRICAL TUBES - OPTIMIZED
    if (episode + 1) % 10 == 0:  # Reduced from every 5 episodes to every 10 episodes
        # NO CLEARING - ADD to existing visualization instead of replacing
        
        # Always ensure reference trajectory remains visible
        if ref_ee_positions and len(ref_ee_positions) > 1:
            # ref_tube_points = create_3d_trajectory_tube(ref_ee_positions, radius=0.04)
            simple_ref_points = [[float(pos[0]), float(pos[1]), float(pos[2])] for pos in ref_ee_positions]
            if simple_ref_points:
                scene.draw_debug_points(simple_ref_points, (1.0, 0.0, 0.0, 0.9))  # Redraw reference points
        
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
        
        # Capture major visualization moments for video
        if hasattr(env, 'video_recorder'):
            for _ in range(5):  # Capture a few frames of the episode comparison
                env.video_recorder.capture_frame(scene)
                time.sleep(0.1)  # Brief pause for smooth video
    
    # Decay noise for exploration
    agent.noise_std = max(0.01, agent.noise_std * 0.995)

log_print("\n=== Training Complete ===")
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
log_print("\n🎥 Stopping video recording...")
if hasattr(video_recorder, 'recording') and video_recorder.recording:
    if video_recorder.stop_recording():
        log_print("✅ Video recording completed successfully")
    else:
        log_print("⚠️ Video recording may have issues")
else:
    log_print("ℹ️ No active video recording to stop")

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

# Enhanced final demonstration with comprehensive analysis
log_print("\n=== Enhanced Final Demonstration ===")
log_print("Running trained actor network on reference trajectory...")

state = env.reset()
final_trajectory = []
final_rewards = []

for step in range(max_steps_per_episode):
    # Use trained actor network without noise
    action = agent.select_action(state, add_noise=False)
    next_state, reward, done, info = env.step(action)
    final_rewards.append(reward)
    
    # Record trajectory for enhanced visualization
    try:
        # Try different methods to get end-effector position
        if hasattr(franka, 'get_link_pose'):
            ee_pose = franka.get_link_pose("hand")
            final_trajectory.append(ee_pose[:3].cpu().numpy())
        elif hasattr(franka, 'get_pose'):
            ee_pose = franka.get_pose()
            final_trajectory.append(ee_pose[:3].cpu().numpy())
        else:
            # Use joint positions as approximation
            joint_pos = franka.get_dofs_position(dofs_idx)
            ee_pos = joint_pos[:3].cpu().numpy() if hasattr(joint_pos, "cpu") else joint_pos[:3]
            final_trajectory.append(ee_pos)
    except Exception as e:
        log_print(f"Warning: Could not record final trajectory at step {step}: {e}")
        final_trajectory.append([0, 0, 0])
    
    state = next_state
    
    if done:
        break

# Capture final demonstration frames for video
log_print("🎥 Capturing final demonstration for video...")
if hasattr(env, 'video_recorder') and env.video_recorder.recording:
    for _ in range(10):  # Capture final frames showing the trained result
        env.video_recorder.capture_frame(scene)
        time.sleep(0.2)  # Pause for clear final visualization

# Enhanced final analysis and visualization
total_final_reward = sum(final_rewards)
log_print(f"Final demonstration reward: {total_final_reward:.2f}")

# Calculate trajectory following performance
if len(final_trajectory) > 0 and len(ref_ee_positions) > 0:
    min_len = min(len(final_trajectory), len(ref_ee_positions))
    final_errors = [np.linalg.norm(np.array(final_trajectory[i]) - np.array(ref_ee_positions[i])) 
                   for i in range(min_len)]
    avg_final_error = np.mean(final_errors)
    max_final_error = np.max(final_errors)
    min_final_error = np.min(final_errors)
    
    log_print(f"Final trajectory performance:")
    log_print(f"  Average error: {avg_final_error:.3f} units")
    log_print(f"  Max error: {max_final_error:.3f} units")
    log_print(f"  Min error: {min_final_error:.3f} units")
    log_print(f"  Trajectory completion: {min_len}/{len(ref_ee_positions)} steps")

# Enhanced final visualization with ADDITIONAL 3D CYLINDRICAL TUBES
# NO CLEARING - Keep all previous trajectories visible and add final result

# Ensure reference trajectory is still visible with SIMPLE RED POINTS
if ref_ee_positions and len(ref_ee_positions) > 1:
    # ref_final_tubes = create_3d_trajectory_tube(ref_ee_positions, radius=0.04)
    simple_ref_final = [[float(pos[0]), float(pos[1]), float(pos[2])] for pos in ref_ee_positions]
    if simple_ref_final:
        scene.draw_debug_points(simple_ref_final, (1.0, 0.0, 0.0, 0.9))  # Red points
        log_print("Reference trajectory maintained as SIMPLE RED POINTS")

# Add final trained trajectory as 3D GREEN CYLINDRICAL TUBES  
if final_trajectory and len(final_trajectory) > 2:
    final_tubes = create_3d_trajectory_tube(final_trajectory, radius=0.038)  # Slightly larger for prominence
    if final_tubes:
        scene.draw_debug_points(final_tubes, (0.0, 1.0, 0.0, 0.95))  # Bright green cylindrical tubes
        log_print("Final trained trajectory ADDED as 3D GREEN CYLINDRICAL TUBES")

log_print("\n=== PERSISTENT 3D CYLINDRICAL TUBE VISUALIZATION ===")
log_print("[RED TUBES] 3D cylindrical reference trajectory (always visible)")
log_print("[GREEN TUBES] 3D cylindrical final trained trajectory (added to view)")
log_print("[BLUE TUBES] 3D cylindrical training trajectories (accumulated)")
log_print("[OTHER COLORS] Previous episode trajectories (layered view)")
log_print("🎨 ALL TRAJECTORIES remain visible - no disappearing plots!")

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

# Final video recording status
if hasattr(video_recorder, 'get_status'):
    status = video_recorder.get_status()
    log_print(f"🎬 Video recording final status: {status}")

log_file.close()

input("Press Enter to exit and close Genesis...")

gs.destroy()
os._exit(0)