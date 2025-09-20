"""
Genesis-Safe TensorBoard Implementation
Ensures clean shutdown and compatibility with gs.destroy()
"""

import os
import subprocess
import webbrowser
import time
import threading
import signal
import atexit
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import logging

# Configure logging to suppress unnecessary warnings
logging.getLogger('tensorboard').setLevel(logging.ERROR)

class GenesisSafeTensorBoardManager:
    """
    Genesis-compatible TensorBoard manager that ensures clean shutdown
    and won't interfere with gs.destroy()
    """
    
    def __init__(self, log_dir="tensorboard_logs", port=6006, auto_open_browser=True):
        self.log_dir = log_dir
        self.port = port
        self.auto_open_browser = auto_open_browser
        self.tensorboard_process = None
        self.is_running = False
        self.shutdown_initiated = False
        
        # Register cleanup functions
        atexit.register(self._emergency_cleanup)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully"""
        if not self.shutdown_initiated:
            print("🔄 TensorBoard received shutdown signal...")
            self.stop_tensorboard(silent=True)
    
    def _emergency_cleanup(self):
        """Emergency cleanup for atexit"""
        if not self.shutdown_initiated and self.tensorboard_process:
            try:
                self.stop_tensorboard(silent=True)
            except:
                pass
    
    def start_tensorboard(self):
        """Start TensorBoard with Genesis-safe settings"""
        try:
            # Stop any existing TensorBoard first
            self.stop_tensorboard(silent=True)
            
            # Genesis-safe TensorBoard command with minimal resource usage
            cmd = [
                "tensorboard",
                f"--logdir={self.log_dir}",
                f"--port={self.port}",
                "--reload_interval=10",  # Less frequent reloads to reduce resource usage
                "--max_reload_threads=1",  # Minimal threading to avoid conflicts
                "--purge_orphaned_data=true",  # Clean up old data
                "--bind_all=false"  # Only bind to localhost for security
            ]
            
            # Start TensorBoard process with proper isolation
            self.tensorboard_process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0,
                start_new_session=True if os.name != 'nt' else False
            )
            
            self.is_running = True
            print(f"🚀 Genesis-safe TensorBoard started on port {self.port}")
            print(f"📊 Dashboard: http://localhost:{self.port}")
            
            # Wait briefly for TensorBoard to start
            time.sleep(2)
            
            # Auto-open browser if requested
            if self.auto_open_browser:
                try:
                    webbrowser.open(f"http://localhost:{self.port}")
                    print("🌐 TensorBoard opened in browser")
                except Exception as e:
                    print(f"⚠️ Could not auto-open browser: {e}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to start TensorBoard: {e}")
            return False
    
    def stop_tensorboard(self, silent=False):
        """Genesis-safe TensorBoard shutdown"""
        if self.shutdown_initiated:
            return True
            
        self.shutdown_initiated = True
        
        try:
            if self.tensorboard_process and self.tensorboard_process.poll() is None:
                if not silent:
                    print("🛑 Stopping TensorBoard gracefully...")
                
                # Method 1: Graceful termination
                try:
                    self.tensorboard_process.terminate()
                    
                    # Wait for graceful shutdown with timeout
                    try:
                        self.tensorboard_process.wait(timeout=3)
                        if not silent:
                            print("✅ TensorBoard stopped gracefully")
                    except subprocess.TimeoutExpired:
                        # Method 2: Force kill if needed
                        if not silent:
                            print("⚠️ TensorBoard didn't respond, force stopping...")
                        self.tensorboard_process.kill()
                        time.sleep(1)
                        
                except Exception as e:
                    if not silent:
                        print(f"⚠️ TensorBoard stop warning: {e}")
                    # Force kill as last resort
                    try:
                        self.tensorboard_process.kill()
                    except:
                        pass
            
            self.is_running = False
            self.tensorboard_process = None
            
            # Brief pause to ensure cleanup
            time.sleep(0.5)
            
            return True
            
        except Exception as e:
            if not silent:
                print(f"⚠️ TensorBoard cleanup error: {e}")
            return False
    
    def is_tensorboard_running(self):
        """Check if TensorBoard is still running"""
        if self.tensorboard_process:
            return self.tensorboard_process.poll() is None
        return False
    
    def genesis_safe_cleanup(self):
        """Special cleanup method for Genesis compatibility"""
        print("🔄 Initiating Genesis-safe TensorBoard cleanup...")
        
        # Stop TensorBoard immediately
        self.stop_tensorboard(silent=False)
        
        # Ensure all subprocesses are cleaned up
        time.sleep(1)
        
        print("✅ TensorBoard cleanup completed for Genesis")
        return True

class GenesisSafeTensorBoardLogger:
    """
    Genesis-safe TensorBoard logger that ensures clean shutdown
    Compatible with gs.destroy() and won't cause hanging
    """
    
    def __init__(self, log_dir="tensorboard_logs", experiment_name=None, auto_start=True):
        """Initialize Genesis-safe TensorBoard logger"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if experiment_name:
            self.log_dir = os.path.join(log_dir, f"{experiment_name}_{timestamp}")
        else:
            self.log_dir = os.path.join(log_dir, f"genesis_training_{timestamp}")
        
        # Create TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        # Initialize Genesis-safe TensorBoard manager
        self.tb_manager = GenesisSafeTensorBoardManager(
            log_dir=log_dir, 
            auto_open_browser=auto_start
        )
        
        # Start TensorBoard if requested
        if auto_start:
            success = self.tb_manager.start_tensorboard()
            if success:
                print("🔥 Genesis-safe TensorBoard initialized successfully")
            else:
                print("⚠️ TensorBoard auto-start failed, but logging will continue")
        
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
        
        print(f"🔥 Genesis-safe TensorBoard logging initialized: {self.log_dir}")
        print(f"📊 Dashboard access: http://localhost:{self.tb_manager.port}")
    
    def log_hyperparameters(self, hparams):
        """Log hyperparameters to TensorBoard"""
        try:
            self.writer.add_hparams(hparams, {})
            print("📝 Hyperparameters logged to TensorBoard")
        except Exception as e:
            print(f"⚠️ Hyperparameter logging error: {e}")
    
    def log_episode_start(self, episode):
        """Log episode start"""
        self.episode_step = episode
    
    def log_step_metrics(self, step_data):
        """Log individual step metrics with error handling"""
        try:
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
                if isinstance(action, (list, tuple)):
                    for i, a in enumerate(action):
                        if a is not None:
                            self.writer.add_scalar(f'Step/Action_{i}', a, self.global_step)
                else:
                    self.writer.add_scalar('Step/Action', action, self.global_step)
            
            if 'loss' in step_data and step_data['loss'] is not None:
                self.writer.add_scalar('Training/Loss', step_data['loss'], self.global_step)
            
            self.global_step += 1
            
        except Exception as e:
            # Silently continue if logging fails to avoid breaking training
            pass
    
    def log_episode_metrics(self, episode_data):
        """Log episode-level metrics with error handling"""
        try:
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
                
        except Exception as e:
            # Silently continue if logging fails
            pass
    
    def _log_rolling_statistics(self, episode):
        """Log rolling statistics for better trend analysis"""
        try:
            # Calculate rolling averages for different window sizes
            for window in [10, 50, 100]:
                if len(self.episode_metrics['total_rewards']) >= window:
                    # Total rewards rolling statistics
                    recent_rewards = self.episode_metrics['total_rewards'][-window:]
                    self.writer.add_scalar(f'Episode/Total_Reward_Avg_{window}', sum(recent_rewards)/len(recent_rewards), episode)
                    
                    # Tracking rewards rolling statistics
                    if len(self.episode_metrics['tracking_rewards']) >= window:
                        recent_tracking = self.episode_metrics['tracking_rewards'][-window:]
                        self.writer.add_scalar(f'Episode/Tracking_Reward_Avg_{window}', sum(recent_tracking)/len(recent_tracking), episode)
        except:
            pass
    
    def log_network_weights(self, model, step):
        """Log network weights and gradients with error handling"""
        try:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    self.writer.add_histogram(f'Weights/{name}', param.data, step)
                    self.writer.add_histogram(f'Gradients/{name}', param.grad.data, step)
        except Exception as e:
            # Silently continue if logging fails
            pass
    
    def genesis_safe_finalize(self):
        """Genesis-safe finalization that ensures clean shutdown"""
        print("🔄 Starting Genesis-safe TensorBoard finalization...")
        
        # Calculate final statistics
        final_stats = None
        try:
            if self.episode_metrics['total_rewards']:
                final_stats = {
                    'Total Episodes': len(self.episode_metrics['total_rewards']),
                    'Best Total Reward': max(self.episode_metrics['total_rewards']),
                    'Final Total Reward': self.episode_metrics['total_rewards'][-1],
                    'Average Total Reward': sum(self.episode_metrics['total_rewards']) / len(self.episode_metrics['total_rewards']),
                    'Total Reward Improvement': self.episode_metrics['total_rewards'][-1] - self.episode_metrics['total_rewards'][0] if len(self.episode_metrics['total_rewards']) > 1 else 0
                }
                
                if self.episode_metrics['tracking_rewards']:
                    final_stats.update({
                        'Best Tracking Reward': max(self.episode_metrics['tracking_rewards']),
                        'Final Tracking Reward': self.episode_metrics['tracking_rewards'][-1],
                        'Average Tracking Reward': sum(self.episode_metrics['tracking_rewards']) / len(self.episode_metrics['tracking_rewards'])
                    })
                
                # Log final summary as text
                summary_text = "\\n".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                                        for k, v in final_stats.items()])
                self.writer.add_text('Final_Summary', summary_text, self.episode_step)
        except Exception as e:
            print(f"⚠️ Final statistics calculation error: {e}")
        
        # Close TensorBoard writer first
        try:
            self.writer.close()
            print("✅ TensorBoard writer closed")
        except Exception as e:
            print(f"⚠️ TensorBoard writer close error: {e}")
        
        # Genesis-safe TensorBoard cleanup
        self.tb_manager.genesis_safe_cleanup()
        
        print(f"📊 TensorBoard results saved in: {os.path.dirname(self.log_dir)}")
        print("✅ Genesis-safe TensorBoard finalization completed")
        
        return final_stats
    
    def finalize(self):
        """Standard finalize method that calls genesis_safe_finalize"""
        return self.genesis_safe_finalize()

# Convenience function for easy integration
def create_genesis_safe_tensorboard(log_dir="tensorboard_logs", experiment_name=None, auto_start=True):
    """
    Create a Genesis-safe TensorBoard logger
    
    Args:
        log_dir: Directory for TensorBoard logs
        experiment_name: Name of the experiment
        auto_start: Whether to auto-start TensorBoard
    
    Returns:
        GenesisSafeTensorBoardLogger instance
    """
    return GenesisSafeTensorBoardLogger(
        log_dir=log_dir,
        experiment_name=experiment_name,
        auto_start=auto_start
    )
