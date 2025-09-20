"""
Genesis-Safe TensorBoard Integration Guide and Modified Training Loop
This file shows how to integrate genesis_safe_tensorboard.py with your existing training code
to ensure clean shutdown and compatibility with gs.destroy()
"""

import os
import sys
import time
import signal
import atexit
import genesis as gs
from genesis_safe_tensorboard import create_genesis_safe_tensorboard

# Example of how to integrate Genesis-safe TensorBoard with your existing training code

class GenesisSafeTrainingSession:
    """
    Enhanced training session with Genesis-safe TensorBoard integration
    Ensures proper cleanup order and no interference with gs.destroy()
    """
    
    def __init__(self, config=None):
        self.config = config or {}
        self.tb_logger = None
        self.genesis_sim = None
        self.cleanup_completed = False
        
        # Register emergency cleanup
        atexit.register(self._emergency_cleanup)
        signal.signal(signal.SIGINT, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals with proper cleanup order"""
        if not self.cleanup_completed:
            print("\\n🔄 Interrupt received, initiating safe shutdown...")
            self.safe_shutdown()
            sys.exit(0)
    
    def _emergency_cleanup(self):
        """Emergency cleanup for unexpected exits"""
        if not self.cleanup_completed:
            self.safe_shutdown()
    
    def initialize_tensorboard(self, experiment_name=None):
        """Initialize Genesis-safe TensorBoard logging"""
        try:
            self.tb_logger = create_genesis_safe_tensorboard(
                log_dir="tensorboard_logs",
                experiment_name=experiment_name,
                auto_start=True
            )
            
            # Log hyperparameters if provided
            if self.config:
                self.tb_logger.log_hyperparameters(self.config)
            
            return True
        except Exception as e:
            print(f"⚠️ TensorBoard initialization failed: {e}")
            return False
    
    def initialize_genesis(self):
        """Initialize Genesis simulation"""
        try:
            # Your Genesis initialization code here
            gs.init()
            print("✅ Genesis initialized successfully")
            return True
        except Exception as e:
            print(f"❌ Genesis initialization failed: {e}")
            return False
    
    def run_training(self, num_episodes=100):
        """Main training loop with proper cleanup handling"""
        try:
            print(f"🚀 Starting training for {num_episodes} episodes...")
            
            for episode in range(num_episodes):
                # Episode initialization
                self.tb_logger.log_episode_start(episode)
                
                episode_reward = 0
                episode_tracking_reward = 0
                episode_position_error = 0
                step_count = 0
                
                # Your episode loop here
                for step in range(1000):  # Max 1000 steps per episode
                    # Your training step logic here
                    # ...
                    
                    # Example step data
                    step_data = {
                        'step': step,
                        'reward': 0.1,  # Replace with actual reward
                        'total_reward': episode_reward,
                        'tracking_reward': 0.05,  # Replace with actual tracking reward
                        'position_error': 0.01,  # Replace with actual position error
                        'action': [0.1, 0.2, 0.3],  # Replace with actual action
                    }
                    
                    # Log step metrics safely
                    self.tb_logger.log_step_metrics(step_data)
                    
                    episode_reward += step_data['reward']
                    episode_tracking_reward += step_data['tracking_reward']
                    episode_position_error += step_data['position_error']
                    step_count += 1
                    
                    # Break condition (replace with your actual termination logic)
                    if step > 100:  # Example: episode ends after 100 steps
                        break
                
                # Log episode metrics safely
                episode_data = {
                    'episode': episode,
                    'total_reward': episode_reward,
                    'tracking_reward': episode_tracking_reward,
                    'position_error': episode_position_error / step_count,
                    'episode_length': step_count,
                }
                
                self.tb_logger.log_episode_metrics(episode_data)
                
                # Progress feedback
                if episode % 10 == 0:
                    print(f"📈 Episode {episode}: Reward={episode_reward:.3f}, "
                          f"Tracking={episode_tracking_reward:.3f}, "
                          f"Error={episode_position_error/step_count:.4f}")
            
            print("🎉 Training completed successfully!")
            return True
            
        except KeyboardInterrupt:
            print("\\n⚠️ Training interrupted by user")
            return False
        except Exception as e:
            print(f"❌ Training error: {e}")
            return False
    
    def safe_shutdown(self):
        """Genesis-safe shutdown sequence - CRITICAL ORDER"""
        if self.cleanup_completed:
            return
        
        print("🔄 Initiating Genesis-safe shutdown sequence...")
        
        # Step 1: Finalize TensorBoard FIRST (before Genesis cleanup)
        if self.tb_logger:
            try:
                print("🔄 Step 1: Finalizing TensorBoard...")
                self.tb_logger.genesis_safe_finalize()
                print("✅ TensorBoard finalized successfully")
            except Exception as e:
                print(f"⚠️ TensorBoard finalization warning: {e}")
            finally:
                self.tb_logger = None
        
        # Step 2: Small delay to ensure TensorBoard cleanup completion
        time.sleep(1)
        
        # Step 3: Genesis cleanup (after TensorBoard is fully stopped)
        try:
            print("🔄 Step 2: Genesis cleanup...")
            gs.destroy()
            print("✅ Genesis destroyed successfully")
        except Exception as e:
            print(f"⚠️ Genesis cleanup warning: {e}")
        
        # Step 4: Mark cleanup as completed
        self.cleanup_completed = True
        print("✅ Genesis-safe shutdown completed")

# Example usage functions

def example_modified_training_integration():
    """
    Example of how to modify your existing training code for Genesis-safe TensorBoard
    """
    
    # Configuration
    config = {
        'learning_rate': 0.001,
        'batch_size': 64,
        'buffer_size': 100000,
        'gamma': 0.99,
        'tau': 0.005
    }
    
    # Create training session
    session = GenesisSafeTrainingSession(config)
    
    # Initialize components in the correct order
    print("🔧 Initializing training session...")
    
    # 1. Initialize TensorBoard first
    if not session.initialize_tensorboard("franka_ddpg_training"):
        print("❌ Failed to initialize TensorBoard")
        return
    
    # 2. Initialize Genesis second
    if not session.initialize_genesis():
        print("❌ Failed to initialize Genesis")
        session.safe_shutdown()
        return
    
    # 3. Run training
    try:
        session.run_training(num_episodes=100)
    finally:
        # 4. Always call safe shutdown
        session.safe_shutdown()

def integrate_with_existing_code_pattern():
    """
    Pattern for integrating with your existing NewTest_v1_pure_env.py
    """
    
    # Replace your existing TensorBoard initialization with:
    tb_logger = create_genesis_safe_tensorboard(
        experiment_name="franka_ddpg_training",
        auto_start=True
    )
    
    try:
        # Your existing Genesis and training code here
        # ...
        
        # Replace your existing logging calls with:
        # tb_logger.log_step_metrics(step_data)
        # tb_logger.log_episode_metrics(episode_data)
        
        pass  # Your training logic here
        
    finally:
        # CRITICAL: Replace your cleanup section with:
        print("🔄 Starting Genesis-safe cleanup...")
        
        # 1. Finalize TensorBoard FIRST
        tb_logger.genesis_safe_finalize()
        
        # 2. Brief pause
        time.sleep(1)
        
        # 3. Genesis cleanup SECOND
        gs.destroy()
        
        # 4. Exit cleanly
        print("✅ All cleanup completed successfully")
        os._exit(0)

def quick_integration_guide():
    """
    Quick integration steps for your existing code
    """
    print("""
    🔧 Genesis-Safe TensorBoard Integration Guide:
    
    1. Import the new module:
       from genesis_safe_tensorboard import create_genesis_safe_tensorboard
    
    2. Replace your TensorBoard initialization:
       tb_logger = create_genesis_safe_tensorboard("experiment_name")
    
    3. Replace your cleanup section with this exact order:
       # Finalize TensorBoard FIRST
       tb_logger.genesis_safe_finalize()
       time.sleep(1)
       
       # Genesis cleanup SECOND
       gs.destroy()
       
       # Exit cleanly
       os._exit(0)
    
    4. The new TensorBoard logger has the same methods:
       - log_step_metrics(step_data)
       - log_episode_metrics(episode_data)
       - log_hyperparameters(config)
    
    ✅ This ensures TensorBoard never interferes with gs.destroy()!
    """)

if __name__ == "__main__":
    # Run example
    quick_integration_guide()
    # example_modified_training_integration()
