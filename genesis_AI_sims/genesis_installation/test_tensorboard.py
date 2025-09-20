#!/usr/bin/env python3
"""
Test TensorBoard functionality with sample data
"""

import os
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter
import datetime

def test_tensorboard_logging():
    """Test TensorBoard logging with sample training data"""
    
    # Create TensorBoard writer
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f"tensorboard_logs/test_run_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir)
    print(f"📊 Testing TensorBoard logging: {log_dir}")
    
    # Log hyperparameters
    hparams = {
        'learning_rate': 0.001,
        'batch_size': 64,
        'episodes': 50,
        'algorithm': 'DDPG_test'
    }
    writer.add_hparams(hparams, {})
    print("✅ Hyperparameters logged")
    
    # Simulate training data over 10 episodes
    for episode in range(10):
        print(f"Logging episode {episode + 1}/10...")
        
        # Simulate episode metrics
        base_reward = -1000 + episode * 50  # Improving over time
        episode_total_reward = base_reward + np.random.normal(0, 100)
        episode_tracking_reward = base_reward * 0.6 + np.random.normal(0, 50)
        
        # Log episode-level metrics
        writer.add_scalar('Episodes/Total_Reward', episode_total_reward, episode)
        writer.add_scalar('Episodes/Tracking_Reward', episode_tracking_reward, episode)
        writer.add_scalar('Episodes/Episode_Length', 100, episode)
        
        # Simulate step-level data for this episode
        for step in range(20):  # 20 steps per episode for testing
            global_step = episode * 20 + step
            
            # Simulate step metrics
            step_reward = np.random.normal(-10, 5)
            position_error = max(0, np.random.normal(1.0, 0.5))
            action_norm = np.random.uniform(0.1, 1.0)
            loss = max(0, np.random.normal(0.5, 0.2))
            
            # Log step-level metrics
            writer.add_scalar('Rewards/Step_Total_Reward', step_reward, global_step)
            writer.add_scalar('Rewards/Step_Tracking_Reward', step_reward * 0.7, global_step)
            writer.add_scalar('Metrics/Position_Error', position_error, global_step)
            writer.add_scalar('Actions/Action_Norm', action_norm, global_step)
            writer.add_scalar('Training/Loss', loss, global_step)
        
        # Log rolling statistics
        if episode >= 4:  # Need at least 5 episodes for rolling stats
            recent_episodes = list(range(max(0, episode-4), episode+1))
            recent_rewards = [base_reward + i*50 + np.random.normal(0, 50) for i in recent_episodes]
            
            writer.add_scalar('Statistics/Min_Reward_Last_5', min(recent_rewards), episode)
            writer.add_scalar('Statistics/Mean_Reward_Last_5', np.mean(recent_rewards), episode)
            writer.add_scalar('Statistics/Max_Reward_Last_5', max(recent_rewards), episode)
        
        # Flush data to ensure it's written
        writer.flush()
        
        # Small delay to make it more realistic
        time.sleep(0.1)
    
    # Close writer
    writer.close()
    print(f"✅ TensorBoard test completed!")
    print(f"📂 Log directory: {log_dir}")
    print(f"🌐 View results at: http://localhost:6006")
    print(f"💡 Run: tensorboard --logdir=tensorboard_logs --port=6006")

if __name__ == "__main__":
    test_tensorboard_logging()
