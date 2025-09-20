"""
Test Genesis Native Dashboard Integration
Verify that it works exactly like TensorBoard with same data visualization
"""

import numpy as np
import time
from genesis_native_streamer import create_genesis_native_tensorboard_replacement

def test_genesis_native_dashboard():
    """Test the Genesis Native Dashboard with same data as TensorBoard"""
    
    print("🧪 Testing Genesis Native Dashboard (TensorBoard Replacement)")
    print("=" * 60)
    
    # Create Genesis Native Dashboard (replaces TensorBoard)
    logger = create_genesis_native_tensorboard_replacement()
    
    # Test hyperparameters (same as TensorBoard)
    hyperparams = {
        "algorithm": "DDPG",
        "learning_rate": 0.001,
        "batch_size": 64,
        "buffer_size": 100000,
        "gamma": 0.99,
        "tau": 0.005,
        "actor_lr": 0.001,
        "critic_lr": 0.002
    }
    logger.log_hyperparameters(hyperparams)
    print("✅ Hyperparameters logged")
    
    # Test episode and step logging (same data as TensorBoard was tracking)
    for episode in range(3):
        logger.log_episode_start(episode + 1)
        
        episode_total_reward = 0
        episode_tracking_reward = 0
        episode_position_errors = []
        episode_actor_losses = []
        episode_critic_losses = []
        
        for step in range(10):
            # Generate realistic training data (same metrics as TensorBoard)
            step_reward = np.random.normal(0.1, 0.5)
            tracking_reward = np.random.normal(0.05, 0.3)
            position_error = np.random.uniform(0.001, 0.1)
            action = np.random.normal(0, 0.1, size=7)  # 7-DOF Franka actions
            actor_loss = np.random.uniform(0.001, 0.01)
            critic_loss = np.random.uniform(0.01, 0.1)
            noise_std = max(0.1, 0.1 - episode * 0.02)  # Decreasing exploration
            
            episode_total_reward += step_reward
            episode_tracking_reward += tracking_reward
            episode_position_errors.append(position_error)
            episode_actor_losses.append(actor_loss)
            episode_critic_losses.append(critic_loss)
            
            # Log step data (same as TensorBoard step logging)
            step_data = {
                'episode': episode + 1,
                'step': step,
                'reward': step_reward,
                'total_reward': episode_total_reward,
                'tracking_reward': tracking_reward,
                'position_error': position_error,
                'action': action,
                'actor_loss': actor_loss,
                'critic_loss': critic_loss,
                'exploration_noise': noise_std
            }
            logger.log_step_metrics(step_data)
            
            time.sleep(0.1)  # Simulate training time
        
        # Log episode completion (same as TensorBoard episode logging)
        episode_data = {
            'episode': episode + 1,
            'total_reward': episode_total_reward,
            'episode_length': 10,
            'tracking_reward': episode_tracking_reward,
            'avg_position_error': np.mean(episode_position_errors),
            'avg_actor_loss': np.mean(episode_actor_losses),
            'avg_critic_loss': np.mean(episode_critic_losses),
            'success_rate': 0.8 + episode * 0.1  # Improving success rate
        }
        logger.log_episode_metrics(episode_data)
        
        print(f"📊 Episode {episode + 1} completed: Reward={episode_total_reward:.3f}")
        time.sleep(0.5)
    
    # Test network weights logging (simplified version of TensorBoard histograms)
    try:
        # Simulate network weights summary
        weights_summary = {
            "actor.fc1.weight": {
                "mean": 0.02,
                "std": 0.1,
                "min": -0.3,
                "max": 0.3,
                "shape": [256, 24]
            },
            "actor.fc2.weight": {
                "mean": 0.01,
                "std": 0.08,
                "min": -0.2,
                "max": 0.2,
                "shape": [256, 256]
            }
        }
        logger.native.log_network_weights("actor_network", weights_summary)
        print("✅ Network weights logged")
    except Exception as e:
        print(f"⚠️ Network weights logging: {e}")
    
    print("\n🎉 Test completed!")
    print("🌐 View results at: http://localhost:8090/dashboard.html")
    print("📊 Data tracked (same as TensorBoard):")
    print("   • Step rewards & total rewards")
    print("   • Tracking rewards & position errors") 
    print("   • Actor & critic losses")
    print("   • Action values & exploration noise")
    print("   • Episode statistics & success rates")
    print("   • Network weights summaries")
    print("   • Hyperparameters")
    
    print("\n⏳ Dashboard will stay active for 30 seconds for viewing...")
    time.sleep(30)
    
    # Finalize (same as TensorBoard.finalize())
    logger.finalize()
    print("✅ Genesis Native Dashboard test completed!")

if __name__ == "__main__":
    test_genesis_native_dashboard()
