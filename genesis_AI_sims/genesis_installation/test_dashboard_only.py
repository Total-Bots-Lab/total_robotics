"""
Test Dashboard Functionality Independently
Tests the safe dashboard without Genesis to verify data streaming
"""

import time
import random
import numpy as np
from safe_dashboard import SafeTrainingDashboard

print("🧪 Testing Safe Dashboard Functionality")
print("📊 This test will verify live data streaming without Genesis")

# Initialize dashboard
dashboard = SafeTrainingDashboard(save_dir="test_dashboard_data", port=8081)
print("✅ Dashboard initialized")

# Start dashboard server
dashboard_started = dashboard.start_dashboard_server()
if dashboard_started:
    print("🌐 Live dashboard available at: http://localhost:8081/dashboard.html")
    print("🔄 Starting data simulation...")
else:
    print("⚠️ Dashboard server failed to start")
    exit(1)

# Simulate training data for 2 episodes
num_episodes = 2
steps_per_episode = 20

try:
    for episode in range(1, num_episodes + 1):
        print(f"\n📊 Simulating Episode {episode}/{num_episodes}")
        
        episode_rewards = []
        episode_tracking_rewards = []
        
        for step in range(steps_per_episode):
            # Generate random but realistic rewards
            total_reward = random.uniform(-10, 10)
            tracking_reward = random.uniform(-5, 15)
            position_error = random.uniform(0, 1)
            
            # Log step data
            dashboard.log_step_data(episode, step, total_reward, tracking_reward, position_error)
            
            episode_rewards.append(total_reward)
            episode_tracking_rewards.append(tracking_reward)
            
            print(f"  Step {step}: reward={total_reward:.2f}, tracking={tracking_reward:.2f}")
            
            # Small delay to simulate real training
            time.sleep(0.1)
        
        # Log episode completion
        dashboard.log_episode_complete(episode, episode_rewards, episode_tracking_rewards)
        print(f"✅ Episode {episode} completed - check dashboard for live updates!")
        
        time.sleep(1)  # Pause between episodes

    print("\n🎉 Dashboard test completed successfully!")
    print("🌐 Dashboard should show live data at: http://localhost:8081/dashboard.html")
    print("📊 Check the dashboard to verify:")
    print("   - Step-wise reward data")
    print("   - Episode summaries")
    print("   - Min/Max/Mean statistics")
    print("   - Live plots and charts")
    
    # Finalize dashboard
    final_stats = dashboard.finalize_dashboard()
    print(f"\n📈 Final dashboard stats: {final_stats}")
    
    # Keep server running for viewing
    print("\n⏳ Server will run for 30 seconds for testing...")
    print("   Visit http://localhost:8081/dashboard.html to view results")
    time.sleep(30)
    
except KeyboardInterrupt:
    print("\n⚠️ Test interrupted by user")
    
finally:
    # Stop server
    dashboard.stop_dashboard_server()
    print("📊 Dashboard server stopped")
    print("✅ Test completed")
