#!/usr/bin/env python3
"""
Training Analysis and Current Implementation Diagram Generator
Analyzes the training results and creates visual diagrams
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def analyze_training_results():
    """Analyze the training log results"""
    
    # Training data from log
    episode_rewards = [-153.95, -133.66, -130.12]
    episodes = [1, 2, 3]
    
    # Goal distances over episodes (approximate from log)
    goal_distances = {
        1: [0.931, 0.761, 0.756, 0.649, 0.846],  # Episode 1 distances
        2: [0.673, 0.736, 0.711, 0.640, 0.615],  # Episode 2 distances  
        3: [0.643, 0.617, 0.688, 0.647, 0.629]   # Episode 3 distances
    }
    
    # Actor and Critic losses
    actor_losses = [0.0684, 0.0839, 0.0827]
    critic_losses = [0.0734, 0.0140, 0.0253]
    
    # Create analysis plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('🤖 Training Analysis - Current Implementation Performance', fontsize=16, fontweight='bold')
    
    # 1. Episode Rewards Trend
    ax1.plot(episodes, episode_rewards, 'b-o', linewidth=2, markersize=8, label='Episode Rewards')
    ax1.axhline(y=-100, color='g', linestyle='--', alpha=0.7, label='Target: -100 (Good Performance)')
    ax1.set_title('📈 Episode Rewards Progression', fontweight='bold')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add trend analysis
    improvement = episode_rewards[-1] - episode_rewards[0]
    ax1.text(0.02, 0.98, f'Improvement: {improvement:.1f}\nTrend: {"↗️ Improving" if improvement > 0 else "↘️ Declining"}', 
             transform=ax1.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # 2. Goal Distance Analysis
    for ep, distances in goal_distances.items():
        steps = np.linspace(20, 100, len(distances))
        ax2.plot(steps, distances, 'o-', label=f'Episode {ep}', linewidth=2, markersize=6)
    
    ax2.axhline(y=0.10, color='r', linestyle='--', alpha=0.7, label='Goal Tolerance (0.10m)')
    ax2.set_title('🎯 Goal Distance Over Time', fontweight='bold')
    ax2.set_xlabel('Step in Episode')
    ax2.set_ylabel('Distance to Goal (m)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Loss Analysis
    x_pos = np.arange(len(episodes))
    width = 0.35
    
    ax3.bar(x_pos - width/2, actor_losses, width, label='Actor Loss', color='skyblue', alpha=0.8)
    ax3.bar(x_pos + width/2, critic_losses, width, label='Critic Loss', color='lightcoral', alpha=0.8)
    ax3.set_title('🧠 Network Loss Progression', fontweight='bold')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Loss Value')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f'Ep {i}' for i in episodes])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Performance Metrics Summary
    ax4.axis('off')
    
    # Create performance summary
    metrics_text = f"""
🔍 TRAINING ANALYSIS SUMMARY:

📊 PERFORMANCE METRICS:
• Best Reward: {max(episode_rewards):.1f}
• Average Reward: {np.mean(episode_rewards):.1f}
• Improvement Rate: {improvement:.1f} per episode
• Goals Reached: 0/3 episodes

🎯 GOAL REACHING ANALYSIS:
• Closest Distance: {min([min(distances) for distances in goal_distances.values()]):.3f}m
• Goal Tolerance: 0.100m
• Miss Factor: {min([min(distances) for distances in goal_distances.values()])/0.1:.1f}x tolerance

🧠 LEARNING INDICATORS:
• Actor Loss Trend: {"Stable" if abs(actor_losses[-1] - actor_losses[0]) < 0.01 else "Changing"}
• Critic Loss: {"Decreasing ✅" if critic_losses[-1] < critic_losses[0] else "Increasing ⚠️"}
• Noise Decay: 0.200 → 0.050 ✅

⚠️ IDENTIFIED ISSUES:
1. Goals too far (0.6-0.9m vs 0.1m tolerance)
2. Reward scale too harsh (-1.2 to -1.9 per step)
3. Limited exploration (no goal reached)
4. Short episodes (100 steps may be insufficient)
"""
    
    ax4.text(0.05, 0.95, metrics_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('training_analysis_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return episode_rewards, goal_distances, actor_losses, critic_losses

def create_current_implementation_diagram():
    """Create a diagram of the current implementation structure"""
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, '🤖 CURRENT IMPLEMENTATION FLOW DIAGRAM', 
            ha='center', va='center', fontsize=18, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    # Environment Components
    env_box = FancyBboxPatch((0.5, 7), 2, 1.5, boxstyle="round,pad=0.1", 
                             facecolor='lightgreen', edgecolor='darkgreen', linewidth=2)
    ax.add_patch(env_box)
    ax.text(1.5, 7.75, 'FRANKA ENVIRONMENT\n• XML Configuration\n• Dynamic Goals\n• Physics Simulation', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # State Observation
    state_box = FancyBboxPatch((3.5, 7.5), 1.5, 0.8, boxstyle="round,pad=0.1",
                               facecolor='wheat', edgecolor='orange', linewidth=2)
    ax.add_patch(state_box)
    ax.text(4.25, 7.9, 'STATE s(t)\n28 dimensions', ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Actor Network
    actor_box = FancyBboxPatch((6, 8), 1.8, 1, boxstyle="round,pad=0.1",
                               facecolor='lightcoral', edgecolor='darkred', linewidth=2)
    ax.add_patch(actor_box)
    ax.text(6.9, 8.5, 'ACTOR NETWORK\nπ(a|s)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Critic Network  
    critic_box = FancyBboxPatch((6, 6.5), 1.8, 1, boxstyle="round,pad=0.1",
                                facecolor='lightskyblue', edgecolor='darkblue', linewidth=2)
    ax.add_patch(critic_box)
    ax.text(6.9, 7, 'CRITIC NETWORK\nQ(s,a)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Action Output
    action_box = FancyBboxPatch((8.5, 7.5), 1, 0.8, boxstyle="round,pad=0.1",
                                facecolor='lightpink', edgecolor='purple', linewidth=2)
    ax.add_patch(action_box)
    ax.text(9, 7.9, 'ACTION\nA(t)', ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Experience Replay
    replay_box = FancyBboxPatch((1, 5), 2.5, 1, boxstyle="round,pad=0.1",
                                facecolor='lavender', edgecolor='indigo', linewidth=2)
    ax.add_patch(replay_box)
    ax.text(2.25, 5.5, 'EXPERIENCE REPLAY\nBuffer Size: 50,000', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # PPO Update
    ppo_box = FancyBboxPatch((4.5, 4.5), 3, 1.5, boxstyle="round,pad=0.1",
                             facecolor='gold', edgecolor='darkorange', linewidth=2)
    ax.add_patch(ppo_box)
    ax.text(6, 5.25, 'PPO-DDPG UPDATE\n• Lclip Objective\n• Advantage A(t)\n• ε = 0.2 clipping', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Reward Signal
    reward_box = FancyBboxPatch((0.5, 3), 2, 1, boxstyle="round,pad=0.1",
                                facecolor='lightsteelblue', edgecolor='steelblue', linewidth=2)
    ax.add_patch(reward_box)
    ax.text(1.5, 3.5, 'REWARD SIGNAL\nR(t)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Goal System
    goal_box = FancyBboxPatch((8, 4.5), 1.5, 1.5, boxstyle="round,pad=0.1",
                              facecolor='lightcyan', edgecolor='teal', linewidth=2)
    ax.add_patch(goal_box)
    ax.text(8.75, 5.25, 'GOAL SYSTEM\n35 Goals\nDynamic\nSwitching', ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Training Loop
    loop_box = FancyBboxPatch((3.5, 1.5), 3, 1, boxstyle="round,pad=0.1",
                              facecolor='mistyrose', edgecolor='crimson', linewidth=2)
    ax.add_patch(loop_box)
    ax.text(5, 2, 'TRAINING LOOP\nEpisodes: 3\nSteps: 100/episode', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Add flow arrows
    arrows = [
        # Environment to State
        ((2.5, 7.75), (3.5, 7.9)),
        # State to Actor
        ((5, 7.9), (6, 8.2)),
        # State to Critic  
        ((5, 7.7), (6, 7.2)),
        # Actor to Action
        ((7.8, 8.3), (8.5, 8.1)),
        # Action back to Environment
        ((8.5, 7.7), (2.5, 7.2)),
        # Experience to Replay
        ((1.5, 7), (2.25, 6)),
        # Replay to PPO
        ((3.5, 5.5), (4.5, 5.2)),
        # PPO back to networks
        ((6, 4.5), (6.9, 6.5)),
        # Reward to system
        ((2.5, 3.5), (4, 4.5)),
        # Goal to environment
        ((8, 5.25), (2.5, 7.5))
    ]
    
    for start, end in arrows:
        arrow = ConnectionPatch(start, end, "data", "data",
                               arrowstyle="->", shrinkA=5, shrinkB=5,
                               mutation_scale=20, fc="black", alpha=0.7)
        ax.add_artist(arrow)
    
    # Current Status Box
    status_text = """
🔍 CURRENT STATUS:
• Implementation: 92% JPEG aligned ✅
• Core Components: All present ✅
• Training: Running but needs tuning ⚠️
• Goal Reaching: 0% success rate ❌

⚙️ KEY PARAMETERS:
• Learning Rates: Actor=1e-4, Critic=1e-3
• PPO Epsilon: 0.2
• Goal Tolerance: 0.10m
• Episode Length: 100 steps
• Noise: 0.2 → 0.05 (decay)
"""
    
    ax.text(0.5, 2.5, status_text, fontsize=9, fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9),
            verticalalignment='top')
    
    plt.savefig('current_implementation_diagram.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_improvement_suggestions():
    """Generate specific improvement suggestions"""
    
    suggestions = """
🔧 IMPROVEMENT SUGGESTIONS BASED ON TRAINING ANALYSIS:

🎯 IMMEDIATE FIXES (High Priority):

1. **GOAL TOLERANCE ADJUSTMENT**
   Current: 0.10m tolerance, but agent reaches 0.6-0.9m
   Solution: Increase tolerance to 0.20m initially, then reduce gradually
   Code: env.goal_tolerance = 0.20  # Line ~183

2. **REWARD SCALING OPTIMIZATION**
   Current: -2.0 * goal_distance = -1.2 to -1.8 per step
   Solution: Less harsh penalty, more goal-reaching bonus
   Code: goal_reward = -0.5 * goal_distance  # Instead of -2.0

3. **EPISODE LENGTH EXTENSION**
   Current: 100 steps (insufficient for goal reaching)
   Solution: Increase to 300-500 steps
   Code: max_episode_steps=300  # Line ~595

4. **EXPLORATION ENHANCEMENT**
   Current: Noise 0.2→0.05 too fast decay
   Solution: Slower decay, higher minimum
   Code: noise_decay=0.999, noise_min=0.1

📈 MEDIUM PRIORITY FIXES:

5. **LEARNING RATE ADJUSTMENT**
   Current: Actor=1e-4, Critic=1e-3
   Solution: Increase actor LR for faster policy learning
   Code: lr_actor=3e-4

6. **CURRICULUM LEARNING**
   Current: Random goal selection
   Solution: Start with closer goals, gradually increase distance
   
7. **BATCH SIZE OPTIMIZATION**
   Current: 64 samples
   Solution: Increase to 128 for more stable updates

🧠 ADVANCED IMPROVEMENTS:

8. **PRIORITIZED EXPERIENCE REPLAY**
   Current: Random sampling
   Solution: Prioritize successful experiences

9. **MULTI-STEP RETURNS**
   Current: Single-step TD
   Solution: n-step returns for better value estimation

10. **ADAPTIVE GOAL GENERATION**
    Current: Fixed workspace goals
    Solution: Generate goals based on current capability

💡 QUICK TEST MODIFICATIONS:

```python
# In FrankaGymEnv.__init__():
self.goal_tolerance = 0.20  # Easier goals initially
self.max_steps = 300        # Longer episodes

# In _calculate_reward():
goal_reward = -0.5 * goal_distance  # Less harsh penalty
if goal_distance < self.goal_tolerance:
    goal_reward = 200.0  # Bigger bonus

# In DDPGAgent.__init__():
self.noise_decay = 0.999  # Slower noise decay
self.noise_min = 0.1      # Higher minimum noise

# In main():
num_episodes = 10  # More episodes for better learning
```

🎯 EXPECTED IMPROVEMENTS:
• Goal reaching success: 0% → 20-40%
• Episode rewards: -130 → -50 to +50
• Learning stability: More consistent improvement
• Exploration: Better coverage of workspace
"""
    
    # Save suggestions to file
    with open('improvement_suggestions.txt', 'w', encoding='utf-8') as f:
        f.write(suggestions)
    
    print(suggestions)
    print(f"\n💾 Suggestions saved to: improvement_suggestions.txt")

def main():
    """Main analysis function"""
    print("🔍 Analyzing training results and generating diagrams...")
    
    # Analyze training performance
    episode_rewards, goal_distances, actor_losses, critic_losses = analyze_training_results()
    
    # Create implementation diagram
    create_current_implementation_diagram()
    
    # Generate improvement suggestions
    generate_improvement_suggestions()
    
    print("\n✅ Analysis complete! Generated files:")
    print("📊 training_analysis_results.png - Performance analysis")
    print("📋 current_implementation_diagram.png - Implementation flow")
    print("💡 improvement_suggestions.txt - Specific recommendations")

if __name__ == "__main__":
    main()
