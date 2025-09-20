"""
REINFORCEMENT LEARNING FLOW DIAGRAM GENERATOR
Professional visualization system for documenting RL implementation architecture
Industry-standard documentation with text-only labels for maximum compatibility
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle
import numpy as np

def create_rl_training_flow():
    """Create comprehensive RL training flow diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Colors for different system components
    init_color = '#E3F2FD'
    env_color = '#F3E5F5'
    loop_color = '#FFF3E0'
    agent_color = '#E8F5E8'
    viz_color = '#FFF8E1'
    
    # Title
    ax.text(5, 11.5, 'REINFORCEMENT LEARNING TRAINING FLOW', 
            fontsize=18, fontweight='bold', ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8))
    
    # 1. Initialization
    init_box = FancyBboxPatch((0.5, 10), 3, 1.5, 
                              boxstyle="round,pad=0.1", 
                              facecolor=init_color, edgecolor='blue', linewidth=2)
    ax.add_patch(init_box)
    ax.text(2, 10.6, 'INITIALIZATION', fontweight='bold', ha='center', fontsize=12)
    ax.text(2, 10.3, '• Genesis Environment Setup', ha='center', fontsize=10)
    ax.text(2, 10.1, '• Neural Network Creation', ha='center', fontsize=10)
    ax.text(2, 9.9, '• Hyperparameter Config', ha='center', fontsize=10)
    ax.text(2, 9.7, '• Trajectory System Init', ha='center', fontsize=10)
    
    # 2. Environment
    env_box = FancyBboxPatch((4.5, 10), 3, 1.5, 
                             boxstyle="round,pad=0.1", 
                             facecolor=env_color, edgecolor='purple', linewidth=2)
    ax.add_patch(env_box)
    ax.text(6, 10.6, 'ENVIRONMENT', fontweight='bold', ha='center', fontsize=12)
    ax.text(6, 10.3, '• Franka Robot Arm', ha='center', fontsize=10)
    ax.text(6, 10.1, '• Physics Simulation', ha='center', fontsize=10)
    ax.text(6, 9.9, '• State Observation', ha='center', fontsize=10)
    ax.text(6, 9.7, '• Reward Calculation', ha='center', fontsize=10)
    
    # 3. Training Loop
    loop_box = FancyBboxPatch((2, 8), 4, 1.5, 
                              boxstyle="round,pad=0.1", 
                              facecolor=loop_color, edgecolor='orange', linewidth=2)
    ax.add_patch(loop_box)
    ax.text(4, 8.6, 'TRAINING LOOP', fontweight='bold', ha='center', fontsize=12)
    ax.text(4, 8.3, '• Experience Collection', ha='center', fontsize=10)
    ax.text(4, 8.1, '• Policy Learning', ha='center', fontsize=10)
    ax.text(4, 7.9, '• Value Function Update', ha='center', fontsize=10)
    ax.text(4, 7.7, '• Performance Evaluation', ha='center', fontsize=10)
    
    # 4. DDPG Agent
    agent_box = FancyBboxPatch((0.5, 5.5), 3, 1.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor=agent_color, edgecolor='green', linewidth=2)
    ax.add_patch(agent_box)
    ax.text(2, 6.6, 'DDPG AGENT', fontweight='bold', ha='center', fontsize=12)
    ax.text(2, 6.3, '• Actor Network', ha='center', fontsize=10)
    ax.text(2, 6.1, '• Critic Network', ha='center', fontsize=10)
    ax.text(2, 5.9, '• Target Networks', ha='center', fontsize=10)
    ax.text(2, 5.7, '• Exploration Noise', ha='center', fontsize=10)
    
    # 5. Visualization System
    viz_box = FancyBboxPatch((4.5, 5.5), 3, 1.5, 
                             boxstyle="round,pad=0.1", 
                             facecolor=viz_color, edgecolor='red', linewidth=2)
    ax.add_patch(viz_box)
    ax.text(6, 6.6, 'VISUALIZATION', fontweight='bold', ha='center', fontsize=12)
    ax.text(6, 6.3, '• Real-time Trajectory', ha='center', fontsize=10)
    ax.text(6, 6.1, '• Goal Markers', ha='center', fontsize=10)
    ax.text(6, 5.9, '• Performance Metrics', ha='center', fontsize=10)
    ax.text(6, 5.7, '• 3D Scene Rendering', ha='center', fontsize=10)
    
    # 6. Episode Flow
    episode_boxes = []
    for i, (label, y_pos) in enumerate([('Reset', 4), ('Step', 3), ('Update', 2), ('Analyze', 1)]):
        box = FancyBboxPatch((1 + i*2, y_pos-0.3), 1.5, 0.6, 
                             boxstyle="round,pad=0.05", 
                             facecolor='lightblue', edgecolor='navy', linewidth=1)
        ax.add_patch(box)
        ax.text(1.75 + i*2, y_pos, label, ha='center', va='center', fontweight='bold')
        episode_boxes.append((1.75 + i*2, y_pos))
    
    # Add arrows for main flow
    arrows = [
        # Initialization to Environment
        ((3.5, 10.2), (4.5, 10.2)),
        # Environment to Training Loop
        ((5, 10), (4.5, 8.8)),
        # Training Loop to Agent
        ((3, 8), (2.5, 7)),
        # Training Loop to Visualization
        ((5, 8), (5.5, 7)),
        # Agent to Episode Flow
        ((2, 5.5), (1.75, 4.3)),
        # Visualization to Episode Flow
        ((6, 5.5), (7.75, 4.3)),
    ]
    
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Add episode flow arrows
    for i in range(len(episode_boxes) - 1):
        start = (episode_boxes[i][0] + 0.75, episode_boxes[i][1])
        end = (episode_boxes[i+1][0] - 0.75, episode_boxes[i+1][1])
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='blue'))
    
    plt.title('Professional RL Training System Architecture', pad=20, fontsize=16)
    plt.tight_layout()
    plt.savefig('rl_training_flow.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_episode_lifecycle():
    """Create detailed episode lifecycle flow diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'EPISODE LIFECYCLE FLOW', 
            fontsize=16, fontweight='bold', ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.8))
    
    # Episode states
    states = [
        ('RESET', 8.5, 'Environment reset\nTrajectory initialization\nGoal selection'),
        ('OBSERVE', 7.5, 'Get current state\nJoint positions\nEnd-effector pose'),
        ('ACT', 6.5, 'Agent action selection\nNoise exploration\nAction execution'),
        ('STEP', 5.5, 'Environment step\nPhysics simulation\nState transition'),
        ('REWARD', 4.5, 'Reward calculation\nGoal proximity\nCollision penalty'),
        ('LEARN', 3.5, 'Experience storage\nBatch sampling\nNetwork updates'),
        ('ANALYZE', 2.5, 'Episode metrics\nTrajectory analysis\nSuccess evaluation'),
    ]
    
    # Create state boxes
    state_positions = []
    colors = ['#FFE0E0', '#E0F0FF', '#E0FFE0', '#FFF0E0', '#F0E0FF', '#E0FFF0', '#FFE0F0']
    
    for i, (state, y_pos, description) in enumerate(states):
        # Main state box
        box = FancyBboxPatch((2, y_pos-0.4), 2.5, 0.8, 
                             boxstyle="round,pad=0.1", 
                             facecolor=colors[i], edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(3.25, y_pos, state, ha='center', va='center', fontweight='bold', fontsize=12)
        
        # Description box
        desc_box = FancyBboxPatch((5, y_pos-0.3), 3.5, 0.6, 
                                  boxstyle="round,pad=0.05", 
                                  facecolor='white', edgecolor='gray', linewidth=1)
        ax.add_patch(desc_box)
        ax.text(6.75, y_pos, description, ha='center', va='center', fontsize=9)
        
        state_positions.append((3.25, y_pos))
    
    # Add flow arrows
    for i in range(len(state_positions) - 1):
        start = (state_positions[i][0], state_positions[i][1] - 0.4)
        end = (state_positions[i+1][0], state_positions[i+1][1] + 0.4)
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    
    # Add cycle arrow from ANALYZE back to RESET
    ax.annotate('', xy=(2, 8.1), xytext=(2, 2.9),
               arrowprops=dict(arrowstyle='->', lw=2, color='red',
                             connectionstyle="arc3,rad=-0.3"))
    
    # Add "Next Episode" label
    ax.text(0.5, 5.5, 'Next\nEpisode', ha='center', va='center', 
            fontweight='bold', fontsize=10, color='red')
    
    plt.title('Episode Management and Control Flow', pad=20, fontsize=14)
    plt.tight_layout()
    plt.savefig('episode_lifecycle.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_network_update_flow():
    """Create detailed network update flow diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(5, 7.5, 'NEURAL NETWORK UPDATE FLOW', 
            fontsize=16, fontweight='bold', ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcyan', alpha=0.8))
    
    # Actor Update Section
    actor_box = FancyBboxPatch((0.5, 6), 4, 1.2, 
                               boxstyle="round,pad=0.1", 
                               facecolor='#FFE0E0', edgecolor='red', linewidth=2)
    ax.add_patch(actor_box)
    ax.text(2.5, 6.8, 'ACTOR UPDATE', fontweight='bold', ha='center', fontsize=12)
    ax.text(2.5, 6.5, '1. Policy Gradient Calculation', ha='center', fontsize=10)
    ax.text(2.5, 6.3, '2. Actor Loss Computation', ha='center', fontsize=10)
    ax.text(2.5, 6.1, '3. Gradient Ascent Update', ha='center', fontsize=10)
    
    # Critic Update Section
    critic_box = FancyBboxPatch((0.5, 4.5), 4, 1.2, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#E0E0FF', edgecolor='blue', linewidth=2)
    ax.add_patch(critic_box)
    ax.text(2.5, 5.6, 'CRITIC UPDATE', fontweight='bold', ha='center', fontsize=12)
    ax.text(2.5, 5.3, '1. Q-value Prediction', ha='center', fontsize=10)
    ax.text(2.5, 5.1, '2. TD Error Calculation', ha='center', fontsize=10)
    ax.text(2.5, 4.9, '3. MSE Loss Minimization', ha='center', fontsize=10)
    ax.text(2.5, 4.7, '4. Critic Weight Update', ha='center', fontsize=10)
    
    # Target Network Update
    target_box = FancyBboxPatch((5.5, 3), 4, 0.8, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#E0FFE0', edgecolor='green', linewidth=2)
    ax.add_patch(target_box)
    ax.text(7.5, 3.6, 'TARGET NETWORK SOFT UPDATE', fontweight='bold', ha='center', fontsize=11)
    ax.text(7.5, 3.3, 'τ * main_weights + (1-τ) * target_weights', ha='center', fontsize=9)
    ax.text(7.5, 3.1, 'Stabilizes training process', ha='center', fontsize=9)
    
    # Performance Tracking
    perf_box = FancyBboxPatch((2, 1), 6, 0.8, 
                              boxstyle="round,pad=0.1", 
                              facecolor='#FFFAE0', edgecolor='orange', linewidth=2)
    ax.add_patch(perf_box)
    ax.text(5, 1.6, 'PERFORMANCE TRACKING', fontweight='bold', ha='center', fontsize=11)
    ax.text(5, 1.3, 'Loss logging • Gradient norms • Learning rates • Convergence metrics', ha='center', fontsize=9)
    
    # Add arrows
    arrows = [
        # Actor to Critic
        ((2.5, 6), (2.5, 5.7)),
        # Critic to Target
        ((4.5, 5.1), (5.5, 3.8)),
        # Target to Performance
        ((7.5, 3), (6, 1.8)),
        # Actor to Performance
        ((2.5, 6), (3.5, 1.8)),
    ]
    
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Add update frequency info
    ax.text(8.5, 6.5, 'Every Step', fontweight='bold', fontsize=10, color='red')
    ax.text(8.5, 5.1, 'Every Step', fontweight='bold', fontsize=10, color='blue')
    ax.text(8.5, 3.4, 'Soft Update\nτ = 0.005', fontweight='bold', fontsize=9, color='green')
    
    plt.title('Deep Deterministic Policy Gradient Updates', pad=20, fontsize=14)
    plt.tight_layout()
    plt.savefig('network_update_flow.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_trajectory_visualization_flow():
    """Create trajectory visualization system flow diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(6, 9.5, 'INDUSTRY-STANDARD TRAJECTORY VISUALIZATION', 
            fontsize=16, fontweight='bold', ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.8))
    
    # 1. Initialization Phase
    init_box = FancyBboxPatch((0.5, 8.2), 3, 1.2, 
                              boxstyle="round,pad=0.1", 
                              facecolor='#E3F2FD', edgecolor='blue', linewidth=2)
    ax.add_patch(init_box)
    ax.text(2, 8.8, 'INITIALIZATION', fontweight='bold', ha='center', fontsize=11)
    ax.text(2, 8.5, '• 500 Marker Pool Pre-allocation', ha='center', fontsize=9)
    ax.text(2, 8.3, '• Trajectory Point Buffer Setup', ha='center', fontsize=9)
    
    # 2. Real-time Tracking
    track_box = FancyBboxPatch((4.5, 8.2), 3, 1.2, 
                               boxstyle="round,pad=0.1", 
                               facecolor='#F3E5F5', edgecolor='purple', linewidth=2)
    ax.add_patch(track_box)
    ax.text(6, 8.8, 'REAL-TIME TRACKING', fontweight='bold', ha='center', fontsize=11)
    ax.text(6, 8.5, '• End-effector Position Capture', ha='center', fontsize=9)
    ax.text(6, 8.3, '• Every 10 Steps Sampling', ha='center', fontsize=9)
    
    # 3. Analysis Phase
    analysis_box = FancyBboxPatch((8.5, 8.2), 3, 1.2, 
                                  boxstyle="round,pad=0.1", 
                                  facecolor='#FFF3E0', edgecolor='orange', linewidth=2)
    ax.add_patch(analysis_box)
    ax.text(10, 8.8, 'ANALYSIS', fontweight='bold', ha='center', fontsize=11)
    ax.text(10, 8.5, '• Efficiency Scoring', ha='center', fontsize=9)
    ax.text(10, 8.3, '• Distance Metrics', ha='center', fontsize=9)
    
    # 4. Goal Visualization
    goal_box = FancyBboxPatch((1, 6.5), 4, 1, 
                              boxstyle="round,pad=0.1", 
                              facecolor='#E8F5E8', edgecolor='green', linewidth=2)
    ax.add_patch(goal_box)
    ax.text(3, 7, 'GOAL VISUALIZATION', fontweight='bold', ha='center', fontsize=11)
    ax.text(3, 6.7, '• Target Position Markers • Success Indicators • Progress Tracking', ha='center', fontsize=9)
    
    # 5. Path Rendering
    path_box = FancyBboxPatch((7, 6.5), 4, 1, 
                              boxstyle="round,pad=0.1", 
                              facecolor='#FFF8E1', edgecolor='gold', linewidth=2)
    ax.add_patch(path_box)
    ax.text(9, 7, 'PATH RENDERING', fontweight='bold', ha='center', fontsize=11)
    ax.text(9, 6.7, '• Color-coded Performance • Smoothed Trajectories • Velocity Indicators', ha='center', fontsize=9)
    
    # 6. Episode Management
    episode_box = FancyBboxPatch((1, 4), 4, 1, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor='#F0F4FF', edgecolor='indigo', linewidth=2)
    ax.add_patch(episode_box)
    ax.text(3, 4.5, 'EPISODE MANAGEMENT', fontweight='bold', ha='center', fontsize=11)
    ax.text(3, 4.2, '• Trajectory Reset • Marker Pool Recycling • Performance Logging', ha='center', fontsize=9)
    
    # 7. Professional Analytics
    analytics_box = FancyBboxPatch((7, 4), 4, 1, 
                                   boxstyle="round,pad=0.1", 
                                   facecolor='#FFF0F5', edgecolor='crimson', linewidth=2)
    ax.add_patch(analytics_box)
    ax.text(9, 4.5, 'PROFESSIONAL ANALYTICS', fontweight='bold', ha='center', fontsize=11)
    ax.text(9, 4.2, '• Multi-episode Comparison • Statistical Analysis • Export Reports', ha='center', fontsize=9)
    
    # 8. Final Report
    report_box = FancyBboxPatch((3.5, 1.8), 5, 0.8, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#F5F5F5', edgecolor='black', linewidth=2)
    ax.add_patch(report_box)
    ax.text(6, 2.2, 'COMPREHENSIVE ANALYSIS REPORT', fontweight='bold', ha='center', fontsize=11)
    ax.text(6, 1.9, 'Performance Metrics • Learning Progress • Optimization Insights', ha='center', fontsize=9)
    
    # Add flow arrows
    arrows = [
        # Horizontal flow
        ((3.5, 8.8), (4.5, 8.8)),
        ((7.5, 8.8), (8.5, 8.8)),
        # Vertical connections
        ((2, 8.2), (2.5, 7.5)),
        ((6, 8.2), (8.5, 7.5)),
        ((10, 8.2), (9.5, 7.5)),
        # To episode management
        ((3, 6.5), (3, 5)),
        ((9, 6.5), (9, 5)),
        # To final report
        ((3, 4), (4.5, 2.6)),
        ((9, 4), (7.5, 2.6)),
    ]
    
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color='darkblue'))
    
    # Add performance indicators
    ax.text(0.5, 3, 'GENESIS 0.3.1\nCOMPATIBLE', fontweight='bold', ha='center', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.8))
    
    ax.text(11.5, 3, 'INDUSTRY\nSTANDARD', fontweight='bold', ha='center', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='gold', alpha=0.8))
    
    plt.title('Professional Trajectory Visualization System Architecture', pad=20, fontsize=14)
    plt.tight_layout()
    plt.savefig('trajectory_visualization_flow.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_curriculum_learning_flow():
    """Create curriculum learning progression diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(6, 7.5, 'CURRICULUM LEARNING PROGRESSION', 
            fontsize=16, fontweight='bold', ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightsteelblue', alpha=0.8))
    
    # Difficulty levels
    levels = [
        ('BASIC', 0.5, 'Simple reach tasks\nLarge tolerance\nStatic targets'),
        ('INTERMEDIATE', 3, 'Precision tasks\nReduced tolerance\nSlow moving targets'),
        ('ADVANCED', 5.5, 'Complex trajectories\nTight tolerance\nDynamic obstacles'),
        ('EXPERT', 8, 'Multi-object tasks\nMinimal tolerance\nReal-time adaptation'),
        ('MASTERY', 10.5, 'Production tasks\nZero tolerance\nFull complexity')
    ]
    
    colors = ['#E8F5E8', '#FFFACD', '#FFE4B5', '#DDA0DD', '#B0C4DE']
    
    # Create level boxes
    level_positions = []
    for i, (level, x_pos, description) in enumerate(levels):
        # Main level box
        box = FancyBboxPatch((x_pos, 5), 2, 1.5, 
                             boxstyle="round,pad=0.1", 
                             facecolor=colors[i], edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(x_pos + 1, 5.9, level, ha='center', va='center', fontweight='bold', fontsize=11)
        ax.text(x_pos + 1, 5.4, description, ha='center', va='center', fontsize=8)
        
        # Progress indicator
        progress = (i + 1) * 20
        ax.text(x_pos + 1, 4.8, f'{progress}%', ha='center', va='center', 
                fontweight='bold', fontsize=10, color='blue')
        
        level_positions.append((x_pos + 1, 5.75))
    
    # Add progression arrows
    for i in range(len(level_positions) - 1):
        start = (level_positions[i][0] + 1, level_positions[i][1])
        end = (level_positions[i+1][0] - 1, level_positions[i+1][1])
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=3, color='green'))
    
    # Performance metrics
    metrics_box = FancyBboxPatch((2, 3), 8, 1, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor='#F0F8FF', edgecolor='navy', linewidth=2)
    ax.add_patch(metrics_box)
    ax.text(6, 3.7, 'ADAPTIVE CURRICULUM METRICS', fontweight='bold', ha='center', fontsize=12)
    ax.text(6, 3.3, 'Success Rate • Learning Speed • Task Complexity • Error Reduction • Skill Transfer', 
            ha='center', fontsize=10)
    
    # Success criteria
    criteria_box = FancyBboxPatch((2, 1.5), 8, 1, 
                                  boxstyle="round,pad=0.1", 
                                  facecolor='#F5FFFA', edgecolor='darkgreen', linewidth=2)
    ax.add_patch(criteria_box)
    ax.text(6, 2.2, 'ADVANCEMENT CRITERIA', fontweight='bold', ha='center', fontsize=12)
    ax.text(6, 1.8, '85% Success Rate • Consistent Performance • Reduced Training Time • Skill Generalization', 
            ha='center', fontsize=10)
    
    # Add advancement connections
    for i, (x_pos, _) in enumerate(level_positions):
        ax.annotate('', xy=(6, 3), xytext=(x_pos, 5),
                   arrowprops=dict(arrowstyle='->', lw=1, color='blue', alpha=0.5))
    
    # Add timeline
    ax.text(6, 0.8, 'TRAINING TIMELINE: 0 → 1000 → 2500 → 5000 → 10000+ Episodes', 
            ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.8))
    
    plt.title('Intelligent Curriculum Learning System', pad=20, fontsize=14)
    plt.tight_layout()
    plt.savefig('curriculum_learning_flow.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_all_diagrams():
    """Generate all professional flow diagrams"""
    print("GENERATING REINFORCEMENT LEARNING FLOW DIAGRAMS")
    print("=" * 60)
    
    diagrams = [
        ("RL Training Flow", create_rl_training_flow),
        ("Episode Lifecycle", create_episode_lifecycle),
        ("Network Update Flow", create_network_update_flow),
        ("Trajectory Visualization", create_trajectory_visualization_flow),
        ("Curriculum Learning", create_curriculum_learning_flow),
    ]
    
    for name, func in diagrams:
        print(f"Creating {name} diagram...")
        try:
            func()
            print(f"✓ {name} diagram saved successfully")
        except Exception as e:
            print(f"✗ Error creating {name} diagram: {e}")
    
    print("\n" + "=" * 60)
    print("DIAGRAM GENERATION COMPLETE")
    print("Generated files:")
    print("• rl_training_flow.png")
    print("• episode_lifecycle.png") 
    print("• network_update_flow.png")
    print("• trajectory_visualization_flow.png")
    print("• curriculum_learning_flow.png")
    print("\nAll diagrams use text-only labels for maximum compatibility")

if __name__ == "__main__":
    # Set matplotlib backend for compatibility
    plt.ioff()  # Turn off interactive mode
    
    # Generate all flow diagrams
    generate_all_diagrams()
