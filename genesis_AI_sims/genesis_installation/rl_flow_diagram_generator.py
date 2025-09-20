#!/usr/bin/env python3
"""
🏭 REINFORCEMENT LEARNING FLOW DIAGRAM GENERATOR
=====================================================

This script generates comprehensive flow diagrams of the RL training system
implemented in the improved_pure_env_training.py file.

Diagrams Generated:
1. Overall RL Training Flow
2. Episode Lifecycle
3. Network Update Process
4. Trajectory Visualization System
5. Curriculum Learning Flow
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_overall_rl_flow():
    """Create the main RL training flow diagram"""
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(5, 11.5, '🏭 REINFORCEMENT LEARNING TRAINING FLOW', 
            fontsize=20, fontweight='bold', ha='center')
    ax.text(5, 11, 'Industry-Standard Franka Robot Training System', 
            fontsize=14, ha='center', style='italic')
    
    # Define colors
    init_color = '#E8F4FD'      # Light blue
    env_color = '#D4F1D4'       # Light green
    agent_color = '#FFE6CC'     # Light orange
    train_color = '#F0E6FF'     # Light purple
    viz_color = '#FFE6E6'       # Light red
    
    # 1. Initialization Phase
    init_box = FancyBboxPatch((0.5, 9.5), 3, 1.5, 
                              boxstyle="round,pad=0.1", 
                              facecolor=init_color, edgecolor='blue', linewidth=2)
    ax.add_patch(init_box)
    ax.text(2, 10.6, '🚀 INITIALIZATION', fontweight='bold', ha='center', fontsize=12)
    ax.text(2, 10.3, '• Genesis Scene Setup', ha='center', fontsize=10)
    ax.text(2, 10.1, '• Franka Robot Loading', ha='center', fontsize=10)
    ax.text(2, 9.9, '• Environment Creation', ha='center', fontsize=10)
    ax.text(2, 9.7, '• DDPG Agent Setup', ha='center', fontsize=10)
    
    # 2. Environment Setup
    env_box = FancyBboxPatch((4.5, 9.5), 3, 1.5, 
                             boxstyle="round,pad=0.1", 
                             facecolor=env_color, edgecolor='green', linewidth=2)
    ax.add_patch(env_box)
    ax.text(6, 10.6, '🎯 ENVIRONMENT', fontweight='bold', ha='center', fontsize=12)
    ax.text(6, 10.3, '• Goal Generation', ha='center', fontsize=10)
    ax.text(6, 10.1, '• Curriculum Learning', ha='center', fontsize=10)
    ax.text(6, 9.9, '• Trajectory Tracking', ha='center', fontsize=10)
    ax.text(6, 9.7, '• Reward Structure', ha='center', fontsize=10)
    
    # 3. Training Loop
    train_box = FancyBboxPatch((2, 7.5), 4, 1.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor=train_color, edgecolor='purple', linewidth=2)
    ax.add_patch(train_box)
    ax.text(4, 8.6, '🔄 TRAINING LOOP', fontweight='bold', ha='center', fontsize=12)
    ax.text(4, 8.3, '• Episode Execution (10 episodes)', ha='center', fontsize=10)
    ax.text(4, 8.1, '• Action Selection (DDPG)', ha='center', fontsize=10)
    ax.text(4, 7.9, '• Network Updates (Actor-Critic)', ha='center', fontsize=10)
    ax.text(4, 7.7, '• Experience Replay', ha='center', fontsize=10)
    
    # 4. Agent Components
    agent_box = FancyBboxPatch((0.5, 5.5), 3, 1.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor=agent_color, edgecolor='orange', linewidth=2)
    ax.add_patch(agent_box)
    ax.text(2, 6.6, '🤖 DDPG AGENT', fontweight='bold', ha='center', fontsize=12)
    ax.text(2, 6.3, '• Actor Network', ha='center', fontsize=10)
    ax.text(2, 6.1, '• Critic Network', ha='center', fontsize=10)
    ax.text(2, 5.9, '• Target Networks', ha='center', fontsize=10)
    ax.text(2, 5.7, '• Exploration Noise', ha='center', fontsize=10)
    
    # 5. Visualization System
    viz_box = FancyBboxPatch((4.5, 5.5), 3, 1.5, 
                             boxstyle="round,pad=0.1", 
                             facecolor=viz_color, edgecolor='red', linewidth=2)
    ax.add_patch(viz_box)
    ax.text(6, 6.6, '📊 VISUALIZATION', fontweight='bold', ha='center', fontsize=12)
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
        # Environment to Training
        ((6, 9.5), (4, 8.5)),
        # Training to Agent
        ((2.5, 7.5), (2, 7)),
        # Training to Visualization
        ((5.5, 7.5), (6, 7)),
        # Episode flow arrows
        ((2.5, 4), (3.5, 3)),
        ((4.5, 3), (5.5, 2)),
        ((6.5, 2), (7.5, 1)),
    ]
    
    for start, end in arrows:
        arrow = ConnectionPatch(start, end, "data", "data",
                               arrowstyle="->", shrinkA=5, shrinkB=5, 
                               mutation_scale=20, fc="black", lw=2)
        ax.add_patch(arrow)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color=init_color, label='Initialization'),
        mpatches.Patch(color=env_color, label='Environment'),
        mpatches.Patch(color=agent_color, label='RL Agent'),
        mpatches.Patch(color=train_color, label='Training'),
        mpatches.Patch(color=viz_color, label='Visualization')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('rl_training_flow.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_episode_lifecycle():
    """Create detailed episode lifecycle diagram"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, '🔄 EPISODE LIFECYCLE FLOW', 
            fontsize=18, fontweight='bold', ha='center')
    
    # Episode phases
    phases = [
        ('🔄 RESET', 8.5, 'Environment reset\nTrajectory initialization\nGoal selection'),
        ('🎯 OBSERVE', 7.5, 'Get current state\nJoint positions\nEnd-effector pose'),
        ('🧠 DECIDE', 6.5, 'Actor network\nAction selection\nNoise exploration'),
        ('⚡ ACT', 5.5, 'Apply action\nPhysics simulation\nTrajectory update'),
        ('💰 REWARD', 4.5, 'Calculate reward\nGoal distance\nProgress tracking'),
        ('📚 LEARN', 3.5, 'Store experience\nNetwork updates\nTarget networks'),
        ('📊 ANALYZE', 2.5, 'Episode metrics\nTrajectory analysis\nSuccess evaluation'),
        ('✅ DONE?', 1.5, 'Check termination\nMax steps reached\nGoal achieved')
    ]
    
    # Draw phase boxes
    for i, (title, y, desc) in enumerate(phases):
        # Main box
        box = FancyBboxPatch((2, y-0.4), 6, 0.8, 
                             boxstyle="round,pad=0.1", 
                             facecolor='lightblue', edgecolor='navy', linewidth=2)
        ax.add_patch(box)
        
        # Title
        ax.text(2.5, y, title, fontweight='bold', fontsize=12, va='center')
        
        # Description
        ax.text(6, y, desc, fontsize=10, va='center')
        
        # Arrow to next phase (except last)
        if i < len(phases) - 1:
            arrow = ConnectionPatch((5, y-0.4), (5, y-0.6), "data", "data",
                                   arrowstyle="->", shrinkA=5, shrinkB=5, 
                                   mutation_scale=20, fc="blue", lw=2)
            ax.add_patch(arrow)
    
    # Loop back arrow
    loop_arrow = ConnectionPatch((8, 1.5), (8, 8.5), "data", "data",
                                arrowstyle="->", shrinkA=5, shrinkB=5, 
                                mutation_scale=20, fc="green", lw=3,
                                connectionstyle="arc3,rad=0.3")
    ax.add_patch(loop_arrow)
    ax.text(8.5, 5, 'Continue\nEpisode', ha='center', va='center', 
            fontweight='bold', color='green')
    
    # Episode end arrow
    end_arrow = ConnectionPatch((2, 1.5), (0.5, 0.5), "data", "data",
                               arrowstyle="->", shrinkA=5, shrinkB=5, 
                               mutation_scale=20, fc="red", lw=3)
    ax.add_patch(end_arrow)
    ax.text(0.5, 0.2, 'Next Episode', ha='center', va='center', 
            fontweight='bold', color='red')
    
    plt.tight_layout()
    plt.savefig('episode_lifecycle.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_network_update_flow():
    """Create network update process diagram"""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, '🧠 NEURAL NETWORK UPDATE FLOW', 
            fontsize=18, fontweight='bold', ha='center')
    
    # Experience Replay
    exp_box = FancyBboxPatch((1, 8), 8, 1, 
                             boxstyle="round,pad=0.1", 
                             facecolor='#FFE6CC', edgecolor='orange', linewidth=2)
    ax.add_patch(exp_box)
    ax.text(5, 8.5, '💾 EXPERIENCE REPLAY BUFFER', fontweight='bold', ha='center', fontsize=12)
    ax.text(5, 8.2, 'Store: (state, action, reward, next_state, done)', ha='center', fontsize=10)
    
    # Sample Batch
    sample_box = FancyBboxPatch((3, 6.5), 4, 0.8, 
                                boxstyle="round,pad=0.1", 
                                facecolor='lightgreen', edgecolor='green', linewidth=2)
    ax.add_patch(sample_box)
    ax.text(5, 6.9, '🎲 SAMPLE BATCH (128)', fontweight='bold', ha='center', fontsize=12)
    
    # Critic Update
    critic_box = FancyBboxPatch((0.5, 4.5), 4, 1.5, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#F0E6FF', edgecolor='purple', linewidth=2)
    ax.add_patch(critic_box)
    ax.text(2.5, 5.6, '🎯 CRITIC UPDATE', fontweight='bold', ha='center', fontsize=12)
    ax.text(2.5, 5.3, '• Compute target Q-values', ha='center', fontsize=9)
    ax.text(2.5, 5.1, '• Calculate TD error', ha='center', fontsize=9)
    ax.text(2.5, 4.9, '• MSE loss minimization', ha='center', fontsize=9)
    ax.text(2.5, 4.7, '• Gradient descent', ha='center', fontsize=9)
    
    # Actor Update
    actor_box = FancyBboxPatch((5.5, 4.5), 4, 1.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor='#E6F3FF', edgecolor='blue', linewidth=2)
    ax.add_patch(actor_box)
    ax.text(7.5, 5.6, '🎭 ACTOR UPDATE (PPO)', fontweight='bold', ha='center', fontsize=12)
    ax.text(7.5, 5.3, '• Compute advantage', ha='center', fontsize=9)
    ax.text(7.5, 5.1, '• PPO clipped objective', ha='center', fontsize=9)
    ax.text(7.5, 4.9, '• Entropy regularization', ha='center', fontsize=9)
    ax.text(7.5, 4.7, '• Policy gradient', ha='center', fontsize=9)
    
    # Target Network Update
    target_box = FancyBboxPatch((2.5, 2.5), 5, 1, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#FFE6E6', edgecolor='red', linewidth=2)
    ax.add_patch(target_box)
    ax.text(5, 3.2, '🎯 TARGET NETWORK SOFT UPDATE', fontweight='bold', ha='center', fontsize=12)
    ax.text(5, 2.9, 'θ_target = τ * θ + (1-τ) * θ_target', ha='center', fontsize=10)
    ax.text(5, 2.7, 'τ = 0.005 (soft update rate)', ha='center', fontsize=9)
    
    # Performance Metrics
    metrics_box = FancyBboxPatch((1, 0.5), 8, 1, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor='lightyellow', edgecolor='gold', linewidth=2)
    ax.add_patch(metrics_box)
    ax.text(5, 1.2, '📊 PERFORMANCE TRACKING', fontweight='bold', ha='center', fontsize=12)
    ax.text(5, 0.9, 'Actor Loss | Critic Loss | Exploration Noise | Success Rate', ha='center', fontsize=10)
    
    # Add arrows
    arrows = [
        ((5, 8), (5, 7.3)),      # Buffer to Sample
        ((5, 6.5), (2.5, 6)),    # Sample to Critic
        ((5, 6.5), (7.5, 6)),    # Sample to Actor
        ((2.5, 4.5), (4, 3.5)),  # Critic to Target
        ((7.5, 4.5), (6, 3.5)),  # Actor to Target
        ((5, 2.5), (5, 1.5)),    # Target to Metrics
    ]
    
    for start, end in arrows:
        arrow = ConnectionPatch(start, end, "data", "data",
                               arrowstyle="->", shrinkA=5, shrinkB=5, 
                               mutation_scale=20, fc="black", lw=2)
        ax.add_patch(arrow)
    
    plt.tight_layout()
    plt.savefig('network_update_flow.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_trajectory_visualization_flow():
    """Create trajectory visualization system diagram"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(6, 9.5, '🏭 INDUSTRY-STANDARD TRAJECTORY VISUALIZATION', 
            fontsize=16, fontweight='bold', ha='center')
    
    # Initialization
    init_box = FancyBboxPatch((0.5, 8), 3.5, 1.2, 
                              boxstyle="round,pad=0.1", 
                              facecolor='#E8F4FD', edgecolor='blue', linewidth=2)
    ax.add_patch(init_box)
    ax.text(2.25, 8.8, '🚀 INITIALIZATION', fontweight='bold', ha='center', fontsize=11)
    ax.text(2.25, 8.5, '• Pre-allocate 500 markers', ha='center', fontsize=9)
    ax.text(2.25, 8.3, '• Create goal visualization', ha='center', fontsize=9)
    ax.text(2.25, 8.1, '• Setup trajectory tracking', ha='center', fontsize=9)
    
    # Real-time Tracking
    track_box = FancyBboxPatch((4.5, 8), 3.5, 1.2, 
                               boxstyle="round,pad=0.1", 
                               facecolor='#D4F1D4', edgecolor='green', linewidth=2)
    ax.add_patch(track_box)
    ax.text(6.25, 8.8, '📍 REAL-TIME TRACKING', fontweight='bold', ha='center', fontsize=11)
    ax.text(6.25, 8.5, '• Record EE positions', ha='center', fontsize=9)
    ax.text(6.25, 8.3, '• Update markers (every 10 steps)', ha='center', fontsize=9)
    ax.text(6.25, 8.1, '• Calculate distances', ha='center', fontsize=9)
    
    # Analysis
    analysis_box = FancyBboxPatch((8.5, 8), 3, 1.2, 
                                  boxstyle="round,pad=0.1", 
                                  facecolor='#FFE6CC', edgecolor='orange', linewidth=2)
    ax.add_patch(analysis_box)
    ax.text(10, 8.8, '📊 ANALYSIS', fontweight='bold', ha='center', fontsize=11)
    ax.text(10, 8.5, '• Trajectory efficiency', ha='center', fontsize=9)
    ax.text(10, 8.3, '• Path length metrics', ha='center', fontsize=9)
    ax.text(10, 8.1, '• Success tracking', ha='center', fontsize=9)
    
    # Goal Markers
    goal_box = FancyBboxPatch((1, 6), 4, 1.5, 
                              boxstyle="round,pad=0.1", 
                              facecolor='#F0E6FF', edgecolor='purple', linewidth=2)
    ax.add_patch(goal_box)
    ax.text(3, 7, '🎯 GOAL VISUALIZATION', fontweight='bold', ha='center', fontsize=11)
    ax.text(3, 6.7, '• Primary Goal (8cm sphere)', ha='center', fontsize=9)
    ax.text(3, 6.5, '• Tolerance Zone (20cm radius)', ha='center', fontsize=9)
    ax.text(3, 6.3, '• Reference Goals (3cm spheres)', ha='center', fontsize=9)
    ax.text(3, 6.1, '• Professional hierarchy', ha='center', fontsize=9)
    
    # Trajectory Markers
    traj_box = FancyBboxPatch((6, 6), 4, 1.5, 
                              boxstyle="round,pad=0.1", 
                              facecolor='#FFE6E6', edgecolor='red', linewidth=2)
    ax.add_patch(traj_box)
    ax.text(8, 7, '📈 TRAJECTORY MARKERS', fontweight='bold', ha='center', fontsize=11)
    ax.text(8, 6.7, '• Real-time path points', ha='center', fontsize=9)
    ax.text(8, 6.5, '• Color-coded episodes', ha='center', fontsize=9)
    ax.text(8, 6.3, '• Success indicators', ha='center', fontsize=9)
    ax.text(8, 6.1, '• Multi-episode comparison', ha='center', fontsize=9)
    
    # Episode Management
    episode_box = FancyBboxPatch((1, 3.5), 4, 1.5, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor='lightyellow', edgecolor='gold', linewidth=2)
    ax.add_patch(episode_box)
    ax.text(3, 4.5, '🔄 EPISODE MANAGEMENT', fontweight='bold', ha='center', fontsize=11)
    ax.text(3, 4.2, '• Reset marker pool', ha='center', fontsize=9)
    ax.text(3, 4.0, '• Initialize tracking', ha='center', fontsize=9)
    ax.text(3, 3.8, '• Finalize trajectory', ha='center', fontsize=9)
    ax.text(3, 3.6, '• Store episode data', ha='center', fontsize=9)
    
    # Professional Metrics
    metrics_box = FancyBboxPatch((6, 3.5), 4, 1.5, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor='lightcyan', edgecolor='teal', linewidth=2)
    ax.add_patch(metrics_box)
    ax.text(8, 4.5, '📋 PROFESSIONAL METRICS', fontweight='bold', ha='center', fontsize=11)
    ax.text(8, 4.2, '• Efficiency = direct/actual', ha='center', fontsize=9)
    ax.text(8, 4.0, '• Total distance traveled', ha='center', fontsize=9)
    ax.text(8, 3.8, '• Success rate tracking', ha='center', fontsize=9)
    ax.text(8, 3.6, '• Historical comparison', ha='center', fontsize=9)
    
    # Final Report
    report_box = FancyBboxPatch((3.5, 1.5), 5, 1, 
                                boxstyle="round,pad=0.1", 
                                facecolor='lightgreen', edgecolor='darkgreen', linewidth=2)
    ax.add_patch(report_box)
    ax.text(6, 2.2, '📊 COMPREHENSIVE ANALYSIS REPORT', fontweight='bold', ha='center', fontsize=11)
    ax.text(6, 1.9, 'Episode statistics • Trajectory efficiency • Success metrics', ha='center', fontsize=9)
    
    # Add flow arrows
    arrows = [
        ((2.25, 8), (2.25, 7.5)),    # Init to Goals
        ((6.25, 8), (8, 7.5)),       # Tracking to Trajectory
        ((10, 8), (8, 5)),           # Analysis to Metrics
        ((3, 6), (3, 5)),            # Goals to Episode
        ((8, 6), (8, 5)),            # Trajectory to Metrics
        ((5, 4.25), (5.5, 2.5)),     # Episode to Report
        ((6.5, 3.5), (6.5, 2.5)),    # Metrics to Report
    ]
    
    for start, end in arrows:
        arrow = ConnectionPatch(start, end, "data", "data",
                               arrowstyle="->", shrinkA=5, shrinkB=5, 
                               mutation_scale=15, fc="black", lw=1.5)
        ax.add_patch(arrow)
    
    plt.tight_layout()
    plt.savefig('trajectory_visualization_flow.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_curriculum_learning_flow():
    """Create curriculum learning progression diagram"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(5, 7.5, '🎓 CURRICULUM LEARNING PROGRESSION', 
            fontsize=16, fontweight='bold', ha='center')
    
    # Level 1: Easy
    easy_box = FancyBboxPatch((0.5, 5.5), 2.5, 1.5, 
                              boxstyle="round,pad=0.1", 
                              facecolor='#D4F1D4', edgecolor='green', linewidth=2)
    ax.add_patch(easy_box)
    ax.text(1.75, 6.6, '🟢 LEVEL 1: EASY', fontweight='bold', ha='center', fontsize=11)
    ax.text(1.75, 6.3, 'Radius: 0.3-0.4m', ha='center', fontsize=9)
    ax.text(1.75, 6.1, 'Goals: 12 positions', ha='center', fontsize=9)
    ax.text(1.75, 5.9, 'Tolerance: 0.20m', ha='center', fontsize=9)
    ax.text(1.75, 5.7, 'Close to robot', ha='center', fontsize=9)
    
    # Level 2: Medium
    medium_box = FancyBboxPatch((3.75, 5.5), 2.5, 1.5, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#FFE6CC', edgecolor='orange', linewidth=2)
    ax.add_patch(medium_box)
    ax.text(5, 6.6, '🟡 LEVEL 2: MEDIUM', fontweight='bold', ha='center', fontsize=11)
    ax.text(5, 6.3, 'Radius: 0.5-0.6m', ha='center', fontsize=9)
    ax.text(5, 6.1, 'Goals: 16 positions', ha='center', fontsize=9)
    ax.text(5, 5.9, 'Mixed difficulty', ha='center', fontsize=9)
    ax.text(5, 5.7, 'Moderate reach', ha='center', fontsize=9)
    
    # Level 3: Hard
    hard_box = FancyBboxPatch((7, 5.5), 2.5, 1.5, 
                              boxstyle="round,pad=0.1", 
                              facecolor='#FFE6E6', edgecolor='red', linewidth=2)
    ax.add_patch(hard_box)
    ax.text(8.25, 6.6, '🔴 LEVEL 3: HARD', fontweight='bold', ha='center', fontsize=11)
    ax.text(8.25, 6.3, 'Radius: 0.7-0.8m', ha='center', fontsize=9)
    ax.text(8.25, 6.1, 'Goals: 16 positions', ha='center', fontsize=9)
    ax.text(8.25, 5.9, 'Full workspace', ha='center', fontsize=9)
    ax.text(8.25, 5.7, 'Maximum reach', ha='center', fontsize=9)
    
    # Progression Criteria
    progress_box = FancyBboxPatch((2, 3.5), 6, 1.5, 
                                  boxstyle="round,pad=0.1", 
                                  facecolor='#F0E6FF', edgecolor='purple', linewidth=2)
    ax.add_patch(progress_box)
    ax.text(5, 4.6, '📈 PROGRESSION CRITERIA', fontweight='bold', ha='center', fontsize=12)
    ax.text(5, 4.3, '• Advance level every 3 goals reached', ha='center', fontsize=10)
    ax.text(5, 4.1, '• Dynamic goal selection based on current level', ha='center', fontsize=10)
    ax.text(5, 3.9, '• Success rate tracking per level', ha='center', fontsize=10)
    ax.text(5, 3.7, '• Adaptive difficulty based on performance', ha='center', fontsize=10)
    
    # Benefits
    benefits_box = FancyBboxPatch((1, 1.5), 8, 1.5, 
                                  boxstyle="round,pad=0.1", 
                                  facecolor='lightyellow', edgecolor='gold', linewidth=2)
    ax.add_patch(benefits_box)
    ax.text(5, 2.6, '✅ CURRICULUM LEARNING BENEFITS', fontweight='bold', ha='center', fontsize=12)
    ax.text(5, 2.3, '• Faster convergence • Stable learning • Better exploration', ha='center', fontsize=10)
    ax.text(5, 2.1, '• Reduced sample complexity • Higher success rates', ha='center', fontsize=10)
    ax.text(5, 1.9, '• Progressive skill development • Robust policy learning', ha='center', fontsize=10)
    ax.text(5, 1.7, '• Industry-standard training methodology', ha='center', fontsize=10)
    
    # Add progression arrows
    arrow1 = ConnectionPatch((3, 6.25), (3.75, 6.25), "data", "data",
                            arrowstyle="->", shrinkA=5, shrinkB=5, 
                            mutation_scale=20, fc="blue", lw=3)
    ax.add_patch(arrow1)
    
    arrow2 = ConnectionPatch((6.25, 6.25), (7, 6.25), "data", "data",
                            arrowstyle="->", shrinkA=5, shrinkB=5, 
                            mutation_scale=20, fc="blue", lw=3)
    ax.add_patch(arrow2)
    
    # Success arrows
    for x in [1.75, 5, 8.25]:
        arrow = ConnectionPatch((x, 5.5), (x, 5), "data", "data",
                               arrowstyle="->", shrinkA=5, shrinkB=5, 
                               mutation_scale=15, fc="green", lw=2)
        ax.add_patch(arrow)
    
    plt.tight_layout()
    plt.savefig('curriculum_learning_flow.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Generate all flow diagrams"""
    print("🏭 GENERATING REINFORCEMENT LEARNING FLOW DIAGRAMS")
    print("=" * 60)
    
    print("1. 📊 Creating Overall RL Training Flow...")
    create_overall_rl_flow()
    
    print("2. 🔄 Creating Episode Lifecycle Diagram...")
    create_episode_lifecycle()
    
    print("3. 🧠 Creating Network Update Flow...")
    create_network_update_flow()
    
    print("4. 📈 Creating Trajectory Visualization Flow...")
    create_trajectory_visualization_flow()
    
    print("5. 🎓 Creating Curriculum Learning Flow...")
    create_curriculum_learning_flow()
    
    print("\n✅ ALL DIAGRAMS GENERATED SUCCESSFULLY!")
    print("📁 Files created:")
    print("   • rl_training_flow.png")
    print("   • episode_lifecycle.png") 
    print("   • network_update_flow.png")
    print("   • trajectory_visualization_flow.png")
    print("   • curriculum_learning_flow.png")

if __name__ == "__main__":
    main()
