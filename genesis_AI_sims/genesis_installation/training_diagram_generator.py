#!/usr/bin/env python3
"""
Training Diagram Generator
Creates a visual representation of the current training implementation
to compare with the JPEG diagram
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_training_diagram():
    """Create comprehensive training flow diagram"""
    
    # Create figure with high DPI for clarity
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Color scheme
    colors = {
        'environment': '#E8F4FD',  # Light blue
        'networks': '#FFF2CC',     # Light yellow
        'training': '#E1D5E7',     # Light purple
        'data': '#D5E8D4',         # Light green
        'flow': '#F8CECC',         # Light red
        'text': '#000000'          # Black
    }
    
    # Title
    ax.text(8, 11.5, 'CURRENT TRAINING IMPLEMENTATION FLOW', 
            fontsize=16, fontweight='bold', ha='center')
    ax.text(8, 11, 'Based on corrected_pure_env_training.py', 
            fontsize=12, ha='center', style='italic')
    
    # ============================================================================
    # ENVIRONMENT SECTION (Left side)
    # ============================================================================
    
    # Environment box
    env_box = FancyBboxPatch((0.5, 7), 4, 3.5, 
                             boxstyle="round,pad=0.1", 
                             facecolor=colors['environment'],
                             edgecolor='blue', linewidth=2)
    ax.add_patch(env_box)
    ax.text(2.5, 10, 'FRANKA ENVIRONMENT', fontsize=12, fontweight='bold', ha='center')
    
    # Environment components
    ax.text(2.5, 9.5, '• Genesis Physics Engine', fontsize=10, ha='center')
    ax.text(2.5, 9.2, '• XML-aligned Joint Limits', fontsize=10, ha='center')
    ax.text(2.5, 8.9, '• Dynamic Goal Generation', fontsize=10, ha='center')
    ax.text(2.5, 8.6, '• Workspace Safety', fontsize=10, ha='center')
    ax.text(2.5, 8.3, '• Pure Environment Learning', fontsize=10, ha='center')
    ax.text(2.5, 8.0, '• State: s(t) [28 dims]', fontsize=10, ha='center', color='red')
    ax.text(2.5, 7.7, '• Action: A(t) [9 dims]', fontsize=10, ha='center', color='red')
    ax.text(2.5, 7.4, '• Reward: R(t)', fontsize=10, ha='center', color='red')
    
    # ============================================================================
    # NEURAL NETWORKS SECTION (Center)
    # ============================================================================
    
    # Actor Network
    actor_box = FancyBboxPatch((6, 8.5), 3.5, 2, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['networks'],
                               edgecolor='orange', linewidth=2)
    ax.add_patch(actor_box)
    ax.text(7.75, 10, 'ACTOR NETWORK', fontsize=12, fontweight='bold', ha='center')
    ax.text(7.75, 9.6, 'Input: s(t) → Output: A(t)', fontsize=10, ha='center')
    ax.text(7.75, 9.3, '• 3 Linear Layers (256 hidden)', fontsize=9, ha='center')
    ax.text(7.75, 9.0, '• ReLU + Tanh activation', fontsize=9, ha='center')
    ax.text(7.75, 8.7, '• Xavier initialization', fontsize=9, ha='center')
    
    # Critic Network
    critic_box = FancyBboxPatch((6, 6), 3.5, 2, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['networks'],
                                edgecolor='orange', linewidth=2)
    ax.add_patch(critic_box)
    ax.text(7.75, 7.5, 'CRITIC NETWORK', fontsize=12, fontweight='bold', ha='center')
    ax.text(7.75, 7.1, 'Input: s(t), A(t) → Output: Q(s,a)', fontsize=10, ha='center')
    ax.text(7.75, 6.8, '• State + Action processing', fontsize=9, ha='center')
    ax.text(7.75, 6.5, '• Combined feature fusion', fontsize=9, ha='center')
    ax.text(7.75, 6.2, '• Q-value estimation', fontsize=9, ha='center')
    
    # Target Networks
    target_box = FancyBboxPatch((10.5, 7.25), 3, 1.5, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['networks'],
                                edgecolor='gray', linewidth=2, linestyle='--')
    ax.add_patch(target_box)
    ax.text(12, 8.3, 'TARGET NETWORKS', fontsize=11, fontweight='bold', ha='center')
    ax.text(12, 7.9, 'Actor_target', fontsize=10, ha='center')
    ax.text(12, 7.6, 'Critic_target', fontsize=10, ha='center')
    ax.text(12, 7.3, 'Soft update: τ=0.005', fontsize=9, ha='center')
    
    # ============================================================================
    # TRAINING ALGORITHM SECTION (Right side)
    # ============================================================================
    
    # PPO-DDPG Algorithm
    algo_box = FancyBboxPatch((11, 3.5), 4.5, 3, 
                              boxstyle="round,pad=0.1", 
                              facecolor=colors['training'],
                              edgecolor='purple', linewidth=2)
    ax.add_patch(algo_box)
    ax.text(13.25, 6, 'PPO-DDPG ALGORITHM', fontsize=12, fontweight='bold', ha='center')
    
    # PPO Components
    ax.text(13.25, 5.6, '1. Ratio = π_θ(a|s) / π_θ_old(a|s)', fontsize=9, ha='center')
    ax.text(13.25, 5.3, '2. A(t) = Q_target - Q_current', fontsize=9, ha='center')
    ax.text(13.25, 5.0, '3. Lclip = min(ratio×A(t), clip×A(t))', fontsize=9, ha='center')
    ax.text(13.25, 4.7, '4. Actor Loss = -Lclip + entropy', fontsize=9, ha='center')
    ax.text(13.25, 4.4, '5. Critic Loss = MSE(Q, Q_target)', fontsize=9, ha='center')
    ax.text(13.25, 4.1, '6. Backpropagation', fontsize=9, ha='center')
    ax.text(13.25, 3.8, 'ε = 0.2, entropy_coeff = 0.01', fontsize=8, ha='center', style='italic')
    
    # ============================================================================
    # EXPERIENCE REPLAY SECTION (Bottom)
    # ============================================================================
    
    # Experience Replay Buffer
    replay_box = FancyBboxPatch((1, 1), 6, 2, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['data'],
                                edgecolor='green', linewidth=2)
    ax.add_patch(replay_box)
    ax.text(4, 2.5, 'EXPERIENCE REPLAY BUFFER', fontsize=12, fontweight='bold', ha='center')
    ax.text(4, 2.1, 'Store: (s(t), A(t), R(t), s(t+1), done)', fontsize=10, ha='center')
    ax.text(4, 1.8, 'Capacity: 50,000 transitions', fontsize=10, ha='center')
    ax.text(4, 1.5, 'Batch Size: 64', fontsize=10, ha='center')
    ax.text(4, 1.2, 'Random Sampling for Training', fontsize=10, ha='center')
    
    # Training Loop
    loop_box = FancyBboxPatch((8.5, 1), 6, 2, 
                              boxstyle="round,pad=0.1", 
                              facecolor=colors['flow'],
                              edgecolor='red', linewidth=2)
    ax.add_patch(loop_box)
    ax.text(11.5, 2.5, 'TRAINING LOOP', fontsize=12, fontweight='bold', ha='center')
    ax.text(11.5, 2.1, '1. Select Action with Noise', fontsize=10, ha='center')
    ax.text(11.5, 1.8, '2. Environment Step', fontsize=10, ha='center')
    ax.text(11.5, 1.5, '3. Store Experience', fontsize=10, ha='center')
    ax.text(11.5, 1.2, '4. Update Networks', fontsize=10, ha='center')
    
    # ============================================================================
    # ARROWS AND CONNECTIONS
    # ============================================================================
    
    # Environment to Networks
    arrow1 = ConnectionPatch((4.5, 8.75), (6, 9.5), "data", "data",
                           arrowstyle="->", shrinkA=5, shrinkB=5, 
                           mutation_scale=20, fc="blue", lw=2)
    ax.add_patch(arrow1)
    ax.text(5.25, 9.2, 's(t)', fontsize=9, ha='center', color='blue', fontweight='bold')
    
    # Actor to Environment
    arrow2 = ConnectionPatch((6, 9), (4.5, 8.25), "data", "data",
                           arrowstyle="->", shrinkA=5, shrinkB=5, 
                           mutation_scale=20, fc="orange", lw=2)
    ax.add_patch(arrow2)
    ax.text(5.25, 8.5, 'A(t)', fontsize=9, ha='center', color='orange', fontweight='bold')
    
    # Environment to Replay Buffer
    arrow3 = ConnectionPatch((2.5, 7), (3, 3), "data", "data",
                           arrowstyle="->", shrinkA=5, shrinkB=5, 
                           mutation_scale=20, fc="green", lw=2)
    ax.add_patch(arrow3)
    ax.text(2, 5, 'Experience', fontsize=9, ha='center', color='green', 
            fontweight='bold', rotation=90)
    
    # Replay Buffer to Training
    arrow4 = ConnectionPatch((7, 2), (11, 4.5), "data", "data",
                           arrowstyle="->", shrinkA=5, shrinkB=5, 
                           mutation_scale=20, fc="purple", lw=2)
    ax.add_patch(arrow4)
    ax.text(9, 3.5, 'Batch\nSampling', fontsize=9, ha='center', color='purple', fontweight='bold')
    
    # Training to Networks (Gradients)
    arrow5 = ConnectionPatch((11, 5.5), (9.5, 8), "data", "data",
                           arrowstyle="->", shrinkA=5, shrinkB=5, 
                           mutation_scale=20, fc="red", lw=2)
    ax.add_patch(arrow5)
    ax.text(10.5, 6.8, 'Gradients', fontsize=9, ha='center', color='red', 
            fontweight='bold', rotation=60)
    
    # Target Network Updates
    arrow6 = ConnectionPatch((10.5, 8), (9.5, 8.5), "data", "data",
                           arrowstyle="->", shrinkA=5, shrinkB=5, 
                           mutation_scale=20, fc="gray", lw=2, linestyle='--')
    ax.add_patch(arrow6)
    ax.text(10, 8.7, 'Soft\nUpdate', fontsize=8, ha='center', color='gray')
    
    arrow7 = ConnectionPatch((10.5, 7.5), (9.5, 7), "data", "data",
                           arrowstyle="->", shrinkA=5, shrinkB=5, 
                           mutation_scale=20, fc="gray", lw=2, linestyle='--')
    ax.add_patch(arrow7)
    
    # ============================================================================
    # MATHEMATICAL FORMULAS
    # ============================================================================
    
    # Key formulas box
    formula_box = FancyBboxPatch((0.5, 4), 4.5, 2.5, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor='#FFFACD',
                                 edgecolor='black', linewidth=1)
    ax.add_patch(formula_box)
    ax.text(2.75, 6.2, 'KEY MATHEMATICAL COMPONENTS', fontsize=11, fontweight='bold', ha='center')
    
    ax.text(2.75, 5.8, 'PPO Clipped Objective:', fontsize=10, ha='center', fontweight='bold')
    ax.text(2.75, 5.5, 'Lclip = E[min(rt(θ)·Ât, clip(rt(θ), 1-ε, 1+ε)·Ât)]', fontsize=9, ha='center')
    
    ax.text(2.75, 5.1, 'Advantage Estimation:', fontsize=10, ha='center', fontweight='bold')
    ax.text(2.75, 4.8, 'A(t) = Q_target(s,a) - Q_current(s,a)', fontsize=9, ha='center')
    
    ax.text(2.75, 4.4, 'Actor Loss = -Lclip - β·H(π)', fontsize=9, ha='center')
    ax.text(2.75, 4.1, 'Critic Loss = MSE(Q_target, Q_current)', fontsize=9, ha='center')
    
    # ============================================================================
    # LEGEND
    # ============================================================================
    
    ax.text(0.5, 0.5, 'LEGEND:', fontsize=10, fontweight='bold')
    ax.text(0.5, 0.2, '• Blue: Environment Interface  • Orange: Neural Networks  • Purple: Training Algorithm', fontsize=8)
    ax.text(0.5, 0.0, '• Green: Data Storage  • Red: Training Flow  • Gray: Target Network Updates', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('current_training_diagram.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig('current_training_diagram.pdf', bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print("✅ Training diagram saved as:")
    print("   📊 current_training_diagram.png (High-res image)")
    print("   📄 current_training_diagram.pdf (Vector format)")
    
    plt.show()

def create_detailed_algorithm_flow():
    """Create detailed algorithm flow diagram"""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(7, 9.5, 'DETAILED ALGORITHM FLOW', 
            fontsize=16, fontweight='bold', ha='center')
    
    # Step boxes
    steps = [
        (2, 8, "1. ENVIRONMENT\nSTEP", "s(t), A(t) → s(t+1), R(t)"),
        (6, 8, "2. STORE\nEXPERIENCE", "(s, a, r, s', done)\n→ Replay Buffer"),
        (10, 8, "3. SAMPLE\nBATCH", "Random batch\nfrom buffer"),
        (2, 6, "4. COMPUTE\nTARGET Q", "Q_target = R + γ·Q'(s',a')"),
        (6, 6, "5. ADVANTAGE\nESTIMATION", "A(t) = Q_target - Q(s,a)"),
        (10, 6, "6. RATIO\nCOMPUTATION", "rt = π(a|s) / π_old(a|s)"),
        (2, 4, "7. PPO CLIPPING", "Lclip = min(rt·A, clip·A)"),
        (6, 4, "8. ACTOR\nUPDATE", "∇θ Lclip + entropy"),
        (10, 4, "9. CRITIC\nUPDATE", "∇φ MSE(Q, Q_target)"),
        (6, 2, "10. SOFT TARGET\nUPDATE", "θ' ← τθ + (1-τ)θ'")
    ]
    
    for i, (x, y, title, desc) in enumerate(steps):
        # Color based on step type
        if i < 3:
            color = '#E8F4FD'  # Environment steps - blue
        elif i < 6:
            color = '#D5E8D4'  # Data processing - green
        elif i < 9:
            color = '#E1D5E7'  # Network updates - purple
        else:
            color = '#FFF2CC'  # Target updates - yellow
        
        box = FancyBboxPatch((x-0.8, y-0.6), 1.6, 1.2, 
                             boxstyle="round,pad=0.1", 
                             facecolor=color,
                             edgecolor='black', linewidth=1)
        ax.add_patch(box)
        ax.text(x, y+0.2, title, fontsize=10, fontweight='bold', ha='center')
        ax.text(x, y-0.2, desc, fontsize=8, ha='center')
    
    # Add arrows between steps
    arrows = [
        ((2.8, 8), (5.2, 8)),    # 1→2
        ((6.8, 8), (9.2, 8)),    # 2→3
        ((10, 7.4), (6.8, 6.6)), # 3→5
        ((10, 7.4), (2.8, 6.6)), # 3→4
        ((2.8, 6), (5.2, 6)),    # 4→5
        ((6.8, 6), (9.2, 6)),    # 5→6
        ((9.2, 6), (2.8, 4.6)),  # 6→7
        ((2.8, 4), (5.2, 4)),    # 7→8
        ((6.8, 4), (9.2, 4)),    # 8→9
        ((6, 3.4), (6, 2.6))     # 9→10
    ]
    
    for start, end in arrows:
        arrow = ConnectionPatch(start, end, "data", "data",
                               arrowstyle="->", shrinkA=5, shrinkB=5, 
                               mutation_scale=15, fc="black", lw=1.5)
        ax.add_patch(arrow)
    
    plt.tight_layout()
    plt.savefig('detailed_algorithm_flow.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print("✅ Detailed algorithm flow saved as:")
    print("   📊 detailed_algorithm_flow.png")
    
    plt.show()

if __name__ == "__main__":
    print("🎨 Generating training diagrams...")
    print("\n1. Creating comprehensive training flow diagram...")
    create_training_diagram()
    
    print("\n2. Creating detailed algorithm flow diagram...")
    create_detailed_algorithm_flow()
    
    print("\n🎯 Compare these diagrams with your JPEG image to verify alignment!")
