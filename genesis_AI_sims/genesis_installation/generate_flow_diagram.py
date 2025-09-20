#!/usr/bin/env python3
"""
🏭 TRAINING CONTROL SYSTEM FLOW DIAGRAM GENERATOR
==================================================

This script generates a comprehensive flow diagram of the reinforcement learning
training control system with industry-standard trajectory visualization.

The diagram shows:
- System initialization and setup
- Training loop control flow
- Actor-Critic network interactions
- Experience replay buffer management
- Trajectory visualization system
- Goal management and curriculum learning
- Professional metrics and analysis

Usage:
    python generate_flow_diagram.py
    
This will create a detailed flowchart showing the complete training system architecture.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Arrow
import numpy as np

def create_training_flow_diagram():
    """Generate comprehensive training control system flow diagram"""
    
    # Create figure with high resolution for professional quality
    fig, ax = plt.subplots(1, 1, figsize=(20, 24))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 24)
    ax.axis('off')
    
    # Color scheme for different components
    colors = {
        'initialization': '#E8F4FD',  # Light blue
        'training_loop': '#FFF2CC',   # Light yellow
        'networks': '#D5E8D4',        # Light green
        'experience': '#F8CECC',      # Light red
        'visualization': '#E1D5E7',   # Light purple
        'analysis': '#FFE6CC',        # Light orange
        'control': '#DBEAFE'          # Light blue-gray
    }
    
    # Helper function to create boxes
    def create_box(x, y, width, height, text, color, text_size=9):
        box = FancyBboxPatch((x, y), width, height,
                           boxstyle="round,pad=0.1",
                           facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + width/2, y + height/2, text, ha='center', va='center', 
                fontsize=text_size, weight='bold', wrap=True)
    
    # Helper function to create arrows
    def create_arrow(x1, y1, x2, y2, color='black', style='->', width=2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle=style, color=color, lw=width))
    
    # Title
    ax.text(10, 23, '🏭 TRAINING CONTROL SYSTEM FLOW DIAGRAM', 
            ha='center', va='center', fontsize=20, weight='bold')
    ax.text(10, 22.5, 'Reinforcement Learning with Industry-Standard Trajectory Visualization', 
            ha='center', va='center', fontsize=14, style='italic')
    
    # 1. SYSTEM INITIALIZATION (Top Level)
    create_box(1, 21, 18, 1, '🚀 SYSTEM INITIALIZATION\n• Genesis Physics Engine • Actor-Critic Networks • Experience Replay Buffer\n• Trajectory Visualization System • Goal Management • Professional Metrics', 
               colors['initialization'], 10)
    
    # 2. ENVIRONMENT SETUP
    create_box(1, 19.5, 5.5, 1, '🌍 ENVIRONMENT SETUP\n• Genesis Scene Creation\n• Franka Robot Loading\n• Camera & Lighting Setup\n• Joint Limits Configuration', 
               colors['initialization'])
    
    create_box(7, 19.5, 5.5, 1, '🧠 NETWORK INITIALIZATION\n• Actor Network (State→Action)\n• Critic Network (State→Value)\n• Target Networks\n• Optimizer Setup', 
               colors['networks'])
    
    create_box(13.5, 19.5, 5.5, 1, '🎯 TRAJECTORY SYSTEM\n• Pre-allocate 500 Markers\n• Goal Marker Creation\n• Visualization Pipeline\n• Analysis Metrics Setup', 
               colors['visualization'])
    
    # Arrows from initialization
    create_arrow(3.75, 21, 3.75, 20.5, 'blue')
    create_arrow(9.75, 21, 9.75, 20.5, 'green')
    create_arrow(16.25, 21, 16.25, 20.5, 'purple')
    
    # 3. TRAINING LOOP START
    create_box(8, 18, 4, 0.8, '🔄 TRAINING LOOP START\nEpisode Counter++', 
               colors['training_loop'])
    
    # Arrows to training loop
    create_arrow(3.75, 19.5, 8, 18.5, 'blue')
    create_arrow(9.75, 19.5, 10, 18.8, 'green')
    create_arrow(16.25, 19.5, 12, 18.5, 'purple')
    
    # 4. EPISODE INITIALIZATION
    create_box(1, 16.5, 5, 1, '📊 EPISODE INITIALIZATION\n• Reset Environment State\n• Initialize Trajectory Tracking\n• Reset Marker Pool\n• Select Curriculum Goal', 
               colors['control'])
    
    create_box(7, 16.5, 6, 1, '🎯 GOAL & VISUALIZATION SETUP\n• Create Goal Markers (Primary + Tolerance)\n• Reset Trajectory Points Array\n• Initialize Performance Metrics\n• Professional Episode Logging', 
               colors['visualization'])
    
    create_box(14, 16.5, 5, 1, '🏠 ROBOT RESET\n• Move to Home Position\n• Clear Previous Actions\n• Reset Joint States\n• Initialize Observation', 
               colors['control'])
    
    # Arrow from training loop to initialization
    create_arrow(10, 18, 10, 17.5, 'orange')
    create_arrow(8.5, 17.5, 3.5, 17.5, 'orange')
    create_arrow(11.5, 17.5, 16.5, 17.5, 'orange')
    
    # 5. STEP LOOP
    create_box(8, 15, 4, 0.8, '🔁 STEP LOOP\nStep Counter++', 
               colors['training_loop'])
    
    create_arrow(10, 16.5, 10, 15.8, 'orange')
    
    # 6. ACTION GENERATION AND EXECUTION
    create_box(1, 13.5, 4.5, 1, '🧠 ACTION GENERATION\n• Get Current State\n• Actor Network Forward\n• Add Exploration Noise\n• Action Smoothing (25%)', 
               colors['networks'])
    
    create_box(6, 13.5, 4, 1, '⚙️ ACTION EXECUTION\n• Scale to Joint Limits\n• Apply to Robot\n• Physics Step\n• Update Simulation', 
               colors['control'])
    
    create_box(10.5, 13.5, 4, 1, '📍 TRAJECTORY UPDATE\n• Record EE Position\n• Activate Marker from Pool\n• Update Visualization\n• Calculate Metrics', 
               colors['visualization'])
    
    create_box(15, 13.5, 4, 1, '🎯 REWARD CALCULATION\n• Goal Distance Check\n• Trajectory Efficiency\n• Success Detection\n• Performance Scoring', 
               colors['analysis'])
    
    # Arrows for action flow
    create_arrow(10, 15, 3.25, 14.5, 'green')
    create_arrow(5.5, 14, 6, 14, 'green')
    create_arrow(10, 14, 10.5, 14, 'purple')
    create_arrow(14.5, 14, 15, 14, 'orange')
    
    # 7. EXPERIENCE STORAGE
    create_box(6, 12, 8, 1, '💾 EXPERIENCE REPLAY BUFFER\n• Store (State, Action, Reward, Next_State, Done)\n• Buffer Management (10,000 capacity)\n• Mini-batch Sampling for Training', 
               colors['experience'])
    
    # Arrows to experience buffer
    create_arrow(3.25, 13.5, 7, 13, 'red')
    create_arrow(8, 13.5, 9, 13, 'red')
    create_arrow(12.5, 13.5, 11, 13, 'red')
    create_arrow(17, 13.5, 13, 13, 'red')
    
    # 8. NETWORK TRAINING (when buffer has enough samples)
    create_box(2, 10.5, 7, 1, '🎓 ACTOR-CRITIC TRAINING\n• Sample Mini-batch from Buffer\n• Critic Loss: TD Error Minimization\n• Actor Loss: Policy Gradient with Critic Values\n• Soft Target Network Updates', 
               colors['networks'])
    
    create_box(10, 10.5, 7, 1, '📈 PROFESSIONAL METRICS\n• Trajectory Efficiency Calculation\n• Goal Achievement Tracking\n• Episode Performance Analysis\n• Multi-Episode Comparison', 
               colors['analysis'])
    
    # Arrow from experience buffer to training
    create_arrow(8, 12, 5.5, 11.5, 'red')
    create_arrow(12, 12, 13.5, 11.5, 'orange')
    
    # 9. EPISODE COMPLETION CHECK
    create_box(7, 9, 6, 1, '✅ EPISODE COMPLETION\n• Max Steps Reached?\n• Goal Achieved?\n• Finalize Trajectory\n• Episode Analysis', 
               colors['control'])
    
    # Arrows to completion check
    create_arrow(5.5, 10.5, 8.5, 10, 'green')
    create_arrow(13.5, 10.5, 11.5, 10, 'orange')
    
    # 10. TRAJECTORY FINALIZATION
    create_box(1, 7.5, 8, 1, '🏭 TRAJECTORY FINALIZATION\n• Store Complete Trajectory with Metadata\n• Calculate Efficiency Score (Direct/Actual Path)\n• Success/Failure Classification\n• Reset Marker Pool for Next Episode', 
               colors['visualization'])
    
    create_box(10, 7.5, 9, 1, '📊 EPISODE ANALYSIS & LOGGING\n• Performance Metrics Summary\n• Success Rate Calculation\n• Professional Trajectory Report\n• Update Curriculum Difficulty', 
               colors['analysis'])
    
    # Arrows from completion to finalization
    create_arrow(8.5, 9, 5, 8.5, 'purple')
    create_arrow(11.5, 9, 14.5, 8.5, 'orange')
    
    # 11. CONTINUE OR TERMINATE
    create_box(6, 6, 8, 1, '🔄 TRAINING CONTINUATION\n• More Episodes Needed?\n• Performance Targets Met?\n• Return to Episode Start or Terminate', 
               colors['training_loop'])
    
    # Arrows to continuation decision
    create_arrow(5, 7.5, 8, 7, 'purple')
    create_arrow(14.5, 7.5, 12, 7, 'orange')
    
    # 12. FINAL ANALYSIS
    create_box(2, 4.5, 16, 1, '🎉 FINAL TRAINING ANALYSIS\n• Complete Trajectory Database Analysis • Success Rate Statistics • Efficiency Metrics Summary\n• Professional Visualization System Report • Industry-Standard Performance Evaluation', 
               colors['analysis'], 10)
    
    # Arrow to final analysis
    create_arrow(10, 6, 10, 5.5, 'orange')
    
    # FEEDBACK LOOP ARROWS
    # Episode loop back
    create_arrow(6, 6.5, 1, 6.5, 'blue', style='->', width=3)
    create_arrow(1, 6.5, 1, 16, 'blue', style='->', width=3)
    create_arrow(1, 16, 2, 16.5, 'blue', style='->', width=3)
    
    # Step loop back
    create_arrow(15, 15, 19, 15, 'green', style='->', width=2)
    create_arrow(19, 15, 19, 13, 'green', style='->', width=2)
    create_arrow(19, 13, 1, 13, 'green', style='->', width=2)
    create_arrow(1, 13, 1, 14.5, 'green', style='->', width=2)
    create_arrow(1, 14.5, 2, 14.5, 'green', style='->', width=2)
    
    # LEGEND
    legend_y = 3.5
    ax.text(1, legend_y, '🎯 LEGEND', fontsize=14, weight='bold')
    
    legend_items = [
        ('Initialization & Setup', colors['initialization']),
        ('Training Loop Control', colors['training_loop']),
        ('Neural Networks', colors['networks']),
        ('Experience Management', colors['experience']),
        ('Trajectory Visualization', colors['visualization']),
        ('Analysis & Metrics', colors['analysis']),
        ('System Control', colors['control'])
    ]
    
    for i, (label, color) in enumerate(legend_items):
        y_pos = legend_y - 0.3 - (i * 0.25)
        create_box(1, y_pos, 2.5, 0.2, label, color, 8)
    
    # FLOW ARROWS LEGEND
    ax.text(10, legend_y, '🔄 FLOW TYPES', fontsize=14, weight='bold')
    
    flow_types = [
        ('Main Training Flow', 'orange'),
        ('Episode Loop', 'blue'), 
        ('Step Loop', 'green'),
        ('Data Flow', 'red'),
        ('Analysis Flow', 'purple')
    ]
    
    for i, (label, color) in enumerate(flow_types):
        y_pos = legend_y - 0.3 - (i * 0.25)
        ax.text(10, y_pos, f'——— {label}', fontsize=10, color=color, weight='bold')
    
    # KEY FEATURES BOX
    features_text = """🏭 KEY INDUSTRY-STANDARD FEATURES:
    ✅ Real-time Trajectory Visualization
    ✅ Professional Goal Marker System  
    ✅ Multi-Episode Performance Analysis
    ✅ Experience Replay with Mini-batching
    ✅ Actor-Critic with Target Networks
    ✅ Curriculum Learning & Goal Management
    ✅ Comprehensive Metrics & Reporting
    ✅ Genesis 0.3.1 Compatible Implementation"""
    
    ax.text(15, 2, features_text, fontsize=9, va='top', 
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('training_control_system_flow_diagram.png', dpi=300, bbox_inches='tight')
    plt.savefig('training_control_system_flow_diagram.pdf', bbox_inches='tight')
    
    print("🎯 Training Control System Flow Diagram Generated!")
    print("   📁 Saved as: training_control_system_flow_diagram.png (High Resolution)")
    print("   📁 Saved as: training_control_system_flow_diagram.pdf (Vector Format)")
    print("   📊 Shows complete training flow with trajectory visualization")
    print("   🏭 Industry-standard system architecture documented")
    
    plt.show()

def create_detailed_component_diagram():
    """Generate detailed component interaction diagram"""
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(8, 11.5, '🏭 DETAILED COMPONENT INTERACTION DIAGRAM', 
            ha='center', va='center', fontsize=16, weight='bold')
    
    # Component colors
    colors = {
        'env': '#E8F4FD',
        'agent': '#D5E8D4', 
        'buffer': '#F8CECC',
        'viz': '#E1D5E7',
        'analysis': '#FFE6CC'
    }
    
    def create_component_box(x, y, width, height, title, details, color):
        # Main box
        box = FancyBboxPatch((x, y), width, height,
                           boxstyle="round,pad=0.1",
                           facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(box)
        
        # Title
        ax.text(x + width/2, y + height - 0.2, title, 
                ha='center', va='center', fontsize=12, weight='bold')
        
        # Details
        ax.text(x + width/2, y + height/2 - 0.1, details, 
                ha='center', va='center', fontsize=9, wrap=True)
    
    # Environment Component
    create_component_box(1, 8, 4, 3, 
                        '🌍 FRANKA ENVIRONMENT',
                        '''• Genesis Physics Simulation
• Robot State Management  
• Joint Control & Limits
• Observation Generation
• Reward Calculation
• Goal Management System''', 
                        colors['env'])
    
    # Agent Component  
    create_component_box(6, 8, 4, 3,
                        '🧠 DDPG AGENT',
                        '''• Actor Network (π)
• Critic Network (Q)
• Target Networks
• Action Generation
• Exploration Noise
• Network Updates''',
                        colors['agent'])
    
    # Experience Buffer
    create_component_box(11, 8, 4, 3,
                        '💾 EXPERIENCE BUFFER', 
                        '''• Replay Memory (10K)
• Experience Storage
• Mini-batch Sampling
• Priority Management
• Data Preprocessing
• Buffer Statistics''',
                        colors['buffer'])
    
    # Trajectory Visualization
    create_component_box(1, 4, 6, 3,
                        '🎯 TRAJECTORY VISUALIZATION',
                        '''• 500 Pre-allocated Markers
• Real-time EE Path Tracking
• Goal Marker System
• Multi-Episode Comparison
• Professional Logging
• Genesis 0.3.1 Compatible''',
                        colors['viz'])
    
    # Performance Analysis
    create_component_box(8, 4, 6, 3,
                        '📊 PERFORMANCE ANALYSIS',
                        '''• Trajectory Efficiency Metrics
• Success Rate Calculation
• Episode Comparison
• Curriculum Progression
• Professional Reporting
• Industry-Standard Analytics''',
                        colors['analysis'])
    
    # Data Flow Arrows with Labels
    def create_labeled_arrow(x1, y1, x2, y2, label, color='black'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color=color, lw=2))
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y + 0.1, label, ha='center', va='bottom', 
                fontsize=8, weight='bold', color=color)
    
    # Environment ↔ Agent
    create_labeled_arrow(5, 9.5, 6, 9.5, 'State', 'blue')
    create_labeled_arrow(6, 9, 5, 9, 'Action', 'green')
    
    # Agent ↔ Buffer  
    create_labeled_arrow(10, 9.5, 11, 9.5, 'Experience', 'red')
    create_labeled_arrow(11, 9, 10, 9, 'Mini-batch', 'purple')
    
    # Environment → Visualization
    create_labeled_arrow(3, 8, 3, 7, 'EE Position', 'orange')
    
    # Environment → Analysis
    create_labeled_arrow(4.5, 8, 8.5, 7, 'Episode Data', 'brown')
    
    # Visualization → Analysis
    create_labeled_arrow(7, 5.5, 8, 5.5, 'Trajectory Data', 'magenta')
    
    plt.tight_layout()
    plt.savefig('component_interaction_diagram.png', dpi=300, bbox_inches='tight')
    print("🔧 Component Interaction Diagram Generated!")
    print("   📁 Saved as: component_interaction_diagram.png")
    
    plt.show()

if __name__ == "__main__":
    print("🏭 GENERATING TRAINING CONTROL SYSTEM DIAGRAMS...")
    print("="*60)
    
    # Generate main flow diagram
    create_training_flow_diagram()
    
    print("\n" + "="*60)
    
    # Generate component diagram
    create_detailed_component_diagram()
    
    print("\n🎉 ALL DIAGRAMS GENERATED SUCCESSFULLY!")
    print("   Use these diagrams for:")
    print("   • System documentation")
    print("   • Architecture presentations") 
    print("   • Technical reviews")
    print("   • Training materials")
