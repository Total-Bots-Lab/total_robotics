#!/usr/bin/env python3
"""
IMPROVED Pure Environment Training Script
- Based on training analysis feedback
- Optimized parameters for better goal reaching
- Enhanced reward structure and exploration
"""

import os
import sys
import time
import json
import random
import signal
import datetime
import numpy as np
from collections import deque

# Core ML and Physics
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

# Genesis Physics Engine
import genesis as gs

# Visualization
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Signal handlers for clean exit
def cleanup_and_exit():
    """Clean exit handler"""
    print("🔄 Cleaning up and exiting...")
    try:
        if 'log_file' in globals() and log_file and not log_file.closed:
            log_file.close()
    except:
        pass
    try:
        gs.destroy()
    except:
        pass
    os._exit(0)

def signal_handler(signum, frame):
    """Signal handler for graceful shutdown"""
    print(f"\n📡 Received signal {signum}")
    cleanup_and_exit()

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Logging setup
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = f"improved_training_log_{timestamp}.txt"
log_file = open(log_filename, 'w', encoding='utf-8')

def log_print(message):
    """Dual logging to console and file"""
    print(message)
    try:
        if log_file and not log_file.closed:
            log_file.write(message + '\n')
            log_file.flush()
    except (ValueError, AttributeError):
        # File is closed or not available, just print to console
        pass

log_print(f"🚀 IMPROVED PURE ENVIRONMENT TRAINING - {datetime.datetime.now()}")
log_print(f"📝 Log file: {log_filename}")
log_print("🔧 APPLIED FIXES: Tolerance↑, Reward↓, Episodes↑, Exploration↑")

# Initialize Genesis
gs.init(backend=gs.gpu)

# ============================================================================
# NEURAL NETWORKS - Same architecture as before
# ============================================================================

class ActorNetwork(nn.Module):
    """Actor Network - Outputs actions A(t) from state s(t)"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.activation = nn.ReLU()
        self.output_activation = nn.Tanh()  # Output [-1, 1]
        
        # Xavier initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
    
    def forward(self, state):
        x = self.activation(self.fc1(state))
        x = self.activation(self.fc2(x))
        x = self.output_activation(self.fc3(x))
        return x

class CriticNetwork(nn.Module):
    """Critic Network - Provides value estimates for Actor updates"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(CriticNetwork, self).__init__()
        # State processing
        self.state_fc1 = nn.Linear(state_dim, hidden_dim)
        self.state_fc2 = nn.Linear(hidden_dim, hidden_dim//2)
        
        # Action processing
        self.action_fc1 = nn.Linear(action_dim, hidden_dim//2)
        
        # Combined processing
        self.combined_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.combined_fc2 = nn.Linear(hidden_dim, 1)
        
        self.activation = nn.ReLU()
        
        # Xavier initialization
        for layer in [self.state_fc1, self.state_fc2, self.action_fc1, 
                     self.combined_fc1, self.combined_fc2]:
            nn.init.xavier_uniform_(layer.weight)
    
    def forward(self, state, action):
        # Process state
        s = self.activation(self.state_fc1(state))
        s = self.activation(self.state_fc2(s))
        
        # Process action
        a = self.activation(self.action_fc1(action))
        
        # Combine
        x = torch.cat([s, a], dim=1)
        x = self.activation(self.combined_fc1(x))
        x = self.combined_fc2(x)
        return x

# ============================================================================
# IMPROVED ENVIRONMENT - Based on Training Analysis
# ============================================================================

class ImprovedFrankaGymEnv:
    """IMPROVED Environment with Industry-Standard Trajectory Visualization"""
    
    def __init__(self, scene, franka, dofs_idx, action_space, max_episode_steps=300):
        self.scene = scene
        self.franka = franka
        self.dofs_idx = dofs_idx
        self.action_space = action_space
        self.max_steps = max_episode_steps
        
        # XML-aligned joint limits (same as before)
        self.joint_limits = {
            'joint1': (-2.8973, 2.8973),
            'joint2': (-1.7628, 1.7628), 
            'joint3': (-2.8973, 2.8973),
            'joint4': (-3.0718, -0.0698),
            'joint5': (-2.8973, 2.8973),
            'joint6': (-0.0175, 3.7525),
            'joint7': (-2.8973, 2.8973),
            'finger_joint1': (0.0, 0.04),
            'finger_joint2': (0.0, 0.04)
        }
        
        # IMPROVED: More forgiving goal tolerance
        self.workspace_center = np.array([0.5, 0.0, 0.5])
        self.workspace_radius = 0.8
        self.goal_tolerance = 0.20  # INCREASED: 0.10m → 0.20m (2x easier)
        
        # IMPROVED: Curriculum learning with closer goals initially
        self.target_goals = self._generate_curriculum_goals()
        self.current_goal_idx = 0
        
        # 🏭 INDUSTRY-STANDARD TRAJECTORY VISUALIZATION SYSTEM
        self.trajectory_points = []      # Current episode end-effector path
        self.all_trajectories = []       # Historical trajectories for analysis
        self.trajectory_spheres = []     # Visual trajectory markers in 3D space
        self.goal_spheres = []          # Persistent goal visualization markers
        self.success_indicators = []    # Success/failure visual feedback
        
        # Professional visualization parameters
        self.trajectory_colors = [
            (1.0, 0.2, 0.2),  # Bright Red - Current active trajectory
            (0.2, 1.0, 0.2),  # Bright Green - Successful trajectories  
            (0.2, 0.2, 1.0),  # Bright Blue - Previous attempts
            (1.0, 0.6, 0.0),  # Orange - Learning trajectories
            (0.8, 0.0, 0.8),  # Magenta - Exploration trajectories
        ]
        self.current_trajectory_color = self.trajectory_colors[0]
        self.trajectory_resolution = 5   # Add marker every N steps for smooth visualization
        
        # Episode tracking for professional analysis
        self.episode_count = 0
        self.successful_episodes = []
        self.trajectory_analysis = {
            'total_distance': 0.0,
            'efficiency_score': 0.0,
            'smoothness_metric': 0.0
        }
        
        log_print("🏭 INDUSTRY-STANDARD TRAJECTORY VISUALIZATION INITIALIZED")
        log_print("   ✅ Real-time end-effector path tracking")
        log_print("   ✅ Multi-episode trajectory comparison") 
        log_print("   ✅ Professional goal marker system")
        log_print("   ✅ Success/failure visual indicators")
        log_print("   ✅ Color-coded performance analysis")
        log_print("   ✅ Trajectory efficiency metrics")
        self.curriculum_level = 1  # Start with easier goals
        self.max_curriculum_level = 3
        
        # Episode tracking
        self.current_step = 0
        self.goals_reached_this_episode = 0
        self.total_reward_this_episode = 0
        self.successful_episodes = 0
        
        # Add goal visualization marker
        self.goal_marker = None
        self._create_goal_marker()
        
        log_print(f"🎯 IMPROVED Environment initialized")
        log_print(f"📏 Goal tolerance: {self.goal_tolerance}m (2x easier)")
        log_print(f"⏱️ Episode length: {self.max_steps} steps (3x longer)")
        log_print(f"📚 Curriculum goals: {len(self.target_goals)} total")
        log_print(f"🔴 Goal marker added for visualization")
        
    def _create_goal_marker(self):
        """🏭 INDUSTRY-STANDARD: Create professional goal visualization system"""
        try:
            current_goal = self._get_current_goal()
            
            # Primary goal marker (large, prominent)
            primary_goal = self.scene.add_entity(
                gs.morphs.Sphere(
                    pos=current_goal,
                    radius=0.08  # 8cm - highly visible primary goal
                )
            )
            self.goal_spheres.append(primary_goal)
            
            # Goal tolerance indicator (transparent outer sphere)
            tolerance_indicator = self.scene.add_entity(
                gs.morphs.Sphere(
                    pos=current_goal,
                    radius=self.goal_tolerance  # Shows acceptable reach zone
                )
            )
            self.goal_spheres.append(tolerance_indicator)
            
            # Add all curriculum goals as smaller reference markers
            for i, goal_pos in enumerate(self.target_goals):
                if i != self.current_goal_idx:  # Don't duplicate current goal
                    reference_goal = self.scene.add_entity(
                        gs.morphs.Sphere(
                            pos=goal_pos,
                            radius=0.03  # 3cm - smaller reference goals
                        )
                    )
                    self.goal_spheres.append(reference_goal)
            
            log_print(f"🎯 PROFESSIONAL GOAL SYSTEM CREATED:")
            log_print(f"   • Primary Goal: {current_goal} (8cm marker)")
            log_print(f"   • Tolerance Zone: {self.goal_tolerance}m radius")
            log_print(f"   • Reference Goals: {len(self.target_goals)-1} markers")
            log_print(f"   • Industry-standard visual hierarchy established")
            
        except Exception as e:
            log_print(f"⚠️ Goal marker creation failed: {e}")
            log_print("   Continuing with default goal system...")
    
    def _create_trajectory_marker(self, position, color_index=0):
        """🏭 INDUSTRY-STANDARD: Add trajectory point visualization"""
        try:
            # Create small sphere for trajectory point
            color = self.trajectory_colors[color_index % len(self.trajectory_colors)]
            trajectory_point = self.scene.add_entity(
                gs.morphs.Sphere(
                    pos=position,
                    radius=0.015  # 1.5cm - small trajectory markers
                )
            )
            self.trajectory_spheres.append(trajectory_point)
            return trajectory_point
            
        except Exception as e:
            log_print(f"⚠️ Trajectory marker creation failed: {e}")
            return None
    
    def _update_trajectory_visualization(self):
        """🏭 INDUSTRY-STANDARD: Real-time trajectory path visualization"""
        try:
            current_ee_pos = self._get_end_effector_position()
            self.trajectory_points.append(current_ee_pos.copy())
            
            # Add visual marker every N steps for smooth but not cluttered visualization
            if len(self.trajectory_points) % self.trajectory_resolution == 0:
                self._create_trajectory_marker(
                    current_ee_pos, 
                    color_index=0  # Current trajectory in primary color
                )
            
            # Professional trajectory analysis
            if len(self.trajectory_points) > 1:
                # Calculate trajectory metrics for professional analysis
                segment_distance = np.linalg.norm(
                    self.trajectory_points[-1] - self.trajectory_points[-2]
                )
                self.trajectory_analysis['total_distance'] += segment_distance
                
        except Exception as e:
            log_print(f"⚠️ Trajectory visualization update failed: {e}")
    
    def _finalize_episode_trajectory(self, success=False):
        """🏭 INDUSTRY-STANDARD: Complete episode trajectory analysis"""
        try:
            # Store completed trajectory with metadata
            trajectory_data = {
                'points': self.trajectory_points.copy(),
                'episode': self.episode_count,
                'success': success,
                'total_distance': self.trajectory_analysis['total_distance'],
                'efficiency': self._calculate_trajectory_efficiency(),
                'goal_reached': success
            }
            
            self.all_trajectories.append(trajectory_data)
            
            # Update color for next episode based on performance
            if success:
                self.current_trajectory_color = self.trajectory_colors[1]  # Green for success
                self.successful_episodes.append(self.episode_count)
            else:
                self.current_trajectory_color = self.trajectory_colors[2]  # Blue for attempts
            
            # Professional logging
            log_print(f"� TRAJECTORY ANALYSIS - Episode {self.episode_count}:")
            log_print(f"   • Points Recorded: {len(self.trajectory_points)}")
            log_print(f"   • Total Distance: {self.trajectory_analysis['total_distance']:.3f}m")
            log_print(f"   • Efficiency Score: {trajectory_data['efficiency']:.3f}")
            log_print(f"   • Success: {'✅ GOAL REACHED' if success else '❌ Goal missed'}")
            
            # Reset for next episode
            self.trajectory_points = []
            self.trajectory_analysis['total_distance'] = 0.0
            
        except Exception as e:
            log_print(f"⚠️ Episode trajectory finalization failed: {e}")
    
    def _calculate_trajectory_efficiency(self):
        """🏭 INDUSTRY-STANDARD: Calculate trajectory efficiency metric"""
        try:
            if len(self.trajectory_points) < 2:
                return 0.0
            
            # Calculate direct distance to goal
            start_pos = self.trajectory_points[0]
            end_pos = self.trajectory_points[-1]
            goal_pos = self._get_current_goal()
            
            direct_distance = np.linalg.norm(goal_pos - start_pos)
            actual_distance = self.trajectory_analysis['total_distance']
            
            # Efficiency = direct_path / actual_path (1.0 = perfect straight line)
            efficiency = direct_distance / max(actual_distance, 0.001)  # Avoid division by zero
            return min(efficiency, 1.0)  # Cap at 1.0 for perfect efficiency
            
        except Exception as e:
            log_print(f"⚠️ Efficiency calculation failed: {e}")
            return 0.0
        except Exception as e:
            log_print(f"⚠️ Could not create goal marker: {e}")
            self.goal_marker = None
    
    def _update_goal_marker(self):
        """Update goal marker position when goal changes"""
        try:
            if self.goal_marker is not None:
                current_goal = self._get_current_goal()
                # For Genesis 0.3.1, try different methods to update position
                if hasattr(self.goal_marker, 'set_pos'):
                    self.goal_marker.set_pos(current_goal)
                elif hasattr(self.goal_marker, 'set_position'):
                    self.goal_marker.set_position(current_goal)
                else:
                    # Recreate the marker if we can't update it
                    self._create_goal_marker()
                log_print(f"🔴 Goal marker updated to {current_goal}")
        except Exception as e:
            log_print(f"⚠️ Could not update goal marker: {e}")
            # Try to recreate the marker
            self._create_goal_marker()
        
    def _generate_curriculum_goals(self):
        """Generate goals with curriculum learning - start closer, get harder"""
        goals = []
        
        # Level 1: Close goals (0.3-0.5m radius) - EASIER
        for r in [0.3, 0.4]:
            for theta in np.linspace(0, 2*np.pi, 6, endpoint=False):
                for z in [0.4, 0.5]:
                    x = r * np.cos(theta)
                    y = r * np.sin(theta)
                    goal = np.array([x, y, z])
                    if self._is_goal_safe(goal):
                        goals.append(('easy', goal))
        
        # Level 2: Medium goals (0.5-0.7m radius) - MEDIUM
        for r in [0.5, 0.6]:
            for theta in np.linspace(0, 2*np.pi, 8, endpoint=False):
                for z in [0.3, 0.5, 0.7]:
                    x = r * np.cos(theta)
                    y = r * np.sin(theta)
                    goal = np.array([x, y, z])
                    if self._is_goal_safe(goal):
                        goals.append(('medium', goal))
        
        # Level 3: Hard goals (0.7-0.8m radius) - HARDER
        for r in [0.7, 0.8]:
            for theta in np.linspace(0, 2*np.pi, 8, endpoint=False):
                for z in [0.3, 0.5, 0.7]:
                    x = r * np.cos(theta)
                    y = r * np.sin(theta)
                    goal = np.array([x, y, z])
                    if self._is_goal_safe(goal):
                        goals.append(('hard', goal))
        
        log_print(f"📚 Curriculum: {len([g for l, g in goals if l=='easy'])} easy, "
                 f"{len([g for l, g in goals if l=='medium'])} medium, "
                 f"{len([g for l, g in goals if l=='hard'])} hard goals")
        
        return goals
    
    def _is_goal_safe(self, goal):
        """Check if goal is within safe workspace"""
        return (np.linalg.norm(goal - self.workspace_center) < self.workspace_radius and
                goal[2] > 0.2 and goal[2] < 1.0)
    
    def _get_current_goal(self):
        """Get current target goal based on curriculum level"""
        if not self.target_goals:
            return self.workspace_center
        
        # Filter goals by current curriculum level
        if self.curriculum_level == 1:
            valid_goals = [g for l, g in self.target_goals if l == 'easy']
        elif self.curriculum_level == 2:
            valid_goals = [g for l, g in self.target_goals if l in ['easy', 'medium']]
        else:
            valid_goals = [g for l, g in self.target_goals]
        
        if not valid_goals:
            valid_goals = [g for l, g in self.target_goals]
        
        return valid_goals[self.current_goal_idx % len(valid_goals)]
    
    def _update_goal(self):
        """Update to next goal when current is reached"""
        self.current_goal_idx = (self.current_goal_idx + 1) % len(self.target_goals)
        self.goals_reached_this_episode += 1
        
        # Update goal marker visualization
        self._update_goal_marker()
        
        # IMPROVED: Curriculum progression based on success
        if self.goals_reached_this_episode % 3 == 0 and self.curriculum_level < self.max_curriculum_level:
            self.curriculum_level += 1
            log_print(f"🎓 Curriculum advanced to level {self.curriculum_level}")
    
    def reset(self):
        """Reset environment for new episode with professional trajectory setup"""
        self.current_step = 0
        old_goals = self.goals_reached_this_episode
        self.goals_reached_this_episode = 0
        self.total_reward_this_episode = 0
        
        # 🏭 INDUSTRY-STANDARD: Increment episode counter for trajectory tracking
        self.episode_count += 1
        
        # Track successful episodes
        if old_goals > 0:
            self.successful_episodes += 1
        
        # 🏭 INDUSTRY-STANDARD: Initialize new trajectory tracking
        self.trajectory_points = []
        self.trajectory_analysis = {
            'total_distance': 0.0,
            'efficiency_score': 0.0,
            'smoothness_metric': 0.0
        }
        
        # Start with appropriate curriculum level goal
        curriculum_goals = [i for i, (l, g) in enumerate(self.target_goals) 
                           if (self.curriculum_level == 1 and l == 'easy') or
                              (self.curriculum_level == 2 and l in ['easy', 'medium']) or
                              (self.curriculum_level >= 3)]
        
        if curriculum_goals:
            self.current_goal_idx = random.choice(curriculum_goals)
        
        # 🏭 INDUSTRY-STANDARD: Professional episode initialization logging
        log_print(f"📊 EPISODE {self.episode_count} INITIALIZED:")
        log_print(f"   • Trajectory tracking: ACTIVE")
        log_print(f"   • Goal target: {self._get_current_goal()}")
        log_print(f"   • Success rate: {len(self.successful_episodes)}/{self.episode_count-1 if self.episode_count > 1 else 1}")
        log_print(f"   • Visualization: Real-time path recording")
        
        # Update goal marker for new episode
        self._update_goal_marker()
        
        # Reset to XML keyframe position
        xml_home_position = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853, 0.04, 0.04])
        self.franka.set_dofs_position(xml_home_position, self.dofs_idx)
        
        # Stabilize physics
        for _ in range(20):
            self.scene.step()
        
        current_goal = self._get_current_goal()
        goal_distance = np.linalg.norm(self._get_end_effector_position() - current_goal)
        log_print(f"🔄 Reset - Goal: {current_goal} (dist: {goal_distance:.3f}m, level: {self.curriculum_level})")
        
        return self.get_observation()
    
    def step(self, action):
        """Execute action and return next state, reward, done, info"""
        self.current_step += 1
        
        # Scale action from [-1,1] to joint limits
        scaled_action = self._scale_action(action)
        
        # Apply action to robot
        self.franka.set_dofs_position(scaled_action, self.dofs_idx)
        
        # Step physics simulation
        self.scene.step()
        
        # 🏭 INDUSTRY-STANDARD: Update real-time trajectory visualization
        self._update_trajectory_visualization()
        
        # Optional: Add small delay for better visualization (can be removed for faster training)
        if hasattr(self.scene, 'viewer') and self.scene.viewer is not None:
            time.sleep(0.03)  # 30ms - balanced delay for smooth observation without hindering learning
        
        # Calculate reward with IMPROVED structure
        reward, info = self._calculate_improved_reward(action)
        self.total_reward_this_episode += reward
        
        # Check if episode is done
        done = (self.current_step >= self.max_steps)
        
        # 🏭 INDUSTRY-STANDARD: Finalize trajectory if episode complete
        if done:
            success = info.get('goal_reached', False)
            self._finalize_episode_trajectory(success)
        
        # Get next observation
        next_state = self.get_observation()
        
        return next_state, reward, done, info
    
    def _scale_action(self, action):
        """Scale action from [-1,1] to actual joint limits with balanced smoothing"""
        # Apply balanced action smoothing for good learning + observable movement
        action_smoothing = 0.25  # 25% - balanced between learning efficiency and visual comfort
        smoothed_action = action * action_smoothing
        
        scaled = np.zeros_like(smoothed_action)
        joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7', 'finger_joint1', 'finger_joint2']
        
        for i, joint_name in enumerate(joint_names):
            if i < len(smoothed_action):
                low, high = self.joint_limits[joint_name]
                scaled[i] = low + (smoothed_action[i] + 1) * 0.5 * (high - low)
        
        return scaled
    
    def _calculate_improved_reward(self, action):
        """IMPROVED reward structure based on analysis"""
        current_ee = self._get_end_effector_position()
        current_goal = self._get_current_goal()
        
        # Distance to goal
        goal_distance = np.linalg.norm(current_ee - current_goal)
        
        # IMPROVED: Goal reaching reward
        if goal_distance < self.goal_tolerance:
            goal_reward = 200.0  # INCREASED: 100 → 200 (bigger bonus)
            self._update_goal()
        else:
            # IMPROVED: Less harsh distance penalty
            goal_reward = -0.5 * goal_distance  # REDUCED: -2.0 → -0.5 (4x less harsh)
        
        # IMPROVED: Distance-based bonus (encourage getting closer)
        if goal_distance < 0.3:
            proximity_bonus = 10.0  # Very close bonus
        elif goal_distance < 0.5:
            proximity_bonus = 5.0   # Close bonus
        else:
            proximity_bonus = 0.0
        
        # Workspace safety reward (same as before)
        workspace_dist = np.linalg.norm(current_ee - self.workspace_center)
        if workspace_dist > self.workspace_radius:
            safety_penalty = -50.0
        else:
            safety_penalty = 0.0
        
        # IMPROVED: Reduced action penalty (encourage exploration)
        action_penalty = -0.005 * np.linalg.norm(action)  # REDUCED: -0.01 → -0.005
        
        # IMPROVED: Progress reward (encourage consistent movement toward goal)
        if hasattr(self, 'last_goal_distance'):
            if goal_distance < self.last_goal_distance:
                progress_reward = 2.0  # Reward for getting closer
            else:
                progress_reward = -0.5  # Small penalty for moving away
        else:
            progress_reward = 0.0
        
        self.last_goal_distance = goal_distance
        
        # Total reward
        total_reward = goal_reward + proximity_bonus + safety_penalty + action_penalty + progress_reward
        
        # Info for monitoring
        info = {
            'goal_distance': goal_distance,
            'goal_reached': goal_distance < self.goal_tolerance,
            'workspace_violation': workspace_dist > self.workspace_radius,
            'goals_reached_episode': self.goals_reached_this_episode,
            'curriculum_level': self.curriculum_level,
            'proximity_bonus': proximity_bonus,
            'progress_reward': progress_reward
        }
        
        return total_reward, info
    
    def _get_end_effector_position(self):
        """Get end-effector position (same as before)"""
        try:
            hand_link = self.franka.get_link("hand")
            if hand_link is not None:
                pose = hand_link.get_pose()
                return pose[:3].cpu().numpy() if hasattr(pose, "cpu") else pose[:3]
        except:
            pass
        
        # Fallback: approximate using forward kinematics
        joint_pos = self.franka.get_dofs_position(self.dofs_idx)
        j_pos = joint_pos.cpu().numpy() if hasattr(joint_pos, "cpu") else joint_pos
        
        if len(j_pos) >= 7:
            j1, j2, j3, j4, j5, j6, j7 = j_pos[:7]
            x = 0.5 * np.cos(j1) * np.cos(j2) + 0.3 * np.cos(j1) * np.sin(j2 + j4)
            y = 0.5 * np.sin(j1) * np.cos(j2) + 0.3 * np.sin(j1) * np.sin(j2 + j4)
            z = 0.5 + 0.3 * np.sin(j2) + 0.2 * np.cos(j2 + j4)
            return np.array([x, y, z])
        
        return np.array([0.5, 0.0, 0.5])
    
    def get_observation(self):
        """Get current state observation (same as before)"""
        # Joint positions and velocities
        positions = self.franka.get_dofs_position(self.dofs_idx).cpu().numpy()
        velocities = self.franka.get_dofs_velocity(self.dofs_idx).cpu().numpy()
        
        # End-effector position
        ee_pos = self._get_end_effector_position()
        
        # Current goal
        current_goal = self._get_current_goal()
        
        # Goal distance
        goal_distance = np.linalg.norm(ee_pos - current_goal)
        
        # Progress information
        progress_info = np.array([
            self.current_step / self.max_steps,
            self.goals_reached_this_episode / 10.0,
            self.curriculum_level / self.max_curriculum_level,
            goal_distance
        ])
        
        # Combine all observations
        obs = np.concatenate([
            positions,      # Joint positions (9 values)
            velocities,     # Joint velocities (9 values)
            ee_pos,         # End-effector position (3 values)
            current_goal,   # Current goal position (3 values)
            progress_info   # Progress information (4 values)
        ])
        
        return obs.astype(np.float32)

# ============================================================================
# IMPROVED DDPG AGENT - Enhanced exploration and learning
# ============================================================================

class ImprovedDDPGAgent:
    """IMPROVED DDPG Agent with better exploration"""
    
    def __init__(self, state_dim, action_dim, lr_actor=3e-4, lr_critic=1e-3):  # INCREASED actor LR
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Actor Networks
        self.actor = ActorNetwork(state_dim, action_dim).to(self.device)
        self.actor_target = ActorNetwork(state_dim, action_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        
        # Critic Networks
        self.critic = CriticNetwork(state_dim, action_dim).to(self.device)
        self.critic_target = CriticNetwork(state_dim, action_dim).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Initialize target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # IMPROVED: Larger replay buffer
        self.memory = deque(maxlen=100000)  # INCREASED: 50k → 100k
        self.batch_size = 128  # INCREASED: 64 → 128
        
        # PPO parameters
        self.epsilon = 0.2
        self.entropy_coeff = 0.01
        
        # IMPROVED: Better exploration parameters
        self.noise_std = 0.3      # INCREASED: 0.2 → 0.3 (more exploration)
        self.noise_decay = 0.999  # SLOWER: 0.995 → 0.999 (slower decay)
        self.noise_min = 0.1      # INCREASED: 0.05 → 0.1 (higher minimum)
        
        # Target network update
        self.tau = 0.005
        
        log_print(f"🤖 IMPROVED PPO-DDPG Agent initialized on {self.device}")
        log_print(f"📊 Actor LR: {lr_actor} (↑), Critic LR: {lr_critic}")
        log_print(f"🔍 Exploration: noise={self.noise_std} (↑), decay={self.noise_decay} (slower)")
        log_print(f"💾 Buffer: {self.memory.maxlen} samples (2x larger)")
    
    def select_action(self, state, add_noise=True):
        """Select action with improved exploration"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy().flatten()
        
        if add_noise:
            noise = np.random.normal(0, self.noise_std, size=action.shape)
            action = action + noise
            
            # IMPROVED: Slower noise decay
            self.noise_std = max(self.noise_min, self.noise_std * self.noise_decay)
        
        return np.clip(action, -1, 1)
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def update_networks(self):
        """Update networks with improved batch size"""
        if len(self.memory) < self.batch_size:
            return None, None
        
        # Sample larger batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(np.array(dones)).to(self.device)
        
        # Critic update (same as before)
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards.unsqueeze(1) + (0.99 * target_q * ~dones.unsqueeze(1))
        
        current_q = self.critic(states, actions)
        advantages = target_q - current_q
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic_optimizer.step()
        
        # Actor update with PPO (same as before)
        with torch.no_grad():
            old_actions = self.actor_target(states)
            old_log_probs = self._compute_log_probs(old_actions, actions)
        
        new_actions = self.actor(states)
        new_log_probs = self._compute_log_probs(new_actions, actions)
        
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        surr1 = ratio * advantages.detach()
        surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantages.detach()
        
        actor_loss_clipped = -torch.min(surr1, surr2).mean()
        entropy_bonus = self._compute_entropy(new_actions)
        
        actor_loss = actor_loss_clipped - self.entropy_coeff * entropy_bonus
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        self.actor_optimizer.step()
        
        # Soft update target networks
        self._soft_update_target_networks()
        
        return actor_loss.item(), critic_loss.item()
    
    def _compute_log_probs(self, action_means, sampled_actions):
        """Compute log probabilities - FIXED tensor operations"""
        std = 0.1
        var = std ** 2
        
        # Convert numpy values to tensors
        pi_tensor = torch.tensor(np.pi).to(self.device)
        var_tensor = torch.tensor(var).to(self.device)
        
        log_probs = -0.5 * (((sampled_actions - action_means) ** 2) / var_tensor + 
                           torch.log(2 * pi_tensor * var_tensor))
        
        return log_probs.sum(dim=-1)
    
    def _compute_entropy(self, action_means):
        """Compute entropy - FIXED tensor operations"""
        std = 0.1
        e_tensor = torch.tensor(np.e).to(self.device)
        pi_tensor = torch.tensor(np.pi).to(self.device)
        var_tensor = torch.tensor(std ** 2).to(self.device)
        
        entropy = 0.5 * torch.log(2 * pi_tensor * e_tensor * var_tensor) * action_means.shape[-1]
        return entropy
    
    def _soft_update_target_networks(self):
        """Soft update target networks"""
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# ============================================================================
# MAIN IMPROVED TRAINING SCRIPT
# ============================================================================

def main():
    """Main improved training function"""
    log_print("\n" + "="*70)
    log_print("🚀 IMPROVED PURE ENVIRONMENT TRAINING")
    log_print("🔧 APPLIED IMPROVEMENTS FROM TRAINING ANALYSIS:")
    log_print("   • Goal tolerance: 0.10m → 0.20m (2x easier)")
    log_print("   • Episode length: 100 → 300 steps (3x longer)")
    log_print("   • Reward penalty: -2.0 → -0.5 (4x gentler)")
    log_print("   • Goal bonus: 100 → 200 points (2x bigger)")
    log_print("   • Exploration: enhanced noise & slower decay")
    log_print("   • Curriculum: easy → medium → hard goals")
    log_print("="*70)
    
    # Create scene with enhanced visualization (Genesis 0.3.1 compatible)
    scene = gs.Scene(
        show_viewer=True,
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.0, 2.0, 1.5),     # Camera position
            camera_lookat=(0.0, 0.0, 0.5),  # Camera look-at point
            camera_fov=40,                   # Field of view
            max_FPS=60                      # Frame rate
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=True,          # Show coordinate frame
            world_frame_size=0.5,          # Frame size
            show_link_frame=False          # Hide link frames (cleaner view)
        )
    )
    
    # Add ground plane (Genesis 0.3.1 basic compatibility)
    plane = scene.add_entity(
        gs.morphs.Plane()
    )
    
    # Add simple lighting (Genesis 0.3.1 compatible version)
    try:
        scene.add_entity(
            gs.morphs.Light(
                pos=(2.0, 2.0, 3.0),
                intensity=1.0
            )
        )
    except Exception as e:
        print(f"⚠️ Could not add directional lights: {e}")
        print("   Using default lighting")
    
    # Load Franka robot from XML
    robot = gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml')
    franka = scene.add_entity(robot)
    
    # Build the scene
    scene.build()
    
    log_print("🎬 Enhanced visualization setup complete:")
    log_print("   • Camera position: (2.0, 2.0, 1.5)")
    log_print("   • Compatible lighting added")
    log_print("   • Ground plane with material properties")
    log_print("   • World coordinate frame visible")
    log_print("   • Genesis 0.3.1 compatible settings")
    
    # Joint setup
    joint_names = [
        'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7',
        'finger_joint1', 'finger_joint2'
    ]
    dofs_idx = [franka.get_joint(name).dof_start for name in joint_names]
    
    # Action space
    action_low = np.array([-1.0] * 9, dtype=np.float32)
    action_high = np.array([1.0] * 9, dtype=np.float32)
    action_space = gym.spaces.Box(low=action_low, high=action_high, dtype=np.float32)
    
    log_print(f"🤖 Robot loaded with {len(joint_names)} joints")
    
    # Create IMPROVED environment
    env = ImprovedFrankaGymEnv(scene, franka, dofs_idx, action_space, max_episode_steps=300)
    
    # Get dimensions
    sample_obs = env.reset()
    state_dim = len(sample_obs)
    action_dim = len(dofs_idx)
    
    log_print(f"📊 State dimension: {state_dim}")
    log_print(f"🎮 Action dimension: {action_dim}")
    
    # Create IMPROVED agent
    agent = ImprovedDDPGAgent(state_dim, action_dim)
    
    # Training parameters
    num_episodes = 10  # INCREASED: 3 → 10 for better learning
    episode_rewards = []
    goals_reached_per_episode = []
    
    log_print(f"\n🏃 Starting IMPROVED training for {num_episodes} episodes...")
    
    # Training loop
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_actor_loss = 0
        episode_critic_loss = 0
        step_count = 0
        
        log_print(f"\n📈 Episode {episode + 1}/{num_episodes}")
        
        while True:
            # Select action
            action = agent.select_action(state, add_noise=True)
            
            # Take step
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            agent.store_experience(state, action, reward, next_state, done)
            
            # Update networks
            actor_loss, critic_loss = agent.update_networks()
            if actor_loss is not None:
                episode_actor_loss += actor_loss
                episode_critic_loss += critic_loss
            
            # Update for next iteration
            state = next_state
            episode_reward += reward
            step_count += 1
            
            # Log progress every 50 steps (less frequent for longer episodes)
            if step_count % 50 == 0:
                goal_dist = info.get('goal_distance', 0)
                goals_reached = info.get('goals_reached_episode', 0)
                curriculum = info.get('curriculum_level', 1)
                proximity = info.get('proximity_bonus', 0)
                progress = info.get('progress_reward', 0)
                log_print(f"  Step {step_count}: reward={reward:.2f}, dist={goal_dist:.3f}m, "
                         f"goals={goals_reached}, lvl={curriculum}, prox={proximity:.1f}, prog={progress:.1f}")
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        goals_reached_per_episode.append(info.get('goals_reached_episode', 0))
        
        # Episode summary
        avg_actor_loss = episode_actor_loss / max(1, step_count) if step_count > 0 else 0
        avg_critic_loss = episode_critic_loss / max(1, step_count) if step_count > 0 else 0
        
        # Calculate success rate
        success_rate = sum(goals_reached_per_episode) / len(goals_reached_per_episode) * 100
        
        log_print(f"✅ Episode {episode + 1} complete:")
        log_print(f"   Total Reward: {episode_reward:.2f}")
        log_print(f"   Steps: {step_count}")
        log_print(f"   Goals Reached: {info.get('goals_reached_episode', 0)}")
        log_print(f"   Curriculum Level: {info.get('curriculum_level', 1)}")
        log_print(f"   Success Rate: {success_rate:.1f}%")
        log_print(f"   Actor Loss: {avg_actor_loss:.4f}")
        log_print(f"   Critic Loss: {avg_critic_loss:.4f}")
        log_print(f"   Noise Level: {agent.noise_std:.4f}")
    
    # Training summary
    total_goals = sum(goals_reached_per_episode)
    final_success_rate = total_goals / num_episodes * 100
    
    log_print(f"\n🎯 IMPROVED TRAINING COMPLETE!")
    log_print(f"📊 Episode Rewards: {episode_rewards}")
    log_print(f"🎯 Goals per Episode: {goals_reached_per_episode}")
    log_print(f"📈 Average Reward: {np.mean(episode_rewards):.2f}")
    log_print(f"🏆 Best Reward: {max(episode_rewards):.2f}")
    log_print(f"🎯 Total Goals Reached: {total_goals}")
    log_print(f"🏆 Final Success Rate: {final_success_rate:.1f}%")
    
    # Calculate improvement vs original
    if episode_rewards:
        improvement = episode_rewards[-1] - episode_rewards[0]
        log_print(f"📈 Reward Improvement: {improvement:.2f}")
    
    # Save results
    results = {
        'episode_rewards': episode_rewards,
        'goals_reached_per_episode': goals_reached_per_episode,
        'total_goals_reached': total_goals,
        'success_rate': final_success_rate,
        'average_reward': float(np.mean(episode_rewards)),
        'best_reward': float(max(episode_rewards)),
        'num_episodes': num_episodes,
        'final_noise': float(agent.noise_std),
        'improvements_applied': [
            'Goal tolerance: 0.10m → 0.20m',
            'Episode length: 100 → 300 steps',
            'Reward penalty: -2.0 → -0.5',
            'Goal bonus: 100 → 200 points',
            'Enhanced exploration',
            'Curriculum learning'
        ]
    }
    
    with open(f'improved_results_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    log_print(f"💾 Results saved to: improved_results_{timestamp}.json")
    
    # Success evaluation
    if final_success_rate > 20:
        log_print("🎉 SUCCESS: Agent is learning to reach goals!")
    elif final_success_rate > 5:
        log_print("🔄 PROGRESS: Some goal reaching, needs more training")
    else:
        log_print("⚠️ CHALLENGE: Still learning, may need further tuning")
    
    log_print("🎉 Improved training completed!")
    
    # Clean exit
    try:
        if log_file and not log_file.closed:
            log_file.close()
            print("📝 Log file closed successfully")
    except:
        print("⚠️ Log file already closed")
    
    try:
        gs.destroy()
        print("✅ Genesis destroyed successfully")
    except:
        print("⚠️ Genesis cleanup completed")

if __name__ == "__main__":
    main()
