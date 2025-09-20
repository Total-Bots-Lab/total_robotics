#!/usr/bin/env python3
"""
Corrected Pure Environment Training Script
- Follows JPEG diagram RL Package flow exactly
- Aligned with XML robot configuration
- Pure environment learning (no reference trajectories)
- Complete DDPG implementation with Actor-Critic networks
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
log_filename = f"corrected_training_log_{timestamp}.txt"
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

log_print(f"🚀 CORRECTED PURE ENVIRONMENT TRAINING - {datetime.datetime.now()}")
log_print(f"📝 Log file: {log_filename}")

# Initialize Genesis
gs.init(backend=gs.gpu)

# ============================================================================
# NEURAL NETWORKS - Following JPEG Diagram RL Package Structure
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
# ENVIRONMENT - Pure Learning with XML-Aligned Configuration
# ============================================================================

class FrankaGymEnv:
    """Pure Environment Learning Environment - XML Aligned"""
    
    def __init__(self, scene, franka, dofs_idx, action_space, max_episode_steps=200):
        self.scene = scene
        self.franka = franka
        self.dofs_idx = dofs_idx
        self.action_space = action_space
        self.max_steps = max_episode_steps
        
        # XML-aligned joint limits (from panda.xml actuator definitions)
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
        
        # Pure environment learning parameters
        self.workspace_center = np.array([0.5, 0.0, 0.5])  # Center of workspace
        self.workspace_radius = 0.8  # Safe workspace radius
        self.goal_tolerance = 0.10  # 10cm tolerance for goal reaching
        
        # Dynamic goal generation
        self.target_goals = self._generate_workspace_goals()
        self.current_goal_idx = 0
        
        # Episode tracking
        self.current_step = 0
        self.goals_reached_this_episode = 0
        self.total_reward_this_episode = 0
        
        # Curriculum learning
        self.difficulty_level = 1  # Start easy
        self.max_difficulty = 3
        
        log_print(f"🎯 Environment initialized with {len(self.target_goals)} goals")
        log_print(f"📏 Workspace: center={self.workspace_center}, radius={self.workspace_radius}")
        
    def _generate_workspace_goals(self):
        """Generate reachable goals within robot workspace"""
        goals = []
        
        # Generate goals in spherical coordinates around robot base
        for r in [0.4, 0.6, 0.8]:  # Different distances
            for theta in np.linspace(0, 2*np.pi, 8, endpoint=False):  # 8 angles
                for z in [0.3, 0.5, 0.7]:  # Different heights
                    x = r * np.cos(theta)
                    y = r * np.sin(theta)
                    
                    # Check if goal is within safe workspace
                    goal = np.array([x, y, z])
                    if (np.linalg.norm(goal - self.workspace_center) < self.workspace_radius and
                        z > 0.2 and z < 1.0):  # Height constraints
                        goals.append(goal)
        
        log_print(f"📍 Generated {len(goals)} reachable goals")
        return goals
    
    def _get_current_goal(self):
        """Get current target goal"""
        if not self.target_goals:
            return self.workspace_center
        return self.target_goals[self.current_goal_idx % len(self.target_goals)]
    
    def _update_goal(self):
        """Update to next goal when current is reached"""
        self.current_goal_idx = (self.current_goal_idx + 1) % len(self.target_goals)
        self.goals_reached_this_episode += 1
        
        # Increase difficulty periodically
        if self.goals_reached_this_episode % 5 == 0 and self.difficulty_level < self.max_difficulty:
            self.difficulty_level += 1
            log_print(f"🎯 Difficulty increased to level {self.difficulty_level}")
    
    def reset(self):
        """Reset environment for new episode"""
        self.current_step = 0
        self.goals_reached_this_episode = 0
        self.total_reward_this_episode = 0
        self.current_goal_idx = random.randint(0, len(self.target_goals) - 1)
        
        # Reset to XML keyframe position: "0 0 0 -1.57079 0 1.57079 -0.7853 0.04 0.04"
        xml_home_position = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853, 0.04, 0.04])
        self.franka.set_dofs_position(xml_home_position, self.dofs_idx)
        
        # Stabilize physics
        for _ in range(20):
            self.scene.step()
        
        log_print(f"🔄 Environment reset - Goal: {self._get_current_goal()}")
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
        
        # Calculate reward
        reward, info = self._calculate_reward(action)
        self.total_reward_this_episode += reward
        
        # Check if episode is done
        done = (self.current_step >= self.max_steps)
        
        # Get next observation
        next_state = self.get_observation()
        
        return next_state, reward, done, info
    
    def _scale_action(self, action):
        """Scale action from [-1,1] to actual joint limits"""
        scaled = np.zeros_like(action)
        joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7', 'finger_joint1', 'finger_joint2']
        
        for i, joint_name in enumerate(joint_names):
            if i < len(action):
                low, high = self.joint_limits[joint_name]
                scaled[i] = low + (action[i] + 1) * 0.5 * (high - low)
        
        return scaled
    
    def _calculate_reward(self, action):
        """Calculate reward based on pure environment feedback"""
        current_ee = self._get_end_effector_position()
        current_goal = self._get_current_goal()
        
        # Distance to goal
        goal_distance = np.linalg.norm(current_ee - current_goal)
        
        # Goal reaching reward (primary objective)
        if goal_distance < self.goal_tolerance:
            goal_reward = 100.0  # Large bonus for reaching goal
            self._update_goal()  # Move to next goal
        else:
            goal_reward = -2.0 * goal_distance  # Penalty proportional to distance
        
        # Workspace safety reward
        workspace_dist = np.linalg.norm(current_ee - self.workspace_center)
        if workspace_dist > self.workspace_radius:
            safety_penalty = -50.0  # Large penalty for leaving workspace
        else:
            safety_penalty = 0.0
        
        # Action smoothness (energy efficiency)
        action_penalty = -0.01 * np.linalg.norm(action)
        
        # Total reward
        total_reward = goal_reward + safety_penalty + action_penalty
        
        # Info for monitoring
        info = {
            'goal_distance': goal_distance,
            'goal_reached': goal_distance < self.goal_tolerance,
            'workspace_violation': workspace_dist > self.workspace_radius,
            'goals_reached_episode': self.goals_reached_this_episode
        }
        
        return total_reward, info
    
    def _get_end_effector_position(self):
        """Get end-effector position from hand link"""
        try:
            # Try to get hand link pose (from XML: body name="hand")
            hand_link = self.franka.get_link("hand")
            if hand_link is not None:
                pose = hand_link.get_pose()
                return pose[:3].cpu().numpy() if hasattr(pose, "cpu") else pose[:3]
        except:
            pass
        
        # Fallback: approximate using forward kinematics
        joint_pos = self.franka.get_dofs_position(self.dofs_idx)
        j_pos = joint_pos.cpu().numpy() if hasattr(joint_pos, "cpu") else joint_pos
        
        # Simple approximation for Franka end-effector
        if len(j_pos) >= 7:
            j1, j2, j3, j4, j5, j6, j7 = j_pos[:7]
            x = 0.5 * np.cos(j1) * np.cos(j2) + 0.3 * np.cos(j1) * np.sin(j2 + j4)
            y = 0.5 * np.sin(j1) * np.cos(j2) + 0.3 * np.sin(j1) * np.sin(j2 + j4)
            z = 0.5 + 0.3 * np.sin(j2) + 0.2 * np.cos(j2 + j4)
            return np.array([x, y, z])
        
        return np.array([0.5, 0.0, 0.5])  # Default safe position
    
    def get_observation(self):
        """Get current state observation"""
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
            self.current_step / self.max_steps,  # Episode progress
            self.goals_reached_this_episode / 10.0,  # Goals reached (normalized)
            self.difficulty_level / self.max_difficulty,  # Difficulty level
            goal_distance  # Distance to goal
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
# DDPG AGENT - Following JPEG Diagram RL Package Flow
# ============================================================================

class DDPGAgent:
    """DDPG Agent with PPO-style Lclip following JPEG diagram"""
    
    def __init__(self, state_dim, action_dim, lr_actor=1e-4, lr_critic=1e-3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Actor Networks (following JPEG diagram)
        self.actor = ActorNetwork(state_dim, action_dim).to(self.device)
        self.actor_target = ActorNetwork(state_dim, action_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        
        # Critic Networks (following JPEG diagram)
        self.critic = CriticNetwork(state_dim, action_dim).to(self.device)
        self.critic_target = CriticNetwork(state_dim, action_dim).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Initialize target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Experience Replay Buffer (following JPEG diagram)
        self.memory = deque(maxlen=50000)
        self.batch_size = 64
        
        # PPO parameters for Lclip
        self.epsilon = 0.2  # PPO clipping parameter
        self.entropy_coeff = 0.01  # Entropy coefficient
        
        # Exploration noise
        self.noise_std = 0.2
        self.noise_decay = 0.995
        self.noise_min = 0.05
        
        # Target network update
        self.tau = 0.005  # Soft update parameter
        
        log_print(f"🤖 PPO-DDPG Agent initialized on {self.device}")
        log_print(f"📊 Actor LR: {lr_actor}, Critic LR: {lr_critic}")
        log_print(f"🎯 PPO Clip ε: {self.epsilon}, Entropy: {self.entropy_coeff}")
    
    def select_action(self, state, add_noise=True):
        """Select action using actor network with exploration noise"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy().flatten()
        
        if add_noise:
            noise = np.random.normal(0, self.noise_std, size=action.shape)
            action = action + noise
            
            # Decay noise
            self.noise_std = max(self.noise_min, self.noise_std * self.noise_decay)
        
        return np.clip(action, -1, 1)
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def update_networks(self):
        """Update Actor and Critic networks following JPEG diagram with Lclip"""
        if len(self.memory) < self.batch_size:
            return None, None
        
        # Sample batch from experience replay
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(np.array(dones)).to(self.device)
        
        # ============================================================================
        # CRITIC NETWORK UPDATE - A(t) backpropagation
        # ============================================================================
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards.unsqueeze(1) + (0.99 * target_q * ~dones.unsqueeze(1))
        
        current_q = self.critic(states, actions)
        
        # A(t) = Q_target - Q_current (Advantage estimation)
        advantages = target_q - current_q
        
        # Critic loss for value function approximation
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        # BACKPROPAGATION: ∂critic_loss/∂critic_params
        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)  # Retain graph for actor update
        self.critic_optimizer.step()
        
        # ============================================================================
        # ACTOR NETWORK UPDATE - Lclip(θ,t) backpropagation (PPO-style)
        # ============================================================================
        
        # Get old action probabilities (from replay buffer)
        with torch.no_grad():
            old_actions = self.actor_target(states)
            old_log_probs = self._compute_log_probs(old_actions, actions)
        
        # Get new action probabilities from current actor
        new_actions = self.actor(states)
        new_log_probs = self._compute_log_probs(new_actions, actions)
        
        # Probability ratio: π_θ(a|s) / π_θ_old(a|s)
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # PPO Clipped Objective - Lclip(θ,t)
        epsilon = 0.2  # PPO clipping parameter
        
        # Surrogate loss terms
        surr1 = ratio * advantages.detach()  # Unclipped objective
        surr2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages.detach()  # Clipped objective
        
        # Lclip = E[min(surr1, surr2)] - PPO clipped loss
        actor_loss_clipped = -torch.min(surr1, surr2).mean()
        
        # Add entropy bonus for exploration
        entropy_bonus = self._compute_entropy(new_actions)
        entropy_coeff = 0.01
        
        # Final actor loss: Lclip - entropy_bonus
        actor_loss = actor_loss_clipped - entropy_coeff * entropy_bonus
        
        # BACKPROPAGATION: ∂Lclip/∂actor_params
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        
        self.actor_optimizer.step()
        
        # Soft update target networks
        self._soft_update_target_networks()
        
        return actor_loss.item(), critic_loss.item()
    
    def _compute_log_probs(self, action_means, sampled_actions):
        """Compute log probabilities for continuous actions (Gaussian policy)"""
        # Assume Gaussian policy with fixed standard deviation
        std = 0.1
        var = std ** 2
        
        # Convert constants to tensors on the same device
        pi_tensor = torch.tensor(np.pi, device=self.device)
        var_tensor = torch.tensor(var, device=self.device)
        
        # Log probability of Gaussian: log(p(a)) = -0.5 * ((a - μ)² / σ² + log(2πσ²))
        log_probs = -0.5 * (((sampled_actions - action_means) ** 2) / var_tensor + 
                           torch.log(2 * pi_tensor * var_tensor))
        
        return log_probs.sum(dim=-1)  # Sum over action dimensions
    
    def _compute_entropy(self, action_means):
        """Compute entropy for continuous Gaussian policy"""
        # Entropy of multivariate Gaussian: H = 0.5 * log(2πeσ²) * dim
        std = 0.1
        pi_tensor = torch.tensor(np.pi, device=self.device)
        e_tensor = torch.tensor(np.e, device=self.device)
        std_tensor = torch.tensor(std, device=self.device)
        
        entropy = 0.5 * torch.log(2 * pi_tensor * e_tensor * (std_tensor ** 2)) * action_means.shape[-1]
        return entropy
    
    def _soft_update_target_networks(self):
        """Soft update target networks"""
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# ============================================================================
# MAIN TRAINING SCRIPT - TEST VERSION
# ============================================================================

def main():
    """Main training function"""
    log_print("\n" + "="*60)
    log_print("🚀 CORRECTED PURE ENVIRONMENT TRAINING - TEST VERSION")
    log_print("="*60)
    
    # Create scene
    scene = gs.Scene(show_viewer=True)
    scene.add_entity(gs.morphs.Plane())
    
    # Load Franka robot from XML
    robot = gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml')
    franka = scene.add_entity(robot)
    scene.build()
    
    # Joint setup (from XML)
    joint_names = [
        'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7',
        'finger_joint1', 'finger_joint2'
    ]
    dofs_idx = [franka.get_joint(name).dof_start for name in joint_names]
    
    # Action space (XML-aligned)
    action_low = np.array([-1.0] * 9, dtype=np.float32)  # Normalized [-1, 1]
    action_high = np.array([1.0] * 9, dtype=np.float32)
    action_space = gym.spaces.Box(low=action_low, high=action_high, dtype=np.float32)
    
    log_print(f"🤖 Robot loaded with {len(joint_names)} joints")
    log_print(f"🎯 Action space: {action_space.shape}")
    
    # Create environment
    env = FrankaGymEnv(scene, franka, dofs_idx, action_space, max_episode_steps=100)  # Shorter for testing
    
    # Get dimensions
    sample_obs = env.reset()
    state_dim = len(sample_obs)
    action_dim = len(dofs_idx)
    
    log_print(f"📊 State dimension: {state_dim}")
    log_print(f"🎮 Action dimension: {action_dim}")
    
    # Create DDPG agent
    agent = DDPGAgent(state_dim, action_dim)
    
    # Training parameters (TEST VERSION)
    num_episodes = 3  # Just 3 episodes for testing
    episode_rewards = []
    
    log_print(f"\n🏃 Starting training for {num_episodes} episodes...")
    
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
            
            # Log progress
            if step_count % 20 == 0:
                goal_dist = info.get('goal_distance', 0)
                goals_reached = info.get('goals_reached_episode', 0)
                log_print(f"  Step {step_count}: reward={reward:.2f}, goal_dist={goal_dist:.3f}, goals={goals_reached}")
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        
        # Episode summary
        avg_actor_loss = episode_actor_loss / max(1, step_count) if step_count > 0 else 0
        avg_critic_loss = episode_critic_loss / max(1, step_count) if step_count > 0 else 0
        
        log_print(f"✅ Episode {episode + 1} complete:")
        log_print(f"   Total Reward: {episode_reward:.2f}")
        log_print(f"   Steps: {step_count}")
        log_print(f"   Goals Reached: {info.get('goals_reached_episode', 0)}")
        log_print(f"   Actor Loss: {avg_actor_loss:.4f}")
        log_print(f"   Critic Loss: {avg_critic_loss:.4f}")
        log_print(f"   Noise Level: {agent.noise_std:.4f}")
    
    # Training summary
    log_print(f"\n🎯 TRAINING COMPLETE!")
    log_print(f"📊 Episode Rewards: {episode_rewards}")
    log_print(f"📈 Average Reward: {np.mean(episode_rewards):.2f}")
    log_print(f"🏆 Best Reward: {max(episode_rewards):.2f}")
    
    # Save results
    results = {
        'episode_rewards': episode_rewards,
        'average_reward': float(np.mean(episode_rewards)),
        'best_reward': float(max(episode_rewards)),
        'num_episodes': num_episodes,
        'final_noise': float(agent.noise_std)
    }
    
    with open(f'test_results_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    log_print(f"💾 Results saved to: test_results_{timestamp}.json")
    
    # Clean exit
    log_print("🎉 Test completed successfully!")
    
    # Close log file safely
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
