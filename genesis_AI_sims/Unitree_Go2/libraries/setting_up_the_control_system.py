# -*- coding: utf-8 -*-
"""
Created on Sun Jul 27 04:22:59 2025

@author: ritwi
"""

import numpy as np


def setting_up_the_control_system(self):
    
    # Declaring the joint names in the desired order
    print('Declaring the joint names...')
    self.joint_names = [
        'FL_hip_joint',
        'FR_hip_joint',
        'RL_hip_joint',
        'RR_hip_joint',
        'FL_thigh_joint',
        'FR_thigh_joint',
        'RL_thigh_joint',
        'RR_thigh_joint',
        'FL_calf_joint',
        'FR_calf_joint',
        'RL_calf_joint',
        'RR_calf_joint',
        ]
    
    print('Joint Names\n', self.joint_names)
    
    # Defining the Joint Name IDs
    print('Defining the Motor Degree of Freedom IDs...')
    self.motors_dof_idx = [self.robot.get_joint(name).dof_start for name in self.joint_names]
    print('Motor DoF IDs:\n',self.motors_dof_idx)
    
    
    
    'Setting up the primary PI controller'
    print('Setting up the Primary (PID) Controller...')
    
    # Setting up the proportional (kp) and derivative (kd) gain values
    print('Setting up the PID gains...')
    self.kp= 20.0
    self.kd= 0.5
    
    # Create an array of kp values for each motor degree of freedom (DOF)
    self.kp_array = np.full(len(self.motors_dof_idx), self.kp)
    # Print the kp array to verify its values
    print('Proportional Gains Array:\n',self.kp_array)
    
    # Create an array of kd values for each motor degree of freedom (DOF)
    self.kd_array = np.full(len(self.motors_dof_idx), self.kd)
    # Print the kd array to verify its values
    print('Derivative Gains Array:\n',self.kd_array)
    
    print('\nSetting Proportional and Derivative gains for each DOF...')
    # Set the proportional gains (Kp) for the specified motor joints (motors_dof_idx)
    self.robot.set_dofs_kp(self.kp_array, self.motors_dof_idx)
    print('\nThe Proportional gains of the DOFs:')
    print(self.robot.get_dofs_kp(self.motors_dof_idx))
    
    # Set the derivative gains (Kv/Kd) for the specified motor joints (motors_dof_idx)
    self.robot.set_dofs_kv(self.kd_array, self.motors_dof_idx)
    print('\nThe Proportional gains of the DOFs:')
    print(self.robot.get_dofs_kv(self.motors_dof_idx))
    
    
    
    
    
    
    
    
    
    
    
    
    
    