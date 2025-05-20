# -*- coding: utf-8 -*-
"""
Created on Mon May 19 20:17:35 2025

@author: ritwi
"""
import numpy as np
import gymnasium as gym


'Lower limits for joint positions.'
lower_limit_actions = np.array([-0.5, -1.0, -2.0,
                                -0.5, -1.0, -2.0,
                                -0.5, -1.0, -2.0,
                                -0.5, -1.0, -2.0], dtype=np.float32)

'Upper limits for joint positions.'
upper_limit_actions = np.array([0.5,  2.0,  0.0,
                                0.5,  2.0,  0.0,
                                0.5,  2.0,  0.0,
                                0.5,  2.0,  0.0], dtype=np.float32)

'Defining the action space.'
action_space = gym.spaces.Box(low=lower_limit_actions,
                              high=upper_limit_actions,
                              dtype=np.float32)