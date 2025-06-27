# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 01:17:24 2025

@author: ritwi
"""

"""
Setup genesis
"""

import genesis as gs

def Genesis_Simulator(self):
    
    # Initializes Genesis with the CPU backend.(if not gs._initialized:)
    if not gs._initialized:
        gs.init(backend=gs.cpu) # * Need to define cpu and gpu mode latter
    
    



    
    # create scene
    self.scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.02, # * Need to define dt latter from outside
                                          gravity=(0, 0, -10),
                                          substeps=2,),
        
        show_FPS=False,
        
        show_viewer=True, # * Need to define latter from outside
  
      
        # **** Need to figure out all these
        
        # viewer_options=gs.options.ViewerOptions(
        #     max_FPS=int(0.5 / self.dt),
        #     camera_pos=(2.0, 0.0, 2.5),
        #     camera_lookat=(0.0, 0.0, 0.5),
        #     camera_fov=40,
        # ),
        # vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(1))),
        # rigid_options=gs.options.RigidOptions(
        #     dt=self.dt,
        #     constraint_solver=gs.constraint_solver.Newton,
        #     enable_collision=True,
        #     enable_joint_limit=True,
        # ),
        
        
        )
    
 
    
    
    
    
    
    
    # add plain
    # * Need to figure out the more advanced Planes
    self.scene.add_entity(gs.morphs.Plane()) # Adding the flat default plane to the scene for now.
    
    # This plane is similar to the default but take way longer time cause it call the plane from the urdf file
    #self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True)) 
    
    
    
    
    
    
    
    # add robot
    'Integrate the Go2 Robot using xml and Add an entity to the scene.'
    self.robot = self.scene.add_entity(
        gs.morphs.MJCF(file="xml/Unitree_Go2/go2.xml")
        )
    

    # * Need to figure out all the advanced settings
    
    # self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=gs.device)
    # self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=gs.device)
    # self.inv_base_init_quat = inv_quat(self.base_init_quat)
    # self.robot = self.scene.add_entity(
    #     gs.morphs.URDF(
    #         file="urdf/go2/urdf/go2.urdf",
    #         pos=self.base_init_pos.cpu().numpy(),
    #         quat=self.base_init_quat.cpu().numpy(),
    #     ),
    # )
    
    
    
    
    
    
    
    
    
    'Builds the scene.'
    self.scene.build(n_envs=1) #Starting with single genesis env
    
    # * Need to figure out all the advanced settings to multi vectorized the environment
    #self.scene.build(n_envs=num_envs)
    
    