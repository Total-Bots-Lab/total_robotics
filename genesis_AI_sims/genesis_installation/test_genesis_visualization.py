#!/usr/bin/env python3
"""
Genesis Visualization Test Script
Quick test to verify Genesis 3D viewer is working properly
"""

import time
import numpy as np
import genesis as gs

def test_genesis_visualization():
    """Test Genesis 3D visualization setup"""
    
    print("🚀 Testing Genesis 3D Visualization...")
    
    # Initialize Genesis
    gs.init(backend=gs.gpu)
    
    # Create enhanced scene with compatible options
    scene = gs.Scene(
        show_viewer=True,
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.0, 2.0, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
            max_FPS=60
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=True,
            world_frame_size=0.5,
            show_link_frame=False
        )
    )
    
    # Add ground plane
    plane = scene.add_entity(
        gs.morphs.Plane(
            pos=(0, 0, 0),
            size=(5.0, 5.0),
            material=gs.materials.Rigid(
                color=(0.8, 0.8, 0.8, 1.0),
                friction=0.8,
                restitution=0.1
            )
        )
    )
    
    # Add multiple lights
    scene.add_entity(
        gs.morphs.Light(
            pos=(2.0, 2.0, 3.0),
            color=(1.0, 1.0, 1.0),
            intensity=1.0,
            type='directional'
        )
    )
    
    scene.add_entity(
        gs.morphs.Light(
            pos=(-2.0, -2.0, 3.0),
            color=(0.8, 0.8, 1.0),
            intensity=0.5,
            type='directional'
        )
    )
    
    scene.add_entity(
        gs.morphs.Light(
            color=(0.4, 0.4, 0.4),
            intensity=0.3,
            type='ambient'
        )
    )
    
    # Add a test sphere
    test_sphere = scene.add_entity(
        gs.morphs.Sphere(
            pos=(0.5, 0.0, 0.5),
            radius=0.1,
            material=gs.materials.Rigid(
                color=(1.0, 0.0, 0.0, 1.0),  # Red sphere
                friction=0.5,
                restitution=0.8
            )
        )
    )
    
    # Add Franka robot
    try:
        robot = gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml')
        franka = scene.add_entity(robot)
        print("✅ Franka robot loaded successfully")
    except Exception as e:
        print(f"⚠️ Could not load Franka robot: {e}")
        print("   Adding a simple box instead...")
        franka = scene.add_entity(
            gs.morphs.Box(
                pos=(0, 0, 0.5),
                size=(0.2, 0.2, 0.5),
                material=gs.materials.Rigid(
                    color=(0.0, 0.8, 0.0, 1.0),  # Green box
                    friction=0.8,
                    restitution=0.1
                )
            )
        )
    
    # Build the scene
    scene.build()
    
    print("🎬 Scene built successfully!")
    print("📹 3D Viewer should now show:")
    print("   • Gray ground plane")
    print("   • Red test sphere")
    print("   • Green robot/box")
    print("   • Multiple lighting sources")
    print("   • World coordinate frame")
    
    # Run simulation for 10 seconds
    print("\n🏃 Running visualization test for 10 seconds...")
    print("   You should see objects in the 3D viewer window")
    print("   Press [t] in viewer to show keyboard instructions")
    
    for i in range(1000):  # 10 seconds at 100 FPS
        scene.step()
        
        # Move the test sphere in a circle
        if hasattr(test_sphere, 'set_pos'):
            t = i * 0.01
            x = 0.5 * np.cos(t)
            y = 0.5 * np.sin(t)
            z = 0.3 + 0.2 * np.sin(2 * t)
            test_sphere.set_pos([x, y, z])
        
        time.sleep(0.01)  # 100 FPS
        
        if i % 100 == 0:
            print(f"   {i//100 + 1}/10 seconds...")
    
    print("✅ Visualization test complete!")
    print("🎯 If you saw objects moving in the 3D viewer, visualization is working!")
    
    # Cleanup
    try:
        gs.destroy()
        print("🧹 Genesis cleanup complete")
    except:
        print("⚠️ Genesis cleanup completed")

if __name__ == "__main__":
    test_genesis_visualization()
