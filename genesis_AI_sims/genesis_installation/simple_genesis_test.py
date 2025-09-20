#!/usr/bin/env python3
"""
Simple Genesis Visualization Test
Compatible with Genesis 0.3.1
"""

import time
import numpy as np
import genesis as gs

def simple_visualization_test():
    """Simple test for Genesis 3D visualization"""
    
    print("🚀 Testing Genesis 3D Visualization (Simple Version)...")
    
    # Initialize Genesis
    gs.init(backend=gs.gpu)
    
    # Create basic scene with minimal options
    scene = gs.Scene(show_viewer=True)
    
    # Add ground plane (no material parameter)
    scene.add_entity(gs.morphs.Plane())
    
    # Add a simple test sphere (no material parameter)
    test_sphere = scene.add_entity(
        gs.morphs.Sphere(
            pos=(0.5, 0.0, 0.5),
            radius=0.1
        )
    )
    
    # Try to load Franka robot
    try:
        robot = gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml')
        franka = scene.add_entity(robot)
        print("✅ Franka robot loaded successfully")
    except Exception as e:
        print(f"⚠️ Could not load Franka robot: {e}")
        # Add a simple box instead (no material parameter)
        scene.add_entity(
            gs.morphs.Box(
                pos=(0, 0, 0.5),
                size=(0.2, 0.2, 0.5)
            )
        )
    
    # Build the scene
    scene.build()
    
    print("🎬 Scene built successfully!")
    print("📹 You should see:")
    print("   • Ground plane")
    print("   • Sphere")
    print("   • Robot or box")
    
    # Run for 5 seconds
    print("\n🏃 Running test for 5 seconds...")
    
    for i in range(500):  # 5 seconds
        scene.step()
        time.sleep(0.01)
        
        if i % 100 == 0:
            print(f"   {i//100 + 1}/5 seconds...")
    
    print("✅ Test complete!")
    
    # Cleanup
    try:
        gs.destroy()
        print("🧹 Cleanup complete")
    except:
        print("⚠️ Cleanup done")

if __name__ == "__main__":
    simple_visualization_test()
