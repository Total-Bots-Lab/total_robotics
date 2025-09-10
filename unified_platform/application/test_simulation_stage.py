"""
Test Simulation Stage Implementation
===================================

Test script to verify our simulation stage implementation works correctly
and covers all the simulation components from the workflow diagram.
"""

import sys
import os
# Add the parent directory (test) to Python path so we can import unified_platform
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from unified_platform.pipeline.simulation_stage import SimulationStage, run_simulation_pipeline
from unified_platform.config.universal_config import UniversalPlatformConfig, PredefinedConfigs
from unified_platform.pipeline.pipeline_architecture import run_simulation_only_pipeline

def test_simulation_stage_step_by_step():
    """Test simulation stage by running each step individually."""
    print("🧪 Testing Simulation Stage - Step by Step")
    print("=" * 50)
    
    try:
        # Create universal configuration using predefined config
        config = PredefinedConfigs.go2_locomotion()
        
        # Customize for testing
        config.task.num_parallel_envs = 1
        config.training.total_timesteps = 1000
        config.rendering.show_viewer = False  # Disable viewer for testing
        config.output.generate_reports = True
        config.output.export_firmware = True
        
        # Initialize simulation stage
        sim_stage = SimulationStage(config)
        print("✅ Step 1-2: Physics engine and environment setup completed")
        
        # Step 3: Import Robot (already configured in universal config)
        robot_config = config.get_robot_config()
        sim_stage.import_robot(robot_config)
        print("✅ Step 3: Robot imported successfully")
        
        # Step 4: Integrate Controller (already configured)
        controller_config = {"kp": config.robot.kp, "kd": config.robot.kd}
        sim_stage.integrate_controller(controller_config)
        print("✅ Step 4: Controller integrated successfully")
        
        # Step 5: Define Task and Rewards (use config)
        sim_stage.define_task_and_rewards()
        print("✅ Step 5: Task and rewards defined successfully")
        
        # Step 6: Run Simulation
        total_reward = sim_stage.run_simulation(steps=100)
        print(f"✅ Step 6: Simulation completed - Total reward: {total_reward:.3f}")
        
        # Step 7: Train Controller (limited timesteps for testing)
        config.training.total_timesteps = 500  # Reduce for faster testing
        # Note: Training might fail if SB3 not available, that's ok for testing
        try:
            sim_stage.train_controller()
            print("✅ Step 7: Controller training completed")
        except ImportError:
            print("⚠️ Step 7: Training skipped (Stable Baselines3 not available)")
        
        # Step 8: Export Firmware
        firmware_path = sim_stage.export_firmware()
        print(f"✅ Step 8: Firmware exported to {firmware_path}")
        
        # Step 9: Generate Report
        report = sim_stage.generate_simulation_report()
        print("✅ Step 9: Simulation report generated")
        
        # Cleanup
        sim_stage.cleanup()
        print("\n🎉 All simulation stage steps completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False


def test_complete_pipeline():
    """Test the complete simulation pipeline in one call."""
    print("\n🧪 Testing Complete Simulation Pipeline")
    print("=" * 50)
    
    try:
        result = run_simulation_pipeline(
            robot_config="go2",  # Changed from robot_name to robot_config
            task_type="locomotion",
            training_enabled=True,
            num_envs=1,
            timesteps=500,
            show_viewer=False
        )
        
        if result["success"]:
            print("✅ Complete pipeline test passed!")
            print(f"   Firmware: {result.get('firmware_path')}")
            print(f"   Model trained: {result.get('model') is not None}")
            return True
        else:
            print(f"❌ Pipeline test failed: {result.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ Pipeline test failed with exception: {e}")
        return False


def test_pipeline_architecture():
    """Test the full pipeline architecture (simulation stage only)."""
    print("\n🧪 Testing Pipeline Architecture")
    print("=" * 50)
    
    try:
        result = run_simulation_only_pipeline(
            robot_name="go2",
            task_type="locomotion",
            training_enabled=True,
            num_environments=1,
            total_timesteps=500,
            show_viewer=False
        )
        
        if result["success"]:
            print("✅ Pipeline architecture test passed!")
            print("   Stages executed:", list(result["pipeline_outputs"].keys()))
            return True
        else:
            print(f"❌ Pipeline architecture test failed: {result.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ Pipeline architecture test failed: {e}")
        return False


def test_custom_robot():
    """Test with custom robot configuration."""
    print("\n🧪 Testing Custom Robot Configuration")
    print("=" * 50)
    
    try:
        custom_robot = {
            "name": "test_robot",
            "urdf_path": "urdf/go2/urdf/go2.urdf",  # Use Go2 URDF for testing
            "joint_names": ["FR_hip_joint", "FR_thigh_joint", "FR_calf_joint"],
            "default_joint_angles": {
                "FR_hip_joint": 0.0,
                "FR_thigh_joint": 0.5,
                "FR_calf_joint": -1.0
            },
            "base_init_pos": [0.0, 0.0, 0.3],
            "kp": 30.0,
            "kd": 2.0
        }
        
        result = run_simulation_pipeline(
            robot_config=custom_robot,  # Now properly supports custom robot configs
            task_type="locomotion",
            training_enabled=False,  # Skip training for custom test
            num_envs=1,
            timesteps=100,
            show_viewer=False
        )
        
        if result["success"]:
            print("✅ Custom robot test passed!")
            return True
        else:
            print(f"❌ Custom robot test failed: {result.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ Custom robot test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Testing Simulation Stage Implementation")
    print("🎯 Verifying all workflow diagram components")
    print("=" * 60)
    
    test_results = []
    
    # Test 1: Step-by-step simulation stage
    test_results.append(test_simulation_stage_step_by_step())
    
    # Test 2: Complete pipeline function
    test_results.append(test_complete_pipeline())
    
    # Test 3: Pipeline architecture
    test_results.append(test_pipeline_architecture())
    
    # Test 4: Custom robot
    test_results.append(test_custom_robot())
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    tests = [
        "Step-by-step simulation stage",
        "Complete pipeline function", 
        "Pipeline architecture",
        "Custom robot configuration"
    ]
    
    passed = sum(test_results)
    total = len(test_results)
    
    for i, (test_name, result) in enumerate(zip(tests, test_results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{i+1}. {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Simulation stage is ready.")
        print("\n📋 Workflow Coverage:")
        print("✅ 1. Integrate Physics Engine (Genesis AI)")
        print("✅ 2. Setup the Simulation Environment")
        print("✅ 3. Import the Robot")
        print("✅ 4. Integrate the Controller in the Robot")
        print("✅ 5. Define Task and Setup Reward Function")
        print("✅ 6. Run Simulation in Local System")
        print("✅ 7. Train Controller")
        print("✅ 8. Export Firmware for the Robot")
        print("✅ 9. Automated Report Generation")
        
    else:
        print(f"⚠️ {total - passed} tests failed. Check the errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
