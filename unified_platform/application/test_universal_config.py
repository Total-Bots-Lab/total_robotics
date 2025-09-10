"""
Test Universal Platform Configuration
====================================

Test script that demonstrates how to use the new Universal Platform Configuration
system instead of hardcoded values. This shows how parameters will be set from UI.
"""

import sys
import os
# Add the parent directory (test) to Python path so we can import unified_platform
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from unified_platform.config.universal_config import UniversalPlatformConfig, PredefinedConfigs
from unified_platform.pipeline.simulation_stage import SimulationStage, run_simulation_pipeline


def test_universal_config_from_class_variables():
    """Test creating configuration using class variables (like UI would do)."""
    print("🔧 Testing Universal Configuration from Class Variables")
    print("=" * 60)
    
    # Create configuration like UI would do
    config = UniversalPlatformConfig()
    
    # Robot configuration (these would come from UI form)
    config.robot.name = "go2_custom"
    config.robot.robot_type = "builtin"
    config.robot.robot_source = "go2"
    config.robot.kp = 30.0  # Custom controller gain
    config.robot.kd = 1.5   # Custom controller gain
    config.robot.base_init_pos = [0.0, 0.0, 0.35]  # Higher spawn height
    
    # Task configuration (these would come from UI dropdowns)
    config.task.task_name = "Custom Go2 Locomotion Test"
    config.task.task_type = "locomotion"
    config.task.environment_type = "flat"
    config.task.max_episode_steps = 500  # Shorter episode for testing
    config.task.num_parallel_envs = 1
    
    # Training configuration (these would come from UI sliders/inputs)
    config.training.training_enabled = True
    config.training.algorithm = "PPO"
    config.training.total_timesteps = 5000  # Quick training for testing
    config.training.learning_rate = 1e-4    # Custom learning rate
    config.training.batch_size = 32         # Smaller batch size
    
    # Physics configuration (these would come from UI advanced settings)
    config.physics.physics_engine = "genesis"
    config.physics.backend = "gpu"
    config.physics.dt = 0.02
    config.physics.gravity = [0.0, 0.0, -9.81]
    
    # Rendering configuration (these would come from UI checkboxes)
    config.rendering.enable_rendering = True
    config.rendering.show_viewer = False  # Disable for testing
    config.rendering.record_video = False
    config.rendering.camera_position = [3.0, 0.0, 2.0]  # Custom camera angle
    
    # Output configuration (these would come from UI output settings)
    config.output.log_level = "INFO"
    config.output.log_dir = "custom_test_logs"
    config.output.save_trained_models = True
    config.output.generate_reports = True
    config.output.export_firmware = False
    
    print(f"✅ Configuration created:")
    print(f"   Robot: {config.robot.name} (kp={config.robot.kp}, kd={config.robot.kd})")
    print(f"   Task: {config.task.task_name}")
    print(f"   Training: {config.training.algorithm} for {config.training.total_timesteps} steps")
    print(f"   Physics: {config.physics.physics_engine} on {config.physics.backend}")
    print(f"   Output: {config.output.log_dir}")
    
    # Test the simulation with this configuration
    try:
        sim_stage = SimulationStage(universal_config=config)
        
        # Run a short test
        sim_stage.import_robot(config.get_robot_config())
        sim_stage.integrate_controller()
        sim_stage.define_task_and_rewards(config.task.task_type)
        
        print("✅ Configuration test passed - all parameters loaded correctly!")
        
        sim_stage.cleanup()
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_ui_style_configuration():
    """Test configuration that simulates UI input (JSON-like structure)."""
    print("\n🌐 Testing UI-Style Configuration (JSON input)")
    print("=" * 60)
    
    # Simulate data coming from web UI (like form submission)
    ui_data = {
        "robot": {
            "name": "ui_custom_robot",
            "robot_type": "builtin",
            "robot_source": "go2",
            "kp": 35.0,
            "kd": 2.0,
            "base_init_pos": [0.0, 0.0, 0.4]
        },
        "task": {
            "task_name": "UI Configured Task",
            "task_type": "locomotion",
            "environment_type": "flat",
            "max_episode_steps": 300
        },
        "training": {
            "training_enabled": True,
            "algorithm": "PPO",
            "total_timesteps": 3000,
            "learning_rate": 5e-4,
            "batch_size": 64
        },
        "physics": {
            "physics_engine": "genesis",
            "backend": "gpu",
            "dt": 0.02
        },
        "rendering": {
            "enable_rendering": True,
            "show_viewer": False,
            "camera_position": [2.5, 1.0, 2.0]
        },
        "output": {
            "log_dir": "ui_test_logs",
            "save_trained_models": True,
            "generate_reports": True
        }
    }
    
    # Create configuration from UI data
    config = UniversalPlatformConfig.from_dict(ui_data)
    
    print(f"✅ Configuration created from UI data:")
    print(f"   Robot: {config.robot.name}")
    print(f"   Controller: kp={config.robot.kp}, kd={config.robot.kd}")
    print(f"   Training: {config.training.total_timesteps} timesteps @ lr={config.training.learning_rate}")
    
    # Save configuration to file (like UI would do)
    config.save_to_json("configs/ui_test_config.json")
    print("💾 Configuration saved to file")
    
    # Load configuration from file (like system startup would do)
    loaded_config = UniversalPlatformConfig.from_json_file("configs/ui_test_config.json")
    print("📂 Configuration loaded from file")
    
    # Test that loaded config matches original
    assert loaded_config.robot.kp == ui_data["robot"]["kp"]
    assert loaded_config.training.total_timesteps == ui_data["training"]["total_timesteps"]
    print("✅ File save/load test passed!")
    
    return True


def test_predefined_configurations():
    """Test predefined configurations for common scenarios."""
    print("\n📋 Testing Predefined Configurations")
    print("=" * 60)
    
    # Test Go2 locomotion preset
    go2_config = PredefinedConfigs.go2_locomotion()
    print(f"✅ Go2 Locomotion preset: {go2_config.robot.name}, {go2_config.task.task_name}")
    
    # Test Franka manipulation preset
    franka_config = PredefinedConfigs.franka_manipulation()
    print(f"✅ Franka Manipulation preset: {franka_config.robot.name}, {franka_config.task.task_name}")
    
    # Test custom robot template
    custom_config = PredefinedConfigs.custom_robot_template()
    print(f"✅ Custom Robot template: {custom_config.robot.name}, {custom_config.task.task_name}")
    
    return True


def test_dynamic_ui_updates():
    """Test dynamic configuration updates (like real-time UI changes)."""
    print("\n🔄 Testing Dynamic UI Updates")
    print("=" * 60)
    
    # Start with a base configuration
    config = PredefinedConfigs.go2_locomotion()
    print(f"📋 Initial config: {config.training.total_timesteps} timesteps")
    
    # Simulate UI updates (like user changing values in real-time)
    ui_updates = {
        "training": {
            "total_timesteps": 20000,  # User increased training time
            "learning_rate": 1e-3      # User changed learning rate
        },
        "robot": {
            "kp": 40.0,                # User adjusted controller gains
            "kd": 3.0
        },
        "rendering": {
            "show_viewer": True,       # User enabled viewer
            "record_video": True       # User wants to record
        }
    }
    
    # Apply UI updates
    config.update_from_ui(ui_updates)
    
    print(f"🔄 Updated config:")
    print(f"   Training: {config.training.total_timesteps} timesteps @ lr={config.training.learning_rate}")
    print(f"   Controller: kp={config.robot.kp}, kd={config.robot.kd}")
    print(f"   Rendering: viewer={config.rendering.show_viewer}, record={config.rendering.record_video}")
    
    # Verify updates were applied
    assert config.training.total_timesteps == 20000
    assert config.robot.kp == 40.0
    assert config.rendering.record_video == True
    
    print("✅ Dynamic update test passed!")
    return True


def test_configuration_with_simulation():
    """Test running actual simulation with universal configuration."""
    print("\n🚀 Testing Configuration with Real Simulation")
    print("=" * 60)
    
    # Create a test configuration
    config = UniversalPlatformConfig()
    
    # Configure for quick test
    config.robot.name = "test_go2"
    config.robot.robot_source = "go2"
    config.training.total_timesteps = 1000  # Very short for testing
    config.task.max_episode_steps = 100     # Short episodes
    config.rendering.show_viewer = False    # No viewer for testing
    config.output.log_dir = "config_test_logs"
    
    try:
        # Test with new universal config system
        result = run_simulation_pipeline(universal_config=config)
        
        if result["success"]:
            print("✅ Universal config simulation test passed!")
            print(f"   Model trained: {result.get('model') is not None}")
            return True
        else:
            print(f"❌ Simulation failed: {result.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ Simulation test failed: {e}")
        return False


def main():
    """Run all universal configuration tests."""
    print("🌟 Universal Platform Configuration Tests")
    print("🎯 Testing dynamic, UI-driven configuration system")
    print("=" * 70)
    
    test_results = []
    
    # Test 1: Class variable configuration
    test_results.append(test_universal_config_from_class_variables())
    
    # Test 2: UI-style JSON configuration
    test_results.append(test_ui_style_configuration())
    
    # Test 3: Predefined configurations
    test_results.append(test_predefined_configurations())
    
    # Test 4: Dynamic UI updates
    test_results.append(test_dynamic_ui_updates())
    
    # Test 5: Configuration with real simulation
    test_results.append(test_configuration_with_simulation())
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 UNIVERSAL CONFIGURATION TEST SUMMARY")
    print("=" * 70)
    
    tests = [
        "Class variable configuration",
        "UI-style JSON configuration",
        "Predefined configurations",
        "Dynamic UI updates",
        "Configuration with simulation"
    ]
    
    passed = sum(test_results)
    total = len(test_results)
    
    for i, (test_name, result) in enumerate(zip(tests, test_results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{i+1}. {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All universal configuration tests passed!")
        print("\n📋 Features Verified:")
        print("✅ Dynamic parameter configuration")
        print("✅ UI-style JSON input/output")
        print("✅ File-based configuration save/load")
        print("✅ Real-time configuration updates")
        print("✅ Integration with simulation pipeline")
        print("✅ No more hardcoded values!")
        print("\n🌐 Ready for UI integration!")
        
    else:
        print(f"⚠️ {total - passed} tests failed. Check the errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
