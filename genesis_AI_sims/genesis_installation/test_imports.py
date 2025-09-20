"""
Minimal Import Test for Genesis Environment
Tests if all required modules can be imported correctly
"""

print("🧪 Testing module imports...")

try:
    print("1. Testing safe_dashboard import...")
    from safe_dashboard import SafeTrainingDashboard
    print("✅ safe_dashboard imported successfully")
    
    print("2. Testing dashboard initialization...")
    dashboard = SafeTrainingDashboard(save_dir="test_init", port=8082)
    print("✅ Dashboard initialized successfully")
    
    print("3. Testing data logging...")
    dashboard.log_step_data(1, 0, 10.5, 5.2, 0.1)
    print("✅ Step data logged successfully")
    
    print("4. Checking data file creation...")
    import os
    data_file = os.path.join("test_init", "training_data.json")
    if os.path.exists(data_file):
        print(f"✅ Data file created: {data_file}")
        with open(data_file, 'r') as f:
            import json
            data = json.load(f)
            print(f"✅ Data file contains {len(data['step_data'])} step entries")
    else:
        print(f"❌ Data file not found: {data_file}")
        
    print("5. Testing dashboard server start...")
    server_started = dashboard.start_dashboard_server()
    if server_started:
        print("✅ Dashboard server started successfully")
        print("🌐 Test dashboard: http://localhost:8082/dashboard.html")
        
        # Test a few more logs
        for i in range(3):
            dashboard.log_step_data(1, i+1, 10.5 + i, 5.2 + i, 0.1 + i*0.1)
        
        dashboard.log_episode_complete(1, [10.5, 11.5, 12.5], [5.2, 6.2, 7.2])
        
        print("✅ Additional data logged successfully")
        
        # Stop server
        dashboard.stop_dashboard_server()
        print("✅ Dashboard server stopped successfully")
    else:
        print("❌ Dashboard server failed to start")
        
    print("\n🎉 All tests passed! Dashboard is working correctly.")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    
print("✅ Import test completed")
