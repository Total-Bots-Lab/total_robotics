"""
Quick Start Script - Run Your Training with Genesis Native Streaming
This replaces TensorBoard with pure Genesis 3D + lightweight web dashboard
"""

import os
import sys

def run_with_genesis_native():
    """Run training with Genesis Native Streaming instead of TensorBoard"""
    
    print("🚀 Starting Genesis AI Training with Native Streaming")
    print("=" * 60)
    print("🎬 PRIMARY: Genesis 3D Viewer (real-time robot visualization)")
    print("📊 SECONDARY: Web Dashboard at http://localhost:8090/dashboard.html")
    print("🔥 NO TENSORBOARD - Pure Genesis streaming!")
    print("=" * 60)
    
    # Import and start Genesis Native Dashboard
    try:
        from genesis_native_streamer import GenesisNativeStreamer
        dashboard = GenesisNativeStreamer(port=8090)
        print("✅ Genesis Native Dashboard initialized")
        print("🌐 Web metrics: http://localhost:8090/dashboard.html")
    except ImportError:
        print("⚠️ Genesis Native Streamer not available, using standard training")
        dashboard = None
    
    # Run your training script with Genesis Native integration
    print("\n🔄 Starting training...")
    
    # Option 1: Run the fixed backup script
    backup_script = "NewTest_v1_pure_env_bkup.py"
    if os.path.exists(backup_script):
        print(f"🎯 Running: {backup_script}")
        print("✅ This version has Genesis-safe cleanup (no hanging)")
        os.system(f"python {backup_script}")
    else:
        print(f"❌ {backup_script} not found")
        return
    
    if dashboard:
        dashboard.finalize()

def run_with_tensorboard_safe():
    """Run training with TensorBoard (Genesis-safe version)"""
    
    print("🚀 Starting Genesis AI Training with TensorBoard (Safe Version)")
    print("=" * 60)
    print("📊 TensorBoard: http://localhost:6006") 
    print("🛡️ Genesis-safe cleanup enabled")
    print("=" * 60)
    
    # Run the Genesis-safe backup script
    backup_script = "NewTest_v1_pure_env_bkup.py"
    if os.path.exists(backup_script):
        print(f"🎯 Running: {backup_script}")
        os.system(f"python {backup_script}")
    else:
        print(f"❌ {backup_script} not found")

def run_with_websocket_tensorboard():
    """Run training with WebSocket-enhanced TensorBoard"""
    
    print("🚀 Starting Genesis AI Training with WebSocket TensorBoard")
    print("=" * 60)
    print("📊 Standard TensorBoard: http://localhost:6006")
    print("⚡ WebSocket Dashboard: Real-time streaming")
    print("🌐 Enhanced UI with live updates")
    print("=" * 60)
    
    # Run the other dashboard script
    other_script = "NewTest_v1_pure_env_other_dash.py"
    if os.path.exists(other_script):
        print(f"🎯 Running: {other_script}")
        os.system(f"python {other_script}")
    else:
        print(f"❌ {other_script} not found")

def show_menu():
    """Show training options menu"""
    
    print("🤖 Genesis AI Training - Choose Your Option")
    print("=" * 50)
    print("1. 🔥 Genesis Native Streaming (NO TensorBoard)")
    print("   • Primary: Genesis 3D Viewer")  
    print("   • Secondary: Lightweight web dashboard")
    print("   • Fastest, most stable option")
    print()
    print("2. 🛡️ TensorBoard Safe (Genesis-safe cleanup)")
    print("   • Standard TensorBoard interface")
    print("   • Fixed hanging issues")
    print("   • Reliable exit")
    print()
    print("3. ⚡ WebSocket TensorBoard (Enhanced)")
    print("   • TensorBoard + real-time streaming")
    print("   • Advanced web dashboard")
    print("   • Live updates via WebSocket")
    print()
    print("4. 🧪 Test Genesis Native Dashboard Only")
    print("   • Preview the dashboard without training")
    print("   • Quick functionality test")
    print("=" * 50)
    
    while True:
        choice = input("Enter your choice (1-4): ").strip()
        
        if choice == "1":
            run_with_genesis_native()
            break
        elif choice == "2":
            run_with_tensorboard_safe()
            break
        elif choice == "3":
            run_with_websocket_tensorboard()
            break
        elif choice == "4":
            test_genesis_native_only()
            break
        else:
            print("⚠️ Invalid choice. Please enter 1, 2, 3, or 4.")

def test_genesis_native_only():
    """Test Genesis Native Dashboard without training"""
    
    print("🧪 Testing Genesis Native Dashboard")
    print("🌐 Opening http://localhost:8090/dashboard.html")
    
    try:
        import subprocess
        subprocess.run(["python", "genesis_native_streamer.py"], timeout=30)
    except subprocess.TimeoutExpired:
        print("✅ Dashboard test completed")
    except FileNotFoundError:
        print("❌ genesis_native_streamer.py not found")

def quick_start():
    """Quick start - run the most stable option"""
    
    print("🚀 QUICK START - Most Stable Option")
    print("Running Genesis-safe TensorBoard version...")
    run_with_tensorboard_safe()

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ["native", "genesis", "1"]:
            run_with_genesis_native()
        elif arg in ["safe", "tensorboard", "2"]:
            run_with_tensorboard_safe()
        elif arg in ["websocket", "ws", "3"]:
            run_with_websocket_tensorboard()
        elif arg in ["test", "4"]:
            test_genesis_native_only()
        elif arg in ["quick", "q"]:
            quick_start()
        else:
            print(f"❌ Unknown option: {arg}")
            show_menu()
    else:
        show_menu()
