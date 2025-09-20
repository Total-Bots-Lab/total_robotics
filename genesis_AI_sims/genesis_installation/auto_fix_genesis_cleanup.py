"""
Auto-Fix Script for Genesis gs.destroy() Hanging Issue
This script will automatically fix the cleanup section in your NewTest_v1_pure_env_bkup.py
"""

import os
import re

def auto_fix_genesis_cleanup(file_path):
    """
    Automatically fix the Genesis cleanup section to prevent hanging
    """
    try:
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Define the new cleanup code
        new_cleanup = '''# Genesis-safe cleanup with timeout protection - PREVENTS HANGING
print("🔄 Starting Genesis-safe cleanup...")

# STEP 1: Stop TensorBoard processes FIRST (critical for preventing hang)
if 'tensorboard_logger' in globals() and tensorboard_logger is not None:
    try:
        print("🔄 Stopping TensorBoard safely...")
        
        # Stop TensorBoard manager
        if hasattr(tensorboard_logger, 'tb_manager') and tensorboard_logger.tb_manager:
            tensorboard_logger.tb_manager.stop_tensorboard(silent=False)
        
        # Close TensorBoard writer
        if hasattr(tensorboard_logger, 'writer') and tensorboard_logger.writer:
            tensorboard_logger.writer.close()
        
        # Finalize logger if available
        if hasattr(tensorboard_logger, 'finalize'):
            tensorboard_logger.finalize()
            
        print("✅ TensorBoard stopped successfully")
    except Exception as e:
        print(f"⚠️ TensorBoard stop warning: {e}")

# STEP 2: Force kill any remaining TensorBoard processes
try:
    import subprocess
    print("🔄 Killing remaining TensorBoard processes...")
    
    # Kill TensorBoard executable
    subprocess.run(['taskkill', '/f', '/im', 'tensorboard.exe'], 
                  capture_output=True, timeout=3)
    
    # Kill Python processes running TensorBoard
    subprocess.run(['taskkill', '/f', '/fi', 'WINDOWTITLE eq *tensorboard*'], 
                  capture_output=True, timeout=3)
    
    print("✅ TensorBoard processes terminated")
except Exception as e:
    print(f"⚠️ Process cleanup warning: {e}")

# STEP 3: Brief pause to ensure complete process cleanup
print("🔄 Waiting for process cleanup completion...")
time.sleep(3)

# STEP 4: Genesis destroy with timeout protection (prevents infinite hang)
def safe_genesis_destroy():
    """Safe Genesis destroy with error handling"""
    try:
        print("🔄 Destroying Genesis simulation...")
        gs.destroy()
        print("✅ Genesis destroyed successfully")
        return True
    except Exception as e:
        print(f"⚠️ Genesis destroy error: {e}")
        return False

# Run Genesis destroy in separate thread with timeout
import threading
destroy_success = [False]

def destroy_worker():
    """Worker thread for Genesis destroy"""
    destroy_success[0] = safe_genesis_destroy()

# Start destroy thread with timeout protection
destroy_thread = threading.Thread(target=destroy_worker, daemon=True)
destroy_thread.start()
destroy_thread.join(timeout=10)  # 10 second timeout

# Check if destroy completed or timed out
if destroy_thread.is_alive():
    print("⚠️ Genesis destroy timed out after 10 seconds, forcing exit...")
    print("⚠️ This prevents infinite hanging - Genesis may have cleanup issues")
else:
    if destroy_success[0]:
        print("✅ Genesis destroyed successfully")
    else:
        print("⚠️ Genesis destroy completed with errors")

# STEP 5: Final exit (guaranteed to work)
print("✅ All cleanup completed - forcing safe exit")
import os
os._exit(0)  # Force exit regardless of Genesis state'''

        # Pattern to find the problematic cleanup section
        # Look for the section that has gs.destroy() without proper TensorBoard cleanup
        pattern = r'# Attempt clean Genesis shutdown.*?os\._exit\(0\)'
        
        # Replace with the safe cleanup
        if re.search(pattern, content, re.DOTALL):
            updated_content = re.sub(pattern, new_cleanup, content, flags=re.DOTALL)
            
            # Create backup
            backup_path = file_path + '.backup_before_fix'
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Backup created: {backup_path}")
            
            # Write the fixed file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            
            print(f"✅ Genesis cleanup fix applied to {file_path}")
            return True
        else:
            print("⚠️ Could not find the problematic cleanup section")
            return False
            
    except Exception as e:
        print(f"❌ Error applying fix: {e}")
        return False

def main():
    """
    Main function to apply the Genesis cleanup fix
    """
    # Path to your file
    file_path = r"c:\Users\Ritu\Project\total_robotics\genesis_AI_sims\genesis_installation\NewTest_v1_pure_env_bkup.py"
    
    print("🔧 Genesis gs.destroy() Hanging Fix")
    print("=" * 50)
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    print(f"📁 Target file: {file_path}")
    
    # Apply the fix
    success = auto_fix_genesis_cleanup(file_path)
    
    if success:
        print("\n🎉 SUCCESS! Genesis cleanup fix applied!")
        print("\n✅ Benefits:")
        print("• TensorBoard processes stop BEFORE gs.destroy()")
        print("• Timeout protection prevents infinite hanging")
        print("• Force exit guarantees program termination")
        print("• No more Genesis visualizer getting stuck!")
        
        print("\n🚀 You can now run your training without Genesis hanging!")
    else:
        print("\n❌ Fix could not be applied automatically")
        print("📋 Please apply the manual fix from genesis_cleanup_fix.py")

if __name__ == "__main__":
    main()
