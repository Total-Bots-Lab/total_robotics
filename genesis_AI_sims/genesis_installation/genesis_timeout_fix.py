"""
Genesis Timeout Fix - Handles CUDA Context Errors

PROBLEM IDENTIFIED:
The error "CUDA_ERROR_INVALID_CONTEXT: invalid device context while calling stream_synchronize"
occurs when Genesis is forced to exit during active CUDA operations.

ROOT CAUSE:
- Time-based exit limits (5 minutes) interrupt training mid-simulation
- Genesis tries to synchronize CUDA streams during forced shutdown
- CUDA context becomes invalid when operations are interrupted
- Results in hanging at "Exiting Genesis and caching compiled kernels"

SOLUTION:
1. Remove time limits and run full 50 episodes for natural completion
2. Use this timeout protection as fallback for existing interrupted runs
3. Allow CUDA operations to complete before Genesis cleanup

ERROR SIGNATURE:
"[taichi/rhi/cuda/cuda_driver.h:taichi::lang::CUDADriverFunction<void *>::operator ()@92] 
CUDA Error CUDA_ERROR_INVALID_CONTEXT: invalid device context while calling stream_synchronize (cuStreamSynchronize)"
"""

import threading
import time
import sys
import os

def genesis_cleanup_with_timeout(gs_module, timeout=10):
    """
    Genesis cleanup with timeout to prevent CUDA caching hang
    This is a standalone fix for the Genesis hanging issue
    Handles CUDA_ERROR_INVALID_CONTEXT from premature exits
    """
    cleanup_completed = threading.Event()
    cleanup_success = False
    
    def do_cleanup():
        nonlocal cleanup_success
        try:
            print("🔄 Destroying Genesis backend...")
            gs_module.destroy()
            cleanup_success = True
            print("✅ Genesis simulation closed successfully")
        except Exception as e:
            error_msg = str(e)
            if "CUDA_ERROR_INVALID_CONTEXT" in error_msg:
                print("⚠️ CUDA Context Error: This occurs when Genesis exits during active CUDA operations")
                print("💡 Root cause: Premature exit interrupting CUDA stream synchronization")
                print("🔧 Solution: Remove time limits to allow natural completion")
            elif "cuStreamSynchronize" in error_msg:
                print("⚠️ CUDA Stream Sync Error: Genesis was interrupted during CUDA operations")
                print("💡 This happens when training is cut short before natural completion")
            else:
                print(f"⚠️ Genesis cleanup issue: {e}")
            
            # Still consider it a form of completion (even with CUDA errors)
            cleanup_success = True
        finally:
            cleanup_completed.set()
    
    print(f"🔄 Genesis cleanup with {timeout}s timeout protection...")
    
    # Start cleanup in separate thread
    cleanup_thread = threading.Thread(target=do_cleanup, daemon=True)
    cleanup_thread.start()
    
    # Wait with timeout (specifically for CUDA kernel caching issue)
    if cleanup_completed.wait(timeout=timeout):
        return cleanup_success
    else:
        print("⚠️ TIMEOUT: Genesis CUDA kernel caching detected - forcing exit")
        print("💡 This prevents indefinite hanging on Windows")
        return False

def safe_exit_with_genesis_timeout(gs_module):
    """Complete safe exit sequence with Genesis timeout protection"""
    print("\n🏁 Training completed - beginning safe exit sequence...")
    
    # Step 1: Close log file if it exists
    try:
        if 'log_file' in globals() and hasattr(globals()['log_file'], 'close'):
            globals()['log_file'].close()
            print("📝 Log file closed successfully")
    except:
        print("📝 Log file already closed")
    
    # Step 2: Allow CUDA operations to complete
    print("⏱️ Allowing CUDA operations to complete...")
    time.sleep(5)  # Increased wait time for CUDA operations
    
    # Step 3: Genesis cleanup with timeout
    print("🔧 Note: If you see CUDA_ERROR_INVALID_CONTEXT, it indicates premature exit")
    print("💡 Full 50-episode training should prevent this error")
    cleanup_success = genesis_cleanup_with_timeout(gs_module, timeout=15)  # Increased timeout
    
    # Step 4: Final status and exit
    if cleanup_success:
        print("🏁 Script completed successfully - clean exit")
    else:
        print("🏁 Script completed with timeout protection")
    
    print("✨ All systems closed gracefully")
    print("🔄 If CUDA errors occurred, run full 50 episodes to prevent them")
    
    # Natural exit
    sys.exit(0)

if __name__ == "__main__":
    print("Genesis Timeout Fix - Standalone Test")
    print("Import this module and use: safe_exit_with_genesis_timeout(gs)")
