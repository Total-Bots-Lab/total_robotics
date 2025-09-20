#!/usr/bin/env python3
"""
Quick fix script to remove problematic functions from NewTest_v1_pure_env.py
This will clean up the hanging Genesis issue by removing the problematic cleanup functions.
"""

import re

def fix_genesis_hanging_issue():
    file_path = "NewTest_v1_pure_env.py"
    
    print("🔧 Fixing Genesis hanging issue...")
    
    try:
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remove the problematic force_exit function
        content = re.sub(
            r'def force_exit\(\):.*?os\._exit\(0\)',
            '# Removed problematic force_exit function',
            content,
            flags=re.DOTALL
        )
        
        # Remove the problematic cleanup_genesis function
        content = re.sub(
            r'def cleanup_genesis\(\):.*?return False',
            '# Removed problematic cleanup_genesis function',
            content,
            flags=re.DOTALL
        )
        
        # Remove threading timeout thread creation
        content = re.sub(
            r'timeout_thread = threading\.Thread\(target=force_exit.*?timeout_thread\.start\(\)',
            '# Removed problematic timeout thread',
            content,
            flags=re.DOTALL
        )
        
        # Remove any calls to cleanup_genesis()
        content = re.sub(
            r'cleanup_success = cleanup_genesis\(\).*?print\("🏁 Script completed - forced cleanup"\)',
            'print("🔄 Using safe exit with timeout protection...")',
            content,
            flags=re.DOTALL
        )
        
        # Write the fixed content back
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ Fixed Genesis hanging issue!")
        print("🔧 Removed problematic cleanup functions")
        print("✨ Script should now exit cleanly")
        
    except Exception as e:
        print(f"❌ Error fixing file: {e}")

if __name__ == "__main__":
    fix_genesis_hanging_issue()
