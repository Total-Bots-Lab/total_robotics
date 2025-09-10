"""
Application Layer
================

User-facing training, testing, and demonstration scripts.
This layer provides ready-to-use applications for the platform.
"""

# Note: These are scripts meant to be run directly, not imported as modules
# Import functions are available but the main use is via command line

try:
    from .universal_train import (
        test_predefined_robot,
        test_custom_robot, 
        train_with_sb3,
        demo_advanced_usage
    )
    from .test_simulation_stage import main as run_tests
    
    __all__ = [
        'test_predefined_robot',
        'test_custom_robot',
        'train_with_sb3', 
        'demo_advanced_usage',
        'run_tests'
    ]
except ImportError:
    # Scripts may have dependencies that aren't available
    __all__ = []
