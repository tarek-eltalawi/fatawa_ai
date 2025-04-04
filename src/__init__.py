"""
Fatawa package - an Islamic Question Answering system
"""

# Make submodules importable
import os
import sys

# Add paths to make imports work across the project
_module_path = os.path.dirname(os.path.abspath(__file__))
_project_path = os.path.dirname(_module_path)

# If not already in path, add it
if _project_path not in sys.path:
    sys.path.insert(0, _project_path)

