# tests/conftest.py
import sys
from pathlib import Path

# Ensure the package root is on the path regardless of how pytest is invoked
sys.path.insert(0, str(Path(__file__).parent.parent))
