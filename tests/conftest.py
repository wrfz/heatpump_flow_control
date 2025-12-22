"""Configuration for pytest."""

from pathlib import Path
import sys

# Add custom_components to Python path
custom_components_path = Path(__file__).parent.parent / "custom_components"
sys.path.insert(0, str(custom_components_path))
