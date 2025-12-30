"""Configuration for pytest."""

from pathlib import Path
import sys
from unittest.mock import MagicMock

import pytest

# Add custom_components to Python path
custom_components_path = Path(__file__).parent.parent / "custom_components"
sys.path.insert(0, str(custom_components_path))


@pytest.fixture
def mock_config_entry():
    """Create a mock config entry with FlowController in runtime_data."""
    from custom_components.heatpump_flow_control.flow_controller import FlowController

    entry = MagicMock()
    entry.entry_id = "test_entry_id"
    entry.data = {}

    # Create FlowController and store in runtime_data
    controller = FlowController(
        min_vorlauf=25.0,
        max_vorlauf=50.0,
        learning_rate=0.01,
    )
    entry.runtime_data = controller

    return entry
