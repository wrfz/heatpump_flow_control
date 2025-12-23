"""Integration tests for FlowControlNumber entity.

Focus on public API and realistic integration scenarios.
Tests the entity behavior as it would be used by Home Assistant.
"""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

from custom_components.heatpump_flow_control.const import (
    CONF_AUSSEN_TEMP_SENSOR,
    CONF_BETRIEBSART_HEIZEN_WERT,
    CONF_BETRIEBSART_SENSOR,
    CONF_LEARNING_RATE,
    CONF_MAX_VORLAUF,
    CONF_MIN_VORLAUF,
    CONF_POWER_SENSOR,
    CONF_RAUM_IST_SENSOR,
    CONF_RAUM_SOLL_SENSOR,
    CONF_TREND_HISTORY_SIZE,
    CONF_UPDATE_INTERVAL,
    CONF_VORLAUF_IST_SENSOR,
    CONF_VORLAUF_SOLL_ENTITY,
    DOMAIN,
)
from custom_components.heatpump_flow_control.flow_controller import FlowController
from custom_components.heatpump_flow_control.number import FlowControlNumber
import pytest


@pytest.fixture
def mock_hass():
    """Create a mock HomeAssistant instance."""
    hass = MagicMock()
    hass.config.path = Mock(return_value="/tmp/test_model.pkl")
    hass.services.async_call = AsyncMock()
    hass.async_add_executor_job = AsyncMock()

    # Default: all sensors available
    def get_state(entity_id):
        states = {
            "sensor.aussen_temp": MagicMock(state="5.0", domain="sensor"),
            "sensor.raum_ist": MagicMock(state="22.0", domain="sensor"),
            "sensor.raum_soll": MagicMock(state="21.0", domain="input_number"),
            "sensor.vorlauf_ist": MagicMock(state="35.0", domain="sensor"),
            "number.vorlauf_soll": MagicMock(state="35.0", domain="number"),
            f"switch.{DOMAIN}_aktiv": MagicMock(state="off", domain="switch"),
        }
        return states.get(entity_id)

    hass.states.get = Mock(side_effect=get_state)
    return hass


@pytest.fixture
def minimal_config():
    """Create minimal configuration."""
    return {
        CONF_AUSSEN_TEMP_SENSOR: "sensor.aussen_temp",
        CONF_RAUM_IST_SENSOR: "sensor.raum_ist",
        CONF_RAUM_SOLL_SENSOR: "sensor.raum_soll",
        CONF_VORLAUF_IST_SENSOR: "sensor.vorlauf_ist",
        CONF_VORLAUF_SOLL_ENTITY: "number.vorlauf_soll",
    }


@pytest.fixture
def full_config(minimal_config):
    """Create full configuration."""
    config = minimal_config.copy()
    config.update(
        {
            CONF_BETRIEBSART_SENSOR: "sensor.betriebsart",
            CONF_BETRIEBSART_HEIZEN_WERT: "Heizen",
            CONF_POWER_SENSOR: "sensor.power",
            CONF_MIN_VORLAUF: 25.0,
            CONF_MAX_VORLAUF: 50.0,
            CONF_UPDATE_INTERVAL: 60,
            CONF_LEARNING_RATE: 0.01,
            CONF_TREND_HISTORY_SIZE: 10,
        }
    )
    return config


class TestFlowControlNumberPublicAPI:
    """Test public API of FlowControlNumber."""

    def test_initialization(self, mock_hass, minimal_config):
        """Test entity initializes correctly."""
        number = FlowControlNumber(mock_hass, minimal_config, "test_entry")

        # Public properties
        assert number.name == "Heatpump Flow Control Vorlauf Soll"
        assert number.unique_id == f"{DOMAIN}_test_entry_vorlauf_soll"
        assert number.native_min_value == number._min_vorlauf
        assert number.native_max_value == number._max_vorlauf
        assert number.native_step == 0.5
        assert number.native_unit_of_measurement == "°C"
        assert number.icon == "mdi:thermometer-auto"

    def test_available_property(self, mock_hass, minimal_config):
        """Test available property reflects entity state."""
        number = FlowControlNumber(mock_hass, minimal_config, "test_entry")

        # Initially unavailable
        assert number.available is False

        # Can be set to available
        number._available = True
        assert number.available is True

    def test_extra_state_attributes_property(self, mock_hass, minimal_config):
        """Test extra state attributes are accessible."""
        number = FlowControlNumber(mock_hass, minimal_config, "test_entry")

        # Initially empty
        assert number.extra_state_attributes == {}

        # Can be populated
        number._extra_attributes = {"test": "value"}
        assert number.extra_state_attributes == {"test": "value"}

    @pytest.mark.asyncio
    async def test_async_set_native_value(self, mock_hass, minimal_config):
        """Test manual value override through public API."""
        number = FlowControlNumber(mock_hass, minimal_config, "test_entry")
        number.async_write_ha_state = Mock()

        # Set value manually
        await number.async_set_native_value(40.0)

        # Value should be updated
        assert number.native_value == 40.0

        # Should call service to update target entity
        mock_hass.services.async_call.assert_called_once()
        call_args = mock_hass.services.async_call.call_args
        assert call_args[0][0] == "number"
        assert call_args[0][1] == "set_value"
        assert call_args[0][2]["value"] == 40.0

        # State should be written
        number.async_write_ha_state.assert_called_once()


class TestFlowControlNumberLifecycle:
    """Test entity lifecycle methods."""

    @pytest.mark.asyncio
    async def test_async_added_to_hass(self, mock_hass, minimal_config):
        """Test entity setup when added to hass."""
        number = FlowControlNumber(mock_hass, minimal_config, "test_entry")

        # Mock model loading
        with patch.object(number, "_async_load_model", new_callable=AsyncMock):
            # Mock restored state
            last_state = MagicMock()
            last_state.state = "38.5"

            with patch.object(number, "async_get_last_state", return_value=last_state):
                await number.async_added_to_hass()

        # Should restore previous value
        assert number.native_value == 38.5

        # Should have set up listeners
        assert number._update_interval_listener is not None
        assert number._state_change_listener is not None

    @pytest.mark.asyncio
    async def test_async_will_remove_from_hass(self, mock_hass, minimal_config):
        """Test cleanup when entity is removed."""
        number = FlowControlNumber(mock_hass, minimal_config, "test_entry")

        # Set up listeners
        listener1 = Mock()
        listener2 = Mock()
        number._update_interval_listener = listener1
        number._state_change_listener = listener2

        # Remove from hass
        await number.async_will_remove_from_hass()

        # Listeners should be called (cleanup)
        listener1.assert_called_once()
        listener2.assert_called_once()


class TestFlowControlNumberIntegration:
    """Test realistic integration scenarios."""

    @pytest.mark.asyncio
    async def test_full_update_cycle_with_all_sensors(self, mock_hass, full_config):
        """Test complete update cycle with all sensors available."""
        number = FlowControlNumber(mock_hass, full_config, "test_entry")
        number.async_write_ha_state = Mock()

        # Mock all sensor states
        def get_state(entity_id):
            states = {
                "sensor.aussen_temp": MagicMock(state="5.0"),
                "sensor.raum_ist": MagicMock(state="22.0"),
                "sensor.raum_soll": MagicMock(state="21.0", domain="input_number"),
                "sensor.vorlauf_ist": MagicMock(state="35.0"),
                "sensor.betriebsart": MagicMock(state="Heizen"),
                "sensor.power": MagicMock(state="1500.0"),
                f"switch.{DOMAIN}_aktiv": MagicMock(state="off"),
            }
            return states.get(entity_id)

        mock_hass.states.get = Mock(side_effect=get_state)

        # Mock controller calculation
        mock_features = {
            "raum_abweichung": 1.0,
            "aussen_trend": 0.5,
            "aussen_trend_kurz": 0.3,
            "aussen_trend_mittel": 0.2,
            "power_avg_1h": 1400.0,
            "power_favorable_hours": 0.3,
        }
        mock_hass.async_add_executor_job.return_value = (38.5, mock_features)

        # Trigger update through public interface
        await number._async_update_vorlauf_soll()

        # Entity should be available
        assert number.available is True

        # Value should be updated
        assert number.native_value == 38.5

        # Attributes should be populated
        attrs = number.extra_state_attributes
        assert attrs["aussen_temp"] == 5.0
        assert attrs["raum_ist"] == 22.0
        assert attrs["power_current"] == 1500.0
        assert attrs["betriebsart"] == "Heizen"

    @pytest.mark.asyncio
    async def test_handles_unavailable_sensors_gracefully(
        self, mock_hass, minimal_config
    ):
        """Test entity handles unavailable sensors without crashing."""
        number = FlowControlNumber(mock_hass, minimal_config, "test_entry")
        number.async_write_ha_state = Mock()

        # Mock sensor as unavailable
        def get_state(entity_id):
            if entity_id == "sensor.aussen_temp":
                return MagicMock(state="unavailable")
            return MagicMock(state="22.0")

        mock_hass.states.get = Mock(side_effect=get_state)

        # Trigger update
        await number._async_update_vorlauf_soll()

        # Entity should become unavailable
        assert number.available is False

        # No service calls should be made
        mock_hass.services.async_call.assert_not_called()


class TestFlowControllerIntegration:
    """Test integration with FlowController."""

    def test_controller_is_initialized(self, mock_hass, minimal_config):
        """Test that FlowController is properly initialized."""
        number = FlowControlNumber(mock_hass, minimal_config, "test_entry")

        assert isinstance(number._controller, FlowController)
        assert number._controller.min_vorlauf == number._min_vorlauf
        assert number._controller.max_vorlauf == number._max_vorlauf

    @pytest.mark.asyncio
    async def test_model_persistence(self, mock_hass, minimal_config):
        """Test that model is saved after updates."""
        number = FlowControlNumber(mock_hass, minimal_config, "test_entry")
        number.async_write_ha_state = Mock()

        # Mock successful update
        mock_hass.async_add_executor_job.return_value = (
            38.5,
            {
                "raum_abweichung": 1.0,
                "aussen_trend": 0.5,
                "aussen_trend_kurz": 0.3,
                "aussen_trend_mittel": 0.2,
            },
        )

        with patch.object(
            number, "_async_save_model", new_callable=AsyncMock
        ) as mock_save:
            await number._async_update_vorlauf_soll()

            # Model should be saved
            mock_save.assert_called_once()


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_handles_invalid_sensor_values(self, mock_hass, minimal_config):
        """Test handling of invalid sensor values."""
        number = FlowControlNumber(mock_hass, minimal_config, "test_entry")
        number.async_write_ha_state = Mock()

        # Mock invalid sensor value
        def get_state(entity_id):
            return MagicMock(state="not_a_number")

        mock_hass.states.get = Mock(side_effect=get_state)

        # Should not crash
        await number._async_update_vorlauf_soll()

        # Should become unavailable
        assert number.available is False

    @pytest.mark.asyncio
    async def test_recovers_from_controller_exception(self, mock_hass, minimal_config):
        """Test recovery from controller calculation errors."""
        number = FlowControlNumber(mock_hass, minimal_config, "test_entry")
        number.async_write_ha_state = Mock()

        # Mock controller raising exception
        mock_hass.async_add_executor_job.side_effect = Exception("Controller error")

        # Should handle exception gracefully
        await number._async_update_vorlauf_soll()

        # Entity should become unavailable
        assert number.available is False

        # State should still be written
        number.async_write_ha_state.assert_called()
