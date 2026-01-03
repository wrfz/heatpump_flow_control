"""Unit tests for number.py."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from custom_components.heatpump_flow_control.const import (
    CONF_AUSSEN_TEMP_SENSOR,
    CONF_IS_HEATING_ENTITY,
    CONF_LEARNING_RATE,
    CONF_MAX_VORLAUF,
    CONF_MIN_VORLAUF,
    CONF_RAUM_IST_SENSOR,
    CONF_RAUM_SOLL_SENSOR,
    CONF_UPDATE_INTERVAL,
    CONF_VORLAUF_IST_SENSOR,
    CONF_VORLAUF_SOLL_ENTITY,
)
from custom_components.heatpump_flow_control.flow_controller import (
    Features,
    FlowController,
)
from custom_components.heatpump_flow_control.number import (
    FlowControlNumber,
    SensorValues,
    VorlaufSollAndFeatures,
)
import pytest


@pytest.fixture
def mock_hass():
    """Create a mock HomeAssistant instance."""
    hass = MagicMock()
    hass.config.path = Mock(return_value="/tmp/test_model.pkl")
    hass.states.get = Mock(return_value=None)
    hass.services.async_call = AsyncMock()
    hass.async_add_executor_job = AsyncMock()
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
    """Create full configuration with all options."""
    config = minimal_config.copy()
    config.update(
        {
            CONF_IS_HEATING_ENTITY: "sensor.betriebsart",
            CONF_MIN_VORLAUF: 25.0,
            CONF_MAX_VORLAUF: 50.0,
            CONF_UPDATE_INTERVAL: 60,
            CONF_LEARNING_RATE: 0.01,
        }
    )
    return config


class TestFlowControlNumberInit:
    """Test initialization of FlowControlNumber."""

    def test_init_minimal_config(self, mock_hass, minimal_config, mock_config_entry):
        """Test initialization with minimal config."""
        number = FlowControlNumber(mock_hass, minimal_config, mock_config_entry)

        assert number.hass == mock_hass
        assert number._aussen_temp_sensor == "sensor.aussen_temp"
        assert number._raum_ist_sensor == "sensor.raum_ist"
        assert number._raum_soll_sensor == "sensor.raum_soll"
        assert number._vorlauf_ist_sensor == "sensor.vorlauf_ist"
        assert number._vorlauf_soll_entity == "number.vorlauf_soll"
        assert number._is_heating_entity is None
        assert isinstance(number._controller, FlowController)
        assert number._attr_native_value is None
        assert number._available is False

    def test_init_full_config(self, mock_hass, full_config, mock_config_entry):
        """Test initialization with full config."""
        number = FlowControlNumber(mock_hass, full_config, mock_config_entry)

        assert number._is_heating_entity == "sensor.betriebsart"
        assert number._update_interval_minutes == 60

    def test_init_default_values(self, mock_hass, minimal_config, mock_config_entry):
        """Test that default values are applied correctly."""
        number = FlowControlNumber(mock_hass, minimal_config, mock_config_entry)

        assert number._attr_native_min_value == number._controller.min_vorlauf
        assert number._attr_native_max_value == number._controller.max_vorlauf
        assert number._attr_native_step == 0.1


class TestFormatSensorAsFloat:
    """Test _format_sensor_as_float method."""

    def test_format_valid_state(self, mock_hass, minimal_config, mock_config_entry):
        """Test formatting valid sensor state."""
        number = FlowControlNumber(mock_hass, minimal_config, mock_config_entry)
        state = MagicMock()
        state.state = "23.456"

        result = number._format_sensor_as_float(state)
        assert result == "23.46"

    def test_format_none_state(self, mock_hass, minimal_config, mock_config_entry):
        """Test formatting None state."""
        number = FlowControlNumber(mock_hass, minimal_config, mock_config_entry)

        result = number._format_sensor_as_float(None)
        assert result == "None"

    def test_format_unknown_state(self, mock_hass, minimal_config, mock_config_entry):
        """Test formatting unknown state."""
        number = FlowControlNumber(mock_hass, minimal_config, mock_config_entry)
        state = MagicMock()
        state.state = "unknown"

        result = number._format_sensor_as_float(state)
        assert result == "None"

    def test_format_unavailable_state(self, mock_hass, minimal_config, mock_config_entry):
        """Test formatting unavailable state."""
        number = FlowControlNumber(mock_hass, minimal_config, mock_config_entry)
        state = MagicMock()
        state.state = "unavailable"

        result = number._format_sensor_as_float(state)
        assert result == "None"

    def test_format_invalid_state(self, mock_hass, minimal_config, mock_config_entry):
        """Test formatting invalid state."""
        number = FlowControlNumber(mock_hass, minimal_config, mock_config_entry)
        state = MagicMock()
        state.state = "not_a_number"

        result = number._format_sensor_as_float(state)
        assert result == "Invalid"


class TestAsyncGetSensorValues:
    """Test _async_get_sensor_values method."""

    @pytest.mark.asyncio
    async def test_get_sensor_values_success(self, mock_hass, minimal_config, mock_config_entry):
        """Test successful sensor value retrieval."""
        number = FlowControlNumber(mock_hass, minimal_config, mock_config_entry)

        # Mock sensor states
        aussen_state = MagicMock()
        aussen_state.state = "5.0"
        raum_ist_state = MagicMock()
        raum_ist_state.state = "22.0"
        raum_soll_state = MagicMock()
        raum_soll_state.state = "21.0"
        raum_soll_state.domain = "input_number"
        vorlauf_ist_state = MagicMock()
        vorlauf_ist_state.state = "35.0"

        mock_hass.states.get.side_effect = [
            aussen_state,
            raum_ist_state,
            raum_soll_state,
            vorlauf_ist_state,
        ]

        result = await number._async_get_sensor_values()

        assert result is not None
        assert result.aussen_temp == 5.0
        assert result.raum_ist == 22.0
        assert result.raum_soll == 21.0
        assert result.vorlauf_ist == 35.0

    @pytest.mark.asyncio
    async def test_get_sensor_values_climate_entity(self, mock_hass, minimal_config, mock_config_entry):
        """Test sensor value retrieval with climate entity."""
        number = FlowControlNumber(mock_hass, minimal_config, mock_config_entry)

        # Mock sensor states
        aussen_state = MagicMock()
        aussen_state.state = "5.0"
        raum_ist_state = MagicMock()
        raum_ist_state.state = "22.0"
        raum_soll_state = MagicMock()
        raum_soll_state.state = "heat"  # Climate entity state
        raum_soll_state.domain = "climate"
        raum_soll_state.attributes = {"temperature": 21.5}
        vorlauf_ist_state = MagicMock()
        vorlauf_ist_state.state = "35.0"

        mock_hass.states.get.side_effect = [
            aussen_state,
            raum_ist_state,
            raum_soll_state,
            vorlauf_ist_state,
        ]

        result = await number._async_get_sensor_values()

        assert result is not None
        assert result.raum_soll == 21.5

    @pytest.mark.asyncio
    async def test_get_sensor_values_unavailable(self, mock_hass, minimal_config, mock_config_entry):
        """Test sensor value retrieval with unavailable sensor."""
        number = FlowControlNumber(mock_hass, minimal_config, mock_config_entry)

        # Mock sensor states
        aussen_state = MagicMock()
        aussen_state.state = "unavailable"

        mock_hass.states.get.side_effect = [
            aussen_state,
            MagicMock(),
            MagicMock(),
            MagicMock(),
        ]

        result = await number._async_get_sensor_values()

        assert result is None

    @pytest.mark.asyncio
    async def test_get_sensor_values_missing_sensor(self, mock_hass, minimal_config, mock_config_entry):
        """Test sensor value retrieval with missing sensor."""
        number = FlowControlNumber(mock_hass, minimal_config, mock_config_entry)

        mock_hass.states.get.side_effect = [None, MagicMock(), MagicMock(), MagicMock()]

        result = await number._async_get_sensor_values()

        assert result is None

    @pytest.mark.asyncio
    async def test_get_sensor_values_invalid_value(self, mock_hass, minimal_config, mock_config_entry):
        """Test sensor value retrieval with invalid value."""
        number = FlowControlNumber(mock_hass, minimal_config, mock_config_entry)

        # Mock sensor states
        aussen_state = MagicMock()
        aussen_state.state = "not_a_number"

        mock_hass.states.get.side_effect = [
            aussen_state,
            MagicMock(),
            MagicMock(),
            MagicMock(),
        ]

        result = await number._async_get_sensor_values()

        assert result is None


class TestIsHeatingMode:
    """Test _is_heating_mode method."""

    @pytest.mark.asyncio
    async def test_no_sensor_configured(self, mock_hass, minimal_config, mock_config_entry):
        """Test when no betriebsart sensor is configured."""
        number = FlowControlNumber(mock_hass, minimal_config, mock_config_entry)

        result = await number._is_heating_mode()

        assert result is True

    @pytest.mark.asyncio
    async def test_sensor_in_heating_mode(self, mock_hass, full_config, mock_config_entry):
        """Test when sensor shows heating mode."""
        number = FlowControlNumber(mock_hass, full_config, mock_config_entry)

        betriebsart_state = MagicMock()
        betriebsart_state.state = "Heizen"
        mock_hass.states.get.return_value = betriebsart_state

        result = await number._is_heating_mode()

        assert result is True

    @pytest.mark.asyncio
    async def test_sensor_not_in_heating_mode(self, mock_hass, full_config, mock_config_entry):
        """Test when sensor shows different mode."""
        number = FlowControlNumber(mock_hass, full_config, mock_config_entry)

        betriebsart_state = MagicMock()
        betriebsart_state.state = "Kühlen"
        mock_hass.states.get.return_value = betriebsart_state

        result = await number._is_heating_mode()

        assert result is False

    @pytest.mark.asyncio
    async def test_sensor_with_whitespace(self, mock_hass, full_config, mock_config_entry):
        """Test when sensor value has whitespace."""
        number = FlowControlNumber(mock_hass, full_config, mock_config_entry)

        betriebsart_state = MagicMock()
        betriebsart_state.state = "  Heizen  "
        mock_hass.states.get.return_value = betriebsart_state

        result = await number._is_heating_mode()

        assert result is True


class TestAsyncSetVorlaufSoll:
    """Test _async_set_vorlauf_soll method."""

    @pytest.mark.asyncio
    async def test_set_input_number(self, mock_hass, minimal_config, mock_config_entry):
        """Test setting value on input_number entity."""
        number = FlowControlNumber(mock_hass, minimal_config, mock_config_entry)

        target_state = MagicMock()
        target_state.domain = "input_number"
        mock_hass.states.get.return_value = target_state

        await number._async_set_vorlauf_soll(35.5)

        mock_hass.services.async_call.assert_called_once_with(
            "input_number",
            "set_value",
            {"entity_id": "number.vorlauf_soll", "value": 35.5},
        )

    @pytest.mark.asyncio
    async def test_set_number_entity(self, mock_hass, minimal_config, mock_config_entry):
        """Test setting value on number entity."""
        number = FlowControlNumber(mock_hass, minimal_config, mock_config_entry)

        target_state = MagicMock()
        target_state.domain = "number"
        mock_hass.states.get.return_value = target_state

        await number._async_set_vorlauf_soll(35.5)

        mock_hass.services.async_call.assert_called_once_with(
            "number",
            "set_value",
            {"entity_id": "number.vorlauf_soll", "value": 35.5},
        )

    @pytest.mark.asyncio
    async def test_set_climate_entity(self, mock_hass, minimal_config, mock_config_entry):
        """Test setting value on climate entity."""
        number = FlowControlNumber(mock_hass, minimal_config, mock_config_entry)

        target_state = MagicMock()
        target_state.domain = "climate"
        mock_hass.states.get.return_value = target_state

        await number._async_set_vorlauf_soll(35.5)

        mock_hass.services.async_call.assert_called_once_with(
            "climate",
            "set_temperature",
            {"entity_id": "number.vorlauf_soll", "temperature": 35.5},
        )

    @pytest.mark.asyncio
    async def test_set_target_not_found(self, mock_hass, minimal_config, mock_config_entry):
        """Test setting value when target entity not found."""
        number = FlowControlNumber(mock_hass, minimal_config, mock_config_entry)

        mock_hass.states.get.return_value = None

        await number._async_set_vorlauf_soll(35.5)

        # Should not call any service
        mock_hass.services.async_call.assert_not_called()

    @pytest.mark.asyncio
    async def test_set_unsupported_domain(self, mock_hass, minimal_config, mock_config_entry):
        """Test setting value on unsupported domain."""
        number = FlowControlNumber(mock_hass, minimal_config, mock_config_entry)

        target_state = MagicMock()
        target_state.domain = "sensor"
        mock_hass.states.get.return_value = target_state

        await number._async_set_vorlauf_soll(35.5)

        # Should not call any service
        mock_hass.services.async_call.assert_not_called()

    @pytest.mark.asyncio
    async def test_set_rounds_value(self, mock_hass, minimal_config, mock_config_entry):
        """Test that value is rounded correctly."""
        number = FlowControlNumber(mock_hass, minimal_config, mock_config_entry)

        target_state = MagicMock()
        target_state.domain = "number"
        mock_hass.states.get.return_value = target_state

        await number._async_set_vorlauf_soll(35.567)

        mock_hass.services.async_call.assert_called_once_with(
            "number",
            "set_value",
            {"entity_id": "number.vorlauf_soll", "value": 35.6},
        )


class TestAsyncSetNativeValue:
    """Test async_set_native_value method."""

    @pytest.mark.asyncio
    async def test_set_native_value(self, mock_hass, minimal_config, mock_config_entry):
        """Test manual value override."""
        number = FlowControlNumber(mock_hass, minimal_config, mock_config_entry)
        number.async_write_ha_state = Mock()

        target_state = MagicMock()
        target_state.domain = "number"
        mock_hass.states.get.return_value = target_state

        await number.async_set_native_value(40.0)

        assert number._attr_native_value == 40.0
        mock_hass.services.async_call.assert_called_once()
        number.async_write_ha_state.assert_called_once()


class TestAsyncUpdateVorlaufSoll:
    """Test _async_update_vorlauf_soll method."""

    @pytest.mark.asyncio
    async def test_update_success(self, mock_hass, minimal_config, mock_config_entry):
        """Test successful vorlauf soll update."""
        number = FlowControlNumber(mock_hass, minimal_config, mock_config_entry)
        number.async_write_ha_state = Mock()

        mock_sensor_data = SensorValues(
            aussen_temp=5.0,
            raum_ist=22.0,
            raum_soll=21.0,
            vorlauf_ist=35.0
        )

        with patch.object(
            number, "_async_get_sensor_values", return_value=mock_sensor_data
        ):
            # Mock controller calculation - create complete Features object
            mock_features = Features(
                aussen_temp=5.0,
                raum_ist=22.0,
                raum_soll=21.0,
                vorlauf_ist=35.0,
                raum_abweichung=-1.0,
                aussen_trend_1h=0.5,
                stunde_sin=0.0,
                stunde_cos=1.0,
                wochentag_sin=0.0,
                wochentag_cos=1.0,
                temp_diff=-17.0,
                vorlauf_raum_diff=13.0,
            )
            mock_hass.async_add_executor_job.side_effect = [
                VorlaufSollAndFeatures(vorlauf=38.5, features=mock_features),  # berechne_vorlauf_soll
                None,  # save_model
            ]

            # Mock switch state (off)
            switch_state = MagicMock()
            switch_state.state = "off"
            mock_hass.states.get.return_value = switch_state

            await number._async_update_vorlauf_soll()

            assert number._attr_native_value == 38.5
            assert number._available is True
            number.async_write_ha_state.assert_called()

    @pytest.mark.asyncio
    async def test_update_with_switch_enabled(self, mock_hass, minimal_config, mock_config_entry):
        """Test update when control switch is enabled."""
        number = FlowControlNumber(mock_hass, minimal_config, mock_config_entry)
        number.async_write_ha_state = Mock()

        mock_sensor_data = SensorValues(
            aussen_temp=5.0,
            raum_ist=22.0,
            raum_soll=21.0,
            vorlauf_ist=35.0
        )

        with (
            patch.object(
                number, "_async_get_sensor_values", return_value=mock_sensor_data
            ),
            patch.object(number, "_async_set_vorlauf_soll") as mock_set,
        ):
            # Mock controller calculation - create complete Features object
            mock_features = Features(
                aussen_temp=5.0,
                raum_ist=22.0,
                raum_soll=21.0,
                vorlauf_ist=35.0,
                raum_abweichung=-1.0,
                aussen_trend_1h=0.5,
                stunde_sin=0.0,
                stunde_cos=1.0,
                wochentag_sin=0.0,
                wochentag_cos=1.0,
                temp_diff=-17.0,
                vorlauf_raum_diff=13.0,
            )
            mock_hass.async_add_executor_job.side_effect = [
                VorlaufSollAndFeatures(vorlauf=38.5, features=mock_features),  # berechne_vorlauf_soll
                None,  # save_model
            ]

            # Mock switch state (on)
            switch_state = MagicMock()
            switch_state.state = "on"

            target_state = MagicMock()
            target_state.domain = "number"

            mock_hass.states.get.side_effect = [switch_state, target_state]

            await number._async_update_vorlauf_soll()

            mock_set.assert_called_once_with(38.5)

    @pytest.mark.asyncio
    async def test_update_sensors_unavailable(self, mock_hass, minimal_config, mock_config_entry):
        """Test update when sensors are unavailable."""
        number = FlowControlNumber(mock_hass, minimal_config, mock_config_entry)
        number.async_write_ha_state = Mock()

        with patch.object(number, "_async_get_sensor_values", return_value=None):
            await number._async_update_vorlauf_soll()

            assert number._available is False
            number.async_write_ha_state.assert_called()

    @pytest.mark.asyncio
    async def test_update_exception(self, mock_hass, minimal_config, mock_config_entry):
        """Test update with exception."""
        number = FlowControlNumber(mock_hass, minimal_config, mock_config_entry)
        number.async_write_ha_state = Mock()

        with patch.object(
            number, "_async_get_sensor_values", side_effect=Exception("Test error")
        ):
            await number._async_update_vorlauf_soll()

            assert number._available is False
            number.async_write_ha_state.assert_called()


class TestExtraStateAttributes:
    """Test extra_state_attributes property."""

    def test_extra_state_attributes(self, mock_hass, minimal_config, mock_config_entry):
        """Test that extra state attributes are returned."""
        number = FlowControlNumber(mock_hass, minimal_config, mock_config_entry)
        number._extra_attributes = {"test_attr": "test_value"}

        assert number.extra_state_attributes == {"test_attr": "test_value"}


class TestAvailable:
    """Test available property."""

    def test_available_true(self, mock_hass, minimal_config, mock_config_entry):
        """Test available property when entity is available."""
        number = FlowControlNumber(mock_hass, minimal_config, mock_config_entry)
        number._available = True

        assert number.available is True

    def test_available_false(self, mock_hass, minimal_config, mock_config_entry):
        """Test available property when entity is unavailable."""
        number = FlowControlNumber(mock_hass, minimal_config, mock_config_entry)
        number._available = False

        assert number.available is False


class TestCreateTask:
    """Test _create_task method."""

    @pytest.mark.asyncio
    async def test_create_task(self, mock_hass, minimal_config, mock_config_entry):
        """Test task creation and tracking."""
        number = FlowControlNumber(mock_hass, minimal_config, mock_config_entry)

        async def test_coro():
            await asyncio.sleep(0.01)
            return "done"

        task = number._create_task(test_coro())

        assert isinstance(task, asyncio.Task)
        assert task in number._tasks

        result = await task
        assert result == "done"
        # Task should be removed from set after completion
        assert task not in number._tasks
