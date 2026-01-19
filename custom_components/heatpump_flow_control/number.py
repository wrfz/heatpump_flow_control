"""Number platform for Heatpump Flow Control integration."""
import asyncio
from datetime import timedelta
import logging
from typing import Any, cast

from homeassistant.components.number import NumberEntity, NumberMode
from homeassistant.const import STATE_ON, UnitOfTemperature
from homeassistant.core import HomeAssistant, State, callback
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.entity_platform import (
    AddEntitiesCallback,
    Coroutine,
    Mapping,
)
from homeassistant.helpers.event import (
    async_track_state_change_event,
    async_track_time_interval,
)
from homeassistant.helpers.restore_state import RestoreEntity
from homeassistant.util import dt as dt_util

from . import HeatpumpFlowControlConfigEntry, async_save_controller
from .common import get_device_info
from .const import (
    ATTR_AUSSEN_TEMP,
    ATTR_AUSSEN_TREND_1H,
    ATTR_LAST_UPDATE,
    ATTR_MODEL_MAE,
    ATTR_NEXT_UPDATE,
    ATTR_RAUM_ABWEICHUNG,
    ATTR_RAUM_IST,
    ATTR_RAUM_SOLL,
    ATTR_VORLAUF_IST,
    CONF_AUSSEN_TEMP_SENSOR,
    CONF_IS_HEATING_ENTITY,
    CONF_RAUM_IST_SENSOR,
    CONF_RAUM_SOLL_SENSOR,
    CONF_THERMISCHE_LEISTUNG_SENSOR,
    CONF_UPDATE_INTERVAL,
    CONF_VORLAUF_IST_SENSOR,
    CONF_VORLAUF_SOLL_ENTITY,
    DEFAULT_UPDATE_INTERVAL,
    DOMAIN,
)
from .types import SensorValues, VorlaufSollAndFeatures

# pylint: disable=hass-logger-capital, hass-logger-period
# ruff: noqa: BLE001
# ruff: logging-redundant-exc-info

_LOGGER = logging.getLogger(__name__)

async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: HeatpumpFlowControlConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the Flow Control number."""
    config = config_entry.data

    number = FlowControlNumber(
        hass=hass,
        config=config,
        config_entry=config_entry,
    )

    async_add_entities([number], True)


class FlowControlNumber(NumberEntity, RestoreEntity):
    """Representation of Flow Control Number Entity."""

    _attr_mode = NumberMode.BOX
    _attr_native_unit_of_measurement = UnitOfTemperature.CELSIUS
    _attr_icon = "mdi:thermometer-auto"

    def __init__(
        self,
        hass: HomeAssistant,
        config: Mapping[str, Any],
        config_entry: HeatpumpFlowControlConfigEntry,
    ) -> None:
        """Initialize the number."""
        self.hass = hass
        self._config = config
        self._config_entry = config_entry

        self._attr_device_info = get_device_info(config_entry.entry_id)

        # Sensor Entities
        self._aussen_temp_sensor = config[CONF_AUSSEN_TEMP_SENSOR]
        self._raum_ist_sensor = config[CONF_RAUM_IST_SENSOR]
        self._raum_soll_sensor = config[CONF_RAUM_SOLL_SENSOR]
        self._vorlauf_ist_sensor = config[CONF_VORLAUF_IST_SENSOR]
        self._vorlauf_soll_entity = config[CONF_VORLAUF_SOLL_ENTITY]
        self._is_heating_entity = config.get(CONF_IS_HEATING_ENTITY)
        self._thermische_leistung_sensor = config.get(CONF_THERMISCHE_LEISTUNG_SENSOR)

        # Konfiguration
        self._update_interval_minutes = config.get(
            CONF_UPDATE_INTERVAL, DEFAULT_UPDATE_INTERVAL
        )

        # State
        self._attr_native_value = None
        self._extra_attributes = {}
        self._last_update = None
        self._next_update = None
        self._available = False

        # Heating mode tracking
        self._heating_resumed_at = None  # Track when heating resumed
        self._last_prediction_time = None  # Track last successful prediction

        # Listener
        self._update_interval_listener = None
        self._state_change_listener = None
        self._heating_state_listener = None

        # Number properties
        self._attr_name = "Heatpump Flow Control Vorlauf Soll"
        self._attr_unique_id = f"{DOMAIN}_{config_entry.entry_id}_vorlauf_soll"
        # Get min/max from controller
        self._attr_native_min_value = config_entry.runtime_data.min_vorlauf
        self._attr_native_max_value = config_entry.runtime_data.max_vorlauf
        self._attr_native_step = 0.1

        self._tasks: set[asyncio.Task[None]] = set()

    @property
    def _controller(self):
        """Get FlowController from runtime_data."""
        return self._config_entry.runtime_data

    async def async_added_to_hass(self) -> None:
        """Run when entity about to be added to hass."""
        _LOGGER.info("async_added_to_hass()")

        await super().async_added_to_hass()

        # Restore previous state
        last_state = await self.async_get_last_state()
        if last_state is not None and last_state.state not in (
            "unavailable",
            "unknown",
        ):
            try:
                self._attr_native_value = float(last_state.state)
                _LOGGER.info("Restored state: %s", self._attr_native_value)
            except (ValueError, TypeError):
                pass

        # Setup periodic update
        update_interval = timedelta(minutes=self._update_interval_minutes)
        self._update_interval_listener = async_track_time_interval(
            self.hass,
            self._async_update_vorlauf_soll,
            update_interval,
        )

        # Setup state change listener
        sensor_list = [
            self._aussen_temp_sensor,
            self._raum_ist_sensor,
            self._raum_soll_sensor,
            self._vorlauf_ist_sensor,
        ]
        if self._thermische_leistung_sensor:
            sensor_list.append(self._thermische_leistung_sensor)

        self._state_change_listener = async_track_state_change_event(
            self.hass,
            sensor_list,
            self._async_sensor_state_changed,
        )

        # Setup heating state change listener if configured
        if self._is_heating_entity:
            self._heating_state_listener = async_track_state_change_event(
                self.hass,
                [self._is_heating_entity],
                self._async_heating_state_changed,
            )

        # Sofortiges erstes Update nach 5 Sekunden
        self._create_task(self._delayed_first_update())

    async def async_set_native_value(self, value: float) -> None:
        """Set new value (manual override)."""

        _LOGGER.info("async_set_native_value(%.2f)", value)

        self._attr_native_value = value

        # Bei manueller Änderung auch direkt senden (unabhängig vom Switch)
        await self._async_set_vorlauf_soll(value)
        self.async_write_ha_state()

    async def async_will_remove_from_hass(self) -> None:
        """Run when entity will be removed from hass."""
        _LOGGER.info("async_will_remove_from_hass())")

        if self._update_interval_listener:
            self._update_interval_listener()
        if self._state_change_listener:
            self._state_change_listener()
        if self._heating_state_listener:
            self._heating_state_listener()

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return the state attributes."""

        _LOGGER.info("extra_state_attributes() %s", self._extra_attributes)

        return self._extra_attributes

    @property
    def available(self) -> bool:
        """Return True if entity is available."""
        return self._available

    # Private methods

    async def _delayed_first_update(self):
        """First update after startup with delay."""
        await asyncio.sleep(5)
        await self._async_update_vorlauf_soll()

    @callback
    def _async_sensor_state_changed(self, event) -> None:
        """Handle sensor state changes for learning."""

        _LOGGER.debug("_async_sensor_state_changed()")

        # Ignoriere unavailable/unknown states
        new_state = event.data.get("new_state")
        if new_state is None or new_state.state in ("unavailable", "unknown"):
            return

        # Wenn Number noch unavailable ist, prüfe ob jetzt alle verfügbar sind
        if not self._available:
            self._create_task(self._check_and_update_if_ready())

    async def _check_and_update_if_ready(self):
        """Check if all sensors are ready and trigger update."""

        _LOGGER.debug("_check_and_update_if_ready()")

        sensor_values = await self._async_get_sensor_values()
        if sensor_values is not None:
            await self._async_update_vorlauf_soll()

    async def _is_heating_mode(self) -> bool:
        """Check if heat pump is in heating mode.

        Returns:
            True if in heating mode or sensor not configured, False otherwise
        """
        if not self._is_heating_entity:
            _LOGGER.warning("Entity is_heating_entity not configured!")
            return True  # Kein Sensor konfiguriert = immer lernen

        is_heating_state = self.hass.states.get(self._is_heating_entity)
        if is_heating_state is None:
            _LOGGER.warning(
                "is_heating sensor %s not found, assuming heating disabled",
                self._is_heating_entity,
            )
            return False

        if is_heating_state.state in ("unavailable", "unknown"):
            _LOGGER.debug(
                "Is_heating entity %s is %s, assuming heating disabled",
                self._is_heating_entity,
                is_heating_state.state,
            )
            return False

        is_heating = is_heating_state.state == STATE_ON

        _LOGGER.info("_is_heating_mode: %s, %d", is_heating_state.state, is_heating)

        return is_heating

    async def _async_get_sensor_values(self) -> SensorValues | None:
        """Get current values from all sensors."""

        _LOGGER.info("_async_get_sensor_values()")

        try:
            aussen_temp_state_or_none = self.hass.states.get(self._aussen_temp_sensor)
            raum_ist_state_or_none = self.hass.states.get(self._raum_ist_sensor)
            raum_soll_state_or_none = self.hass.states.get(self._raum_soll_sensor)
            vorlauf_ist_state_or_none = self.hass.states.get(self._vorlauf_ist_sensor)

            _LOGGER.info(
                "Reading sensors: aussen=%s, raum_ist=%s, raum_soll=%s, vorlauf_ist=%s",
                self._format_sensor_as_float(aussen_temp_state_or_none),
                self._format_sensor_as_float(raum_ist_state_or_none),
                self._format_sensor_as_float(raum_soll_state_or_none),
                self._format_sensor_as_float(vorlauf_ist_state_or_none),
            )

            # Prüfe auf unavailable/unknown
            sensors = [
                (self._aussen_temp_sensor, aussen_temp_state_or_none),
                (self._raum_ist_sensor, raum_ist_state_or_none),
                (self._raum_soll_sensor, raum_soll_state_or_none),
                (self._vorlauf_ist_sensor, vorlauf_ist_state_or_none),
            ]

            for entity_id, state_obj in sensors:
                if state_obj is None:
                    _LOGGER.warning("Sensor %s state is None", entity_id)
                    return None
                if state_obj.state in ("unavailable", "unknown"):
                    _LOGGER.warning("Sensor %s is %s", entity_id, state_obj.state)
                    return None

            # Extract temperature values

            aussen_temp = float(cast(State, aussen_temp_state_or_none).state)
            raum_ist = float(cast(State, raum_ist_state_or_none).state)
            vorlauf_ist = float(cast(State, vorlauf_ist_state_or_none).state)
            raum_soll_state = cast(State, raum_soll_state_or_none)

            # Raum-Soll kann von Climate-Entity oder Input-Number kommen
            if raum_soll_state.domain == "climate":
                raum_soll = float(raum_soll_state.attributes.get("temperature", 21.0))
            else:
                raum_soll = cast(State, raum_soll_state)
                raum_soll_value = float(raum_soll.state)
                raum_soll = float(raum_soll_value)

            # Thermische Leistung (optional)
            thermische_leistung = None
            if self._thermische_leistung_sensor:
                thermische_leistung_state = self.hass.states.get(self._thermische_leistung_sensor)
                if thermische_leistung_state and thermische_leistung_state.state not in ("unavailable", "unknown"):
                    try:
                        thermische_leistung = float(thermische_leistung_state.state)
                    except (ValueError, TypeError):
                        _LOGGER.warning("Could not parse thermische_leistung sensor value")

        except (ValueError, TypeError) as e:
            _LOGGER.error("Error reading sensor values: %s", e)
            return None

        else:
            return SensorValues(
                aussen_temp=aussen_temp,
                raum_ist=raum_ist,
                raum_soll=raum_soll,
                vorlauf_ist=vorlauf_ist,
                thermische_leistung=thermische_leistung,
            )

    async def _async_update_vorlauf_soll(self, now=None) -> None:
        """Update the Vorlauf-Soll value."""

        is_heating = await self._is_heating_mode()

        _LOGGER.info("_async_update_vorlauf_soll() controler.enabled: %d, is_heating: %d", self._controller.enabled, is_heating)

        try:
            if not self._controller.enabled:
                _LOGGER.info("Controller disabled - skipping update")
                return

            if not is_heating:
                _LOGGER.info("Not in heating mode - skipping update")
                return

            if self._heating_resumed_at is not None:
                time_since_resume = (dt_util.now() - self._heating_resumed_at).total_seconds() / 60
                if time_since_resume < 10.0:
                    _LOGGER.info(
                        "In stabilization (%.1f/10.0 min) - skipping update",
                        time_since_resume
                    )
                    return

            # Normal operation - proceed with prediction
            sensor_values = await self._async_get_sensor_values()
            if sensor_values is None:
                _LOGGER.warning("sensor_values is None, setting unavailable")
                self._available = False
                self.async_write_ha_state()
                return

            # Berechne neuen Vorlauf-Soll
            vorlauf_soll_and_features: VorlaufSollAndFeatures = await self.hass.async_add_executor_job(
                self._controller.berechne_vorlauf_soll,
                sensor_values,
            )

            # Track successful prediction time
            self._last_prediction_time = dt_util.now()

            # Update state
            self._attr_native_value = round(vorlauf_soll_and_features.vorlauf, 1)
            self._last_update = dt_util.now()
            self._next_update = self._last_update + timedelta(
                minutes=self._update_interval_minutes
            )
            self._available = True

            # Update attributes
            model_statistics = self._controller.get_model_statistics()

            # Prüfe aktuelle Betriebsart
            current_betriebsart = None
            if self._is_heating_entity:
                betriebsart_state = self.hass.states.get(self._is_heating_entity)
                if betriebsart_state and betriebsart_state.state not in (
                    "unavailable",
                    "unknown",
                ):
                    current_betriebsart = betriebsart_state.state

            self._extra_attributes = {
                ATTR_AUSSEN_TEMP: round(sensor_values.aussen_temp, 1),
                ATTR_RAUM_IST: round(sensor_values.raum_ist, 1),
                ATTR_RAUM_SOLL: round(sensor_values.raum_soll, 1),
                ATTR_VORLAUF_IST: round(sensor_values.vorlauf_ist, 1),
                ATTR_RAUM_ABWEICHUNG: round(vorlauf_soll_and_features.features.raum_abweichung, 2),
                ATTR_AUSSEN_TREND_1H: round(vorlauf_soll_and_features.features.aussen_trend_1h, 3),
                ATTR_MODEL_MAE: round(model_statistics.mae, 2),
                ATTR_LAST_UPDATE: self._last_update.isoformat(),
                ATTR_NEXT_UPDATE: self._next_update.isoformat(),
                "history_size": model_statistics.history_size,
                "betriebsart": current_betriebsart,
                # NEU: History-basiertes Lernen (persistiert über Reboots)
                "erfahrungen_gelernt": self._extra_attributes.get("erfahrungen_gelernt", 0)
            }

            # Prüfe ob Switch aktiv ist, dann sende an Wärmepumpe
            # Der Switch hat unique_id = {DOMAIN}_{entry_id}_aktiv
            switch_state = None
            switch_unique_id = f"{DOMAIN}_{self._config_entry.entry_id}_aktiv"

            # Suche Switch über unique_id in der Entity Registry
            entity_registry = er.async_get(self.hass)

            for entity_id in entity_registry.entities:
                entry = entity_registry.async_get(entity_id)
                if entry and entry.unique_id == switch_unique_id:
                    switch_state = self.hass.states.get(entity_id)
                    break

            # Fallback: Versuche direkt über konstruierte Entity-ID
            if not switch_state:
                switch_entity_id = f"switch.{DOMAIN}_aktiv"
                switch_state = self.hass.states.get(switch_entity_id)

            _LOGGER.debug(
                "Checking switch: unique_id=%s, entity_id=%s, state=%s",
                switch_unique_id,
                switch_state.entity_id if switch_state else "NOT_FOUND",
                switch_state.state if switch_state else "NOT_FOUND",
            )

            if switch_state and switch_state.state == "on":
                await self._async_set_vorlauf_soll(vorlauf_soll_and_features.vorlauf)
                _LOGGER.info("Control enabled, sending Vorlauf-Soll to heat pump")
            else:
                _LOGGER.info("Control disabled by switch, not sending Vorlauf-Soll")

            self.async_write_ha_state()

            _LOGGER.info(
                "Vorlauf-Soll updated: %.1f°C (Außen: %.1f°C, Raum: %.1f/%.1f°C)",
                vorlauf_soll_and_features.vorlauf,
                sensor_values.aussen_temp,
                sensor_values.raum_ist,
                sensor_values.raum_soll,
            )

            # Nach Berechnung speichern (Feature-Lernen passiert intern im Controller)
            await async_save_controller(self.hass, self._config_entry, self._controller)

        except Exception:
            _LOGGER.exception("Error updating Vorlauf-Soll")
            self._available = False
            self.async_write_ha_state()

    @callback
    def _async_heating_state_changed(self, event) -> None:
        """Handle heating mode state changes."""
        new_state = event.data.get("new_state")
        old_state = event.data.get("old_state")

        if new_state is None or old_state is None:
            return

        new_is_heating = new_state.state == STATE_ON
        old_is_heating = old_state.state == STATE_ON

        # Only react to transition from off to on
        if new_is_heating and not old_is_heating:
            _LOGGER.info("Heating resumed")

            # Check if update_interval has passed since last prediction
            if self._last_prediction_time is not None:
                time_since_last = (dt_util.now() - self._last_prediction_time).total_seconds() / 60
                if time_since_last < self._update_interval_minutes:
                    wait_time = self._update_interval_minutes - time_since_last
                    _LOGGER.info(
                        "Less than update_interval since last prediction: waiting %.1f more minutes (%.1f/%.1f min since last)",
                        wait_time,
                        time_since_last,
                        self._update_interval_minutes
                    )
                    # Stabilization will start at next scheduled update
                    return

            # update_interval passed or first run - start stabilization
            self._heating_resumed_at = dt_util.now()
            _LOGGER.info("Starting 10min stabilization period")
        elif not new_is_heating and old_is_heating:
            _LOGGER.info("Heating disabled")
            self._heating_resumed_at = None

    async def _async_set_vorlauf_soll(self, value: float) -> None:
        """Set the Vorlauf-Soll on the target entity."""

        _LOGGER.info("_async_set_vorlauf_soll()")

        try:
            target_entity = self.hass.states.get(self._vorlauf_soll_entity)
            if target_entity is None:
                _LOGGER.error("Target entity %s not found", self._vorlauf_soll_entity)
                return

            domain = target_entity.domain
            service_data = {
                "entity_id": self._vorlauf_soll_entity,
                "value": round(value, 1),
            }

            if domain == "input_number":
                await self.hass.services.async_call(
                    "input_number",
                    "set_value",
                    service_data,
                )
            elif domain == "number":
                await self.hass.services.async_call(
                    "number",
                    "set_value",
                    service_data,
                )
            elif domain == "climate":
                service_data = {
                    "entity_id": self._vorlauf_soll_entity,
                    "temperature": value,
                }
                await self.hass.services.async_call(
                    "climate",
                    "set_temperature",
                    service_data,
                )
            else:
                _LOGGER.warning("Unsupported target entity domain: %s", domain)

        except Exception as e:
            _LOGGER.error("Error setting Vorlauf-Soll: %s", e)

    def _format_sensor_as_float(self, state: State | None) -> str:
        """Konvertiert einen HA State in einen Float-String (2 Stellen) oder 'None'."""

        if state is None or state.state in ("unknown", "unavailable"):
            return "None"
        try:
            return f"{float(state.state):.2f}"
        except (ValueError, TypeError):
            return "Invalid"

    def _create_task(self, coro: Coroutine) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task
