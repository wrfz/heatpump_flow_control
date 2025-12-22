"""Number platform for Heatpump Flow Control integration."""

import asyncio
from datetime import timedelta
import logging
from pathlib import Path
import pickle
from typing import Any

from homeassistant.components.number import NumberEntity, NumberMode
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import UnitOfTemperature
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.event import (
    async_track_state_change_event,
    async_track_time_interval,
)
from homeassistant.helpers.restore_state import RestoreEntity
from homeassistant.util import dt as dt_util

from .const import (
    ATTR_AUSSEN_TEMP,
    ATTR_AUSSEN_TREND,
    ATTR_AUSSEN_TREND_5MIN,
    ATTR_AUSSEN_TREND_30MIN,
    ATTR_LAST_UPDATE,
    ATTR_MODEL_MAE,
    ATTR_NEXT_UPDATE,
    ATTR_PREDICTIONS_COUNT,
    ATTR_RAUM_ABWEICHUNG,
    ATTR_RAUM_IST,
    ATTR_RAUM_SOLL,
    ATTR_VORLAUF_IST,
    CONF_AUSSEN_TEMP_SENSOR,
    CONF_LEARNING_RATE,
    CONF_MAX_VORLAUF,
    CONF_MIN_VORLAUF,
    CONF_RAUM_IST_SENSOR,
    CONF_RAUM_SOLL_SENSOR,
    CONF_TREND_HISTORY_SIZE,
    CONF_UPDATE_INTERVAL,
    CONF_VORLAUF_IST_SENSOR,
    CONF_VORLAUF_SOLL_ENTITY,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_VORLAUF,
    DEFAULT_MIN_VORLAUF,
    DEFAULT_TREND_HISTORY_SIZE,
    DEFAULT_UPDATE_INTERVAL,
    DOMAIN,
)
from .flow_controller import FlowController

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the Flow Control number."""
    config = config_entry.data

    number = FlowControlNumber(
        hass=hass,
        config=config,
        entry_id=config_entry.entry_id,
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
        config: dict[str, Any],
        entry_id: str,
    ) -> None:
        """Initialize the number."""
        self.hass = hass
        self._config = config
        self._entry_id = entry_id

        # Sensor Entities
        self._aussen_temp_sensor = config[CONF_AUSSEN_TEMP_SENSOR]
        self._raum_ist_sensor = config[CONF_RAUM_IST_SENSOR]
        self._raum_soll_sensor = config[CONF_RAUM_SOLL_SENSOR]
        self._vorlauf_ist_sensor = config[CONF_VORLAUF_IST_SENSOR]
        self._vorlauf_soll_entity = config[CONF_VORLAUF_SOLL_ENTITY]

        # Konfiguration
        self._min_vorlauf = config.get(CONF_MIN_VORLAUF, DEFAULT_MIN_VORLAUF)
        self._max_vorlauf = config.get(CONF_MAX_VORLAUF, DEFAULT_MAX_VORLAUF)
        self._update_interval_minutes = config.get(
            CONF_UPDATE_INTERVAL, DEFAULT_UPDATE_INTERVAL
        )
        self._learning_rate = config.get(CONF_LEARNING_RATE, DEFAULT_LEARNING_RATE)
        self._trend_history_size = config.get(
            CONF_TREND_HISTORY_SIZE, DEFAULT_TREND_HISTORY_SIZE
        )

        # Pfad für die Model-Datei
        model_file_name = f"{DOMAIN}_{entry_id}.model.pkl"
        self._model_path = Path(self.hass.config.path(model_file_name))

        # ML Controller
        self._controller = FlowController(
            min_vorlauf=self._min_vorlauf,
            max_vorlauf=self._max_vorlauf,
            learning_rate=self._learning_rate,
            trend_history_size=self._trend_history_size,
        )

        # State
        self._attr_native_value = None
        self._last_vorlauf_soll = None
        self._extra_attributes = {}
        self._last_update = None
        self._next_update = None
        self._available = False

        # Listener
        self._update_interval_listener = None
        self._state_change_listener = None

        # Number properties
        self._attr_name = "Heatpump Flow Control Vorlauf Soll"
        self._attr_unique_id = f"{DOMAIN}_{entry_id}_vorlauf_soll"
        self._attr_native_min_value = self._min_vorlauf
        self._attr_native_max_value = self._max_vorlauf
        self._attr_native_step = 0.5

    async def async_added_to_hass(self) -> None:
        """Run when entity about to be added to hass."""
        # ML Model laden
        await self._async_load_model()

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
            self._async_periodic_update,
            update_interval,
        )

        # Setup state change listener
        self._state_change_listener = async_track_state_change_event(
            self.hass,
            [
                self._aussen_temp_sensor,
                self._raum_ist_sensor,
                self._raum_soll_sensor,
                self._vorlauf_ist_sensor,
            ],
            self._async_sensor_state_changed,
        )

        # Sofortiges erstes Update nach 5 Sekunden
        asyncio.create_task(self._delayed_first_update())

    async def _delayed_first_update(self):
        """First update after startup with delay."""
        await asyncio.sleep(5)
        await self._async_update_vorlauf_soll()

    async def _async_load_model(self) -> None:
        """Lädt das Modell."""

        def _load():
            _LOGGER.info("Loading ML model from %s", self._model_path)

            if not self._model_path.exists():
                _LOGGER.info("No saved model found, starting fresh")
                return None

            try:
                with self._model_path.open("rb") as f:
                    loaded = pickle.load(f)
                    # Erzwinge Model-Nutzung nach dem Laden
                    loaded.use_fallback = False
                    loaded.min_predictions_for_model = 0
                    return loaded
            except (pickle.UnpicklingError, EOFError, AttributeError) as err:
                _LOGGER.error("Corrupted model found, starting fresh: %s", err)
                return None

        loaded_controller = await self.hass.async_add_executor_job(_load)
        if loaded_controller:
            self._controller = loaded_controller
            _LOGGER.info(
                "ML model loaded (Predictions: %d)", loaded_controller.predictions_count
            )

    async def _async_save_model(self) -> None:
        """Speichert das Modell."""

        def _save():
            try:
                self._model_path.parent.mkdir(parents=True, exist_ok=True)

                with self._model_path.open("wb") as f:
                    _LOGGER.debug("Saving ML model to %s", self._model_path)
                    pickle.dump(self._controller, f)
            except Exception as err:
                _LOGGER.error("Error saving ML model: %s", err)

        await self.hass.async_add_executor_job(_save)

    async def async_will_remove_from_hass(self) -> None:
        """Run when entity will be removed from hass."""
        if self._update_interval_listener:
            self._update_interval_listener()
        if self._state_change_listener:
            self._state_change_listener()

    @callback
    def _async_sensor_state_changed(self, event) -> None:
        """Handle sensor state changes for learning."""

        # Ignoriere unavailable/unknown states
        new_state = event.data.get("new_state")
        if new_state is None or new_state.state in ("unavailable", "unknown"):
            return

        # Wenn Number noch unavailable ist, prüfe ob jetzt alle verfügbar sind
        if not self._available:
            asyncio.create_task(self._check_and_update_if_ready())

        # Nur lernen, nicht neu berechnen (das passiert stündlich)
        asyncio.create_task(self._async_learn_from_current_state())

    async def _check_and_update_if_ready(self):
        """Check if all sensors are ready and trigger update."""
        sensor_values = await self._async_get_sensor_values()
        if sensor_values is not None:
            _LOGGER.info("All sensors now available, triggering update")
            await self._async_update_vorlauf_soll()

    async def _async_learn_from_current_state(self) -> None:
        """Learn from current sensor states."""
        try:
            sensor_values = await self._async_get_sensor_values()
            if sensor_values is None:
                return

            aussen_temp = sensor_values["aussen_temp"]
            raum_ist = sensor_values["raum_ist"]
            raum_soll = sensor_values["raum_soll"]
            vorlauf_ist = sensor_values["vorlauf_ist"]

            # Lerne nur, wenn wir einen Vorlauf-Soll gesetzt haben
            if self._last_vorlauf_soll is not None:
                await self.hass.async_add_executor_job(
                    self._controller.lerne,
                    aussen_temp,
                    raum_ist,
                    raum_soll,
                    vorlauf_ist,
                    self._last_vorlauf_soll,
                )

        except Exception as e:
            _LOGGER.error("Error during learning: %s", e)

    async def _async_periodic_update(self, now=None) -> None:
        """Periodic update callback."""
        await self._async_update_vorlauf_soll()

    async def _async_get_sensor_values(self) -> dict[str, float] | None:
        """Get current values from all sensors."""
        try:
            aussen_temp_state = self.hass.states.get(self._aussen_temp_sensor)
            raum_ist_state = self.hass.states.get(self._raum_ist_sensor)
            raum_soll_state = self.hass.states.get(self._raum_soll_sensor)
            vorlauf_ist_state = self.hass.states.get(self._vorlauf_ist_sensor)

            if not all(
                [aussen_temp_state, raum_ist_state, raum_soll_state, vorlauf_ist_state]
            ):
                _LOGGER.warning("One or more sensor states unavailable")
                return None

            # Prüfe auf unavailable/unknown
            if aussen_temp_state.state in ("unavailable", "unknown"):
                _LOGGER.warning(
                    "Sensor %s is %s", self._aussen_temp_sensor, aussen_temp_state.state
                )
                return None
            if raum_ist_state.state in ("unavailable", "unknown"):
                _LOGGER.warning(
                    "Sensor %s is %s", self._raum_ist_sensor, raum_ist_state.state
                )
                return None
            if vorlauf_ist_state.state in ("unavailable", "unknown"):
                _LOGGER.warning(
                    "Sensor %s is %s", self._vorlauf_ist_sensor, vorlauf_ist_state.state
                )
                return None

            # Extract temperature values
            aussen_temp = float(aussen_temp_state.state)
            raum_ist = float(raum_ist_state.state)
            vorlauf_ist = float(vorlauf_ist_state.state)

            # Raum-Soll kann von Climate-Entity oder Input-Number kommen
            raum_soll_value = raum_soll_state.state
            if raum_soll_state.domain == "climate":
                raum_soll = float(raum_soll_state.attributes.get("temperature", 21.0))
            else:
                raum_soll = float(raum_soll_value)

            return {
                "aussen_temp": aussen_temp,
                "raum_ist": raum_ist,
                "raum_soll": raum_soll,
                "vorlauf_ist": vorlauf_ist,
            }

        except (ValueError, TypeError) as e:
            _LOGGER.error("Error reading sensor values: %s", e)
            return None

    async def _async_update_vorlauf_soll(self) -> None:
        """Update the Vorlauf-Soll value."""
        try:
            _LOGGER.info("_async_update_vorlauf_soll()")

            sensor_values = await self._async_get_sensor_values()
            if sensor_values is None:
                _LOGGER.warning("sensor_values is None, setting unavailable")
                self._available = False
                self.async_write_ha_state()
                return

            aussen_temp = sensor_values["aussen_temp"]
            raum_ist = sensor_values["raum_ist"]
            raum_soll = sensor_values["raum_soll"]
            vorlauf_ist = sensor_values["vorlauf_ist"]

            # Berechne neuen Vorlauf-Soll
            vorlauf_soll, features = await self.hass.async_add_executor_job(
                self._controller.berechne_vorlauf_soll,
                aussen_temp,
                raum_ist,
                raum_soll,
                vorlauf_ist,
            )

            # Update state
            self._attr_native_value = round(vorlauf_soll, 1)
            self._last_vorlauf_soll = vorlauf_soll
            self._last_update = dt_util.now()
            self._next_update = self._last_update + timedelta(
                minutes=self._update_interval_minutes
            )
            self._available = True

            # Update attributes
            model_stats = self._controller.get_model_stats()
            self._extra_attributes = {
                ATTR_AUSSEN_TEMP: round(aussen_temp, 1),
                ATTR_RAUM_IST: round(raum_ist, 1),
                ATTR_RAUM_SOLL: round(raum_soll, 1),
                ATTR_VORLAUF_IST: round(vorlauf_ist, 1),
                ATTR_RAUM_ABWEICHUNG: round(features["raum_abweichung"], 2),
                ATTR_AUSSEN_TREND: round(features["aussen_trend"], 3),
                ATTR_AUSSEN_TREND_5MIN: round(features["aussen_trend_kurz"], 3),
                ATTR_AUSSEN_TREND_30MIN: round(features["aussen_trend_mittel"], 3),
                ATTR_MODEL_MAE: round(model_stats["mae"], 2),
                ATTR_PREDICTIONS_COUNT: model_stats["predictions_count"],
                ATTR_LAST_UPDATE: self._last_update.isoformat(),
                ATTR_NEXT_UPDATE: self._next_update.isoformat(),
                "use_fallback": model_stats["use_fallback"],
                "history_size": model_stats["history_size"],
            }

            # Nach der Berechnung speichern
            await self._async_save_model()

            # Prüfe ob Switch aktiv ist, dann sende an Wärmepumpe
            switch_entity_id = f"switch.{DOMAIN}_aktiv"
            switch_state = self.hass.states.get(switch_entity_id)

            if switch_state and switch_state.state == "on":
                await self._async_set_vorlauf_soll(vorlauf_soll)
                _LOGGER.info("Control enabled, sending Vorlauf-Soll to heat pump")
            else:
                _LOGGER.info("Control disabled by switch, not sending Vorlauf-Soll")

            self.async_write_ha_state()

            _LOGGER.info(
                "Vorlauf-Soll updated: %.1f°C (Außen: %.1f°C, Raum: %.1f/%.1f°C)",
                vorlauf_soll,
                aussen_temp,
                raum_ist,
                raum_soll,
            )

        except Exception as e:
            _LOGGER.error("Error updating Vorlauf-Soll: %s", e, exc_info=True)
            self._available = False
            self.async_write_ha_state()

    async def _async_set_vorlauf_soll(self, value: float) -> None:
        """Set the Vorlauf-Soll on the target entity."""
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

    async def async_set_native_value(self, value: float) -> None:
        """Set new value (manual override)."""
        self._attr_native_value = value
        self._last_vorlauf_soll = value

        # Bei manueller Änderung auch direkt senden (unabhängig vom Switch)
        await self._async_set_vorlauf_soll(value)
        self.async_write_ha_state()

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return the state attributes."""
        return self._extra_attributes

    @property
    def available(self) -> bool:
        """Return True if entity is available."""
        return self._available
