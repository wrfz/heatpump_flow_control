"""Number platform for Heatpump Flow Control integration."""
import asyncio
from datetime import datetime, timedelta
import logging
from pathlib import Path
import pickle
from typing import Any, cast

from homeassistant.components.number import NumberEntity, NumberMode
from homeassistant.components.recorder import get_instance
from homeassistant.components.recorder.history import (
    get_significant_states_with_session,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import UnitOfTemperature
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

from .const import (
    ATTR_AUSSEN_TEMP,
    ATTR_AUSSEN_TREND_1H,
    ATTR_AUSSEN_TREND_2H,
    ATTR_AUSSEN_TREND_3H,
    ATTR_AUSSEN_TREND_6H,
    ATTR_LAST_UPDATE,
    ATTR_MODEL_MAE,
    ATTR_NEXT_UPDATE,
    ATTR_PREDICTIONS_COUNT,
    ATTR_RAUM_ABWEICHUNG,
    ATTR_RAUM_IST,
    ATTR_RAUM_SOLL,
    ATTR_VORLAUF_IST,
    CONF_AUSSEN_TEMP_SENSOR,
    CONF_BETRIEBSART_HEIZEN_WERT,
    CONF_BETRIEBSART_SENSOR,
    CONF_LEARNING_RATE,
    CONF_MAX_VORLAUF,
    CONF_MIN_VORLAUF,
    CONF_RAUM_IST_SENSOR,
    CONF_RAUM_SOLL_SENSOR,
    CONF_TREND_HISTORY_SIZE,
    CONF_UPDATE_INTERVAL,
    CONF_VORLAUF_IST_SENSOR,
    CONF_VORLAUF_SOLL_ENTITY,
    DEFAULT_BETRIEBSART_HEIZEN_WERT,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_VORLAUF,
    DEFAULT_MIN_VORLAUF,
    DEFAULT_TREND_HISTORY_SIZE,
    DEFAULT_UPDATE_INTERVAL,
    DOMAIN,
)
from .flow_controller import FlowController
from .types import SensorValues

# pylint: disable=hass-logger-capital, hass-logger-period
# ruff: noqa: BLE001
# ruff: logging-redundant-exc-info

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
        config: Mapping[str, Any],
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

        # Optional: Betriebsart-Sensor
        self._betriebsart_sensor = config.get(CONF_BETRIEBSART_SENSOR)
        self._betriebsart_heizen_wert = config.get(
            CONF_BETRIEBSART_HEIZEN_WERT, DEFAULT_BETRIEBSART_HEIZEN_WERT
        )

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

        # Flow Controller
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

        self._tasks: set[asyncio.Task[None]] = set()

    async def async_added_to_hass(self) -> None:
        """Run when entity about to be added to hass."""
        _LOGGER.info("async_added_to_hass()")

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
            self._async_update_vorlauf_soll,
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
        self._create_task(self._delayed_first_update())

    async def async_set_native_value(self, value: float) -> None:
        """Set new value (manual override)."""

        _LOGGER.info("async_set_native_value(%.2f)", value)

        self._attr_native_value = value
        self._last_vorlauf_soll = value

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

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return the state attributes."""

        _LOGGER.info("extra_state_attributes()")
        _LOGGER.info(self._extra_attributes)

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

    async def _async_load_model(self) -> None:
        """Lädt das Modell."""

        _LOGGER.info("_async_load_model()")

        def _load() -> FlowController | None:
            _LOGGER.info("Loading model from %s", self._model_path)

            if not self._model_path.exists():
                _LOGGER.info("No saved model found, starting fresh")
                return None

            try:
                with self._model_path.open("rb") as f:
                    loaded = pickle.load(f)
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
                "Model loaded (Predictions: %d)", loaded_controller.predictions_count
            )

    async def _async_save_model(self) -> None:
        """Speichert das Modell."""

        def _save():
            try:
                self._model_path.parent.mkdir(parents=True, exist_ok=True)

                with self._model_path.open("wb") as f:
                    _LOGGER.debug("Saving model to %s", self._model_path)
                    pickle.dump(self._controller, f)
            except Exception as err:
                _LOGGER.error("Error saving model: %s", err)

        await self.hass.async_add_executor_job(_save)

    @callback
    def _async_sensor_state_changed(self, event) -> None:
        """Handle sensor state changes for learning."""

        _LOGGER.info("_async_sensor_state_changed()")

        # Ignoriere unavailable/unknown states
        new_state = event.data.get("new_state")
        if new_state is None or new_state.state in ("unavailable", "unknown"):
            return

        # Wenn Number noch unavailable ist, prüfe ob jetzt alle verfügbar sind
        if not self._available:
            self._create_task(self._check_and_update_if_ready())

    async def _check_and_update_if_ready(self):
        """Check if all sensors are ready and trigger update."""

        _LOGGER.info("_check_and_update_if_ready()")

        sensor_values = await self._async_get_sensor_values()
        if sensor_values is not None:
            _LOGGER.info("All sensors now available, triggering update")
            await self._async_update_vorlauf_soll()

    async def _is_heating_mode(self) -> bool:
        """Check if heat pump is in heating mode.

        Returns:
            True if in heating mode or sensor not configured, False otherwise
        """
        if not self._betriebsart_sensor:
            return True  # Kein Sensor konfiguriert = immer lernen

        betriebsart_state = self.hass.states.get(self._betriebsart_sensor)
        if betriebsart_state is None:
            _LOGGER.warning(
                "Betriebsart sensor %s not found, assuming heating mode",
                self._betriebsart_sensor,
            )
            return True

        if betriebsart_state.state in ("unavailable", "unknown"):
            _LOGGER.debug(
                "Betriebsart sensor %s is %s, assuming heating mode",
                self._betriebsart_sensor,
                betriebsart_state.state,
            )
            return True

        current_mode = betriebsart_state.state.strip()
        is_heating = current_mode == self._betriebsart_heizen_wert

        if not is_heating:
            _LOGGER.debug(
                "Betriebsart is '%s' (expected '%s'), not in heating mode",
                current_mode,
                self._betriebsart_heizen_wert,
            )

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

        except (ValueError, TypeError) as e:
            _LOGGER.error("Error reading sensor values: %s", e)
            return None

        else:
            return SensorValues(aussen_temp=aussen_temp, raum_ist=raum_ist, raum_soll=raum_soll, vorlauf_ist=vorlauf_ist)

    async def _async_update_vorlauf_soll(self, now=None) -> None:
        """Update the Vorlauf-Soll value."""

        _LOGGER.info("_async_update_vorlauf_soll()")

        try:
            sensor_values = await self._async_get_sensor_values()
            if sensor_values is None:
                _LOGGER.warning("sensor_values is None, setting unavailable")
                self._available = False
                self.async_write_ha_state()
                return

            # Berechne neuen Vorlauf-Soll
            vorlauf_soll, features = await self.hass.async_add_executor_job(
                self._controller.berechne_vorlauf_soll,
                sensor_values,
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
            model_statistics = self._controller.get_model_statistics()

            # Prüfe aktuelle Betriebsart
            current_betriebsart = None
            if self._betriebsart_sensor:
                betriebsart_state = self.hass.states.get(self._betriebsart_sensor)
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
                ATTR_RAUM_ABWEICHUNG: round(features.raum_abweichung, 2),
                ATTR_AUSSEN_TREND_1H: round(features.aussen_trend_1h, 3),
                ATTR_AUSSEN_TREND_2H: round(features.aussen_trend_2h, 3),
                ATTR_AUSSEN_TREND_3H: round(features.aussen_trend_3h, 3),
                ATTR_AUSSEN_TREND_6H: round(features.aussen_trend_6h, 3),
                ATTR_MODEL_MAE: round(model_statistics.mae, 2),
                ATTR_PREDICTIONS_COUNT: model_statistics.predictions_count,
                ATTR_LAST_UPDATE: self._last_update.isoformat(),
                ATTR_NEXT_UPDATE: self._next_update.isoformat(),
                "use_fallback": model_statistics.use_fallback,
                "history_size": model_statistics.history_size,
                "betriebsart": current_betriebsart,
                "learning_enabled": current_betriebsart == self._betriebsart_heizen_wert
                if current_betriebsart
                else True,
                # NEU: History-basiertes Lernen (persistiert über Reboots)
                "history_learning": True,
                "erfahrungen_gelernt": self._extra_attributes.get("erfahrungen_gelernt", 0)
            }

            # Prüfe ob Switch aktiv ist, dann sende an Wärmepumpe
            # Der Switch hat unique_id = {DOMAIN}_{entry_id}_aktiv
            switch_state = None
            switch_unique_id = f"{DOMAIN}_{self._entry_id}_aktiv"

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
                await self._async_set_vorlauf_soll(vorlauf_soll)
                _LOGGER.info("Control enabled, sending Vorlauf-Soll to heat pump")
            else:
                _LOGGER.info("Control disabled by switch, not sending Vorlauf-Soll")

            self.async_write_ha_state()

            _LOGGER.info(
                "Vorlauf-Soll updated: %.1f°C (Außen: %.1f°C, Raum: %.1f/%.1f°C)",
                vorlauf_soll,
                sensor_values.aussen_temp,
                sensor_values.raum_ist,
                sensor_values.raum_soll,
            )

            # NEU: History-basiertes Lernen (alle 10 Zyklen)
            if self._controller.predictions_count % 10 == 0:
                await self._async_lerne_aus_history(sensor_values.raum_ist)

            # Nach Berechnung UND Lernen speichern
            await self._async_save_model()

        except Exception:
            _LOGGER.exception("Error updating Vorlauf-Soll")
            self._available = False
            self.async_write_ha_state()

    async def _async_lerne_aus_history(self, current_raum_ist: float) -> None:
        """Lernt aus Home Assistant History-Daten.

        Fragt Sensor-Werte von vor 2-6 Stunden ab und lernt daraus.
        """

        _LOGGER.info("_async_lerne_aus_history()")

        # Prüfe ob in Heizmodus
        if not await self._is_heating_mode():
            _LOGGER.debug("Nicht im Heizmodus, überspringe History-Lernen")
            return

        # Prüfe ob genug Zeit seit letztem Lernen vergangen ist (min 30 Min)
        now = dt_util.now()

        try:
            # Finde EINEN State von vor ~4h (Mitte des 2-6h Fensters)
            target_time = now - timedelta(hours=4)

            # Zeitfenster für Query: target_time ±30 Min (für nearest search)
            start_time = target_time - timedelta(minutes=30)
            end_time = target_time + timedelta(minutes=30)

            # Alle relevanten Entities
            entity_ids = [
                self._aussen_temp_sensor,
                self._raum_ist_sensor,
                self._raum_soll_sensor,
                self._vorlauf_ist_sensor,
                self.entity_id,  # Unsere eigene Entity (vorlauf_soll)
            ]

            _LOGGER.debug(
                "Query History um %s (±30min) für Entities: %s",
                target_time.strftime("%H:%M"),
                entity_ids,
            )

            # Query History über Recorder (Custom Integration kompatibel)
            def _get_history():
                """Query history in executor."""
                instance = get_instance(self.hass)
                with instance.get_session() as session:
                    return get_significant_states_with_session(
                        self.hass,
                        session,
                        start_time,
                        end_time,
                        entity_ids,
                        significant_changes_only=False,
                    )

            history_data = await self.hass.async_add_executor_job(_get_history)

            if not history_data:
                _LOGGER.debug("Keine History-Daten gefunden")
                return

            # Extrahiere einen Historical State (nearest zu target_time)
            historical_state = self._extract_historical_state(
                history_data,
                target_time
            )

            if not historical_state:
                _LOGGER.debug("Kein Historical State gefunden")
                return

            # Prüfe ob wir diesen State schon gelernt haben
            state_timestamp = historical_state['timestamp']

            _LOGGER.info(
                "Lerne aus State von %s (vor %.1fh)",
                state_timestamp.strftime("%H:%M"),
                (now - state_timestamp).total_seconds() / 3600,
            )

            # Führe Lernen aus (mit EINEM State)
            success = self._controller.lerne_aus_history(historical_state, current_raum_ist)

            if success:
                _LOGGER.info("History-Lernen erfolgreich")
                # Stats in Attributes speichern
                self._extra_attributes["erfahrungen_gelernt"] = (
                    self._extra_attributes.get("erfahrungen_gelernt", 0) + 1
                )
                await self._async_save_model()

        except Exception as e:
            _LOGGER.error("Fehler beim History-Lernen: %s", e, exc_info=True)

    def _extract_historical_state(
        self,
        history_data: dict[str, Any],
        target_time: datetime
    ) -> dict[str, Any] | None:
        """Extrahiert einen Historical State aus HA History.

        Findet den State der target_time am nächsten ist.

        Args:
            history_data: HA History Daten
            target_time: Zielzeit (~4h ago)

        Returns:
            State dict oder None wenn nicht gefunden
        """

        # Erstelle Maps: entity_id -> list of states
        aussen_temp_states = history_data.get(self._aussen_temp_sensor, [])
        raum_ist_states = history_data.get(self._raum_ist_sensor, [])
        raum_soll_states = history_data.get(self._raum_soll_sensor, [])
        vorlauf_ist_states = history_data.get(self._vorlauf_ist_sensor, [])
        vorlauf_soll_states = history_data.get(self.entity_id, [])

        if not vorlauf_soll_states:
            _LOGGER.debug("Keine Vorlauf-Soll History gefunden (Entity: %s)", self.entity_id)
            return None

        # Finde den vorlauf_soll State der target_time am nächsten ist
        best_state = None
        best_diff: float | None = None

        for state in vorlauf_soll_states:
            if state.state in ("unavailable", "unknown"):
                continue

            diff: float = abs((state.last_updated - target_time).total_seconds())
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_state = state

        if not best_state:
            _LOGGER.debug("Kein Vorlauf-Soll State nahe target_time gefunden")
            return None

        try:
            timestamp = best_state.last_updated
            vorlauf_soll = float(best_state.state)

            # Finde nearest states für alle anderen Sensoren zum DAMALIGEN Zeitpunkt (±5 Min)
            tolerance = timedelta(minutes=5)

            aussen_temp = self._find_nearest_state_value(aussen_temp_states, timestamp, tolerance)
            raum_ist = self._find_nearest_state_value(raum_ist_states, timestamp, tolerance)
            raum_soll = self._find_nearest_state_value(raum_soll_states, timestamp, tolerance)
            vorlauf_ist = self._find_nearest_state_value(vorlauf_ist_states, timestamp, tolerance)

            # Async Query (kann nicht in sync context, daher machen wir es inline)
            # Für jetzt: nutze states die bereits in history_data sind
            raum_ist_spaeter = None

            # Nur wenn alle Werte verfügbar (inkl. raum_ist_spaeter!)
            if all(v is not None for v in [aussen_temp, raum_ist, raum_soll, vorlauf_ist, raum_ist_spaeter]):
                return {
                    'timestamp': timestamp,
                    'aussen_temp': aussen_temp,
                    'raum_ist': raum_ist,
                    'raum_soll': raum_soll,
                    'vorlauf_ist': vorlauf_ist,
                    'vorlauf_soll': vorlauf_soll,
                    'raum_ist_spaeter': raum_ist_spaeter,  # 2-6h später!
                }
            else:
                _LOGGER.debug(
                    "Überspringe State von %s: Nicht alle Werte verfügbar (raum_ist_spaeter=%s)",
                    timestamp.strftime("%H:%M"),
                    raum_ist_spaeter,
                )
                return None

        except (ValueError, TypeError) as e:
            _LOGGER.debug("Fehler beim Extrahieren: %s", e)
            return None

    def _find_nearest_state_value(
        self,
        states: list[State],
        target_time: datetime,
        tolerance: timedelta,
    ) -> float | None:
        """Findet den nächsten State-Wert zu einer Zielzeit.

        Args:
            states: Liste von States
            target_time: Zielzeitpunkt
            tolerance: Maximale zeitliche Abweichung

        Returns:
            Float-Wert oder None wenn nicht gefunden
        """

        if not states:
            return None

        # Finde State mit kleinster Zeitdifferenz
        best_state = None
        best_diff = None

        for state in states:
            if state.state in ("unavailable", "unknown"):
                continue

            time_diff = abs((state.last_updated - target_time).total_seconds())

            if time_diff <= tolerance.total_seconds():
                if best_diff is None or time_diff < best_diff:
                    best_diff = time_diff
                    best_state = state

        if best_state:
            try:
                return float(best_state.state)
            except (ValueError, TypeError):
                return None

        return None

    async def _async_set_vorlauf_soll(self, value: float) -> None:
        """Set the Vorlauf-Soll on the target entity."""

        _LOGGER.info("_async_update_vorlauf_soll()")

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
