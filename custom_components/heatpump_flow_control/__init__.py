"""Heatpump Flow Control Integration für Home Assistant."""

import logging
from pathlib import Path
import pickle

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .const import (
    CONF_LEARNING_RATE,
    CONF_MAX_THERMISCHE_LEISTUNG,
    CONF_MAX_VORLAUF,
    CONF_MIN_THERMISCHE_LEISTUNG,
    CONF_MIN_VORLAUF,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_THERMISCHE_LEISTUNG,
    DEFAULT_MAX_VORLAUF_LO,
    DEFAULT_MIN_THERMISCHE_LEISTUNG,
    DEFAULT_MIN_VORLAUF_HI,
)
from .flow_controller import PICKLE_VERSION, FlowController

# pylint: disable=hass-logger-capital, hass-logger-period
# ruff: logging-redundant-exc-info

_LOGGER = logging.getLogger(__name__)

DOMAIN = "heatpump_flow_control"
PLATFORMS = ["number", "switch"]

# Custom ConfigEntry type for runtime_data
type HeatpumpFlowControlConfigEntry = ConfigEntry[FlowController]


async def async_setup(hass: HomeAssistant, config: dict) -> bool:
    """Set up the Heatpump Flow Control component."""
    return True


async def async_setup_entry(
    hass: HomeAssistant,
    entry: HeatpumpFlowControlConfigEntry
) -> bool:
    """Set up Heatpump Flow Control from a config entry."""
    _LOGGER.info("async_setup_entry()")

    # Load or create FlowController
    controller = await _async_load_or_create_controller(hass, entry)

    # Store in runtime_data
    entry.runtime_data = controller

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    # Listen for options updates
    entry.async_on_unload(entry.add_update_listener(update_listener))

    return True


async def _async_load_or_create_controller(
    hass: HomeAssistant, entry: ConfigEntry
) -> FlowController:
    """Load controller from disk or create new one."""

    # Get configuration
    min_vorlauf = entry.data.get(CONF_MIN_VORLAUF, DEFAULT_MIN_VORLAUF_HI)
    max_vorlauf = entry.data.get(CONF_MAX_VORLAUF, DEFAULT_MAX_VORLAUF_LO)
    learning_rate = entry.data.get(CONF_LEARNING_RATE, DEFAULT_LEARNING_RATE)
    min_thermische_leistung = entry.data.get(CONF_MIN_THERMISCHE_LEISTUNG, DEFAULT_MIN_THERMISCHE_LEISTUNG)
    max_thermische_leistung = entry.data.get(CONF_MAX_THERMISCHE_LEISTUNG, DEFAULT_MAX_THERMISCHE_LEISTUNG)

    # Model file path
    model_file_name = f"{DOMAIN}.model.pkl"
    model_path = Path(hass.config.path(model_file_name))

    def _load() -> FlowController | None:
        """Load model from disk."""
        _LOGGER.info("Loading model from %s", model_path)

        if not model_path.exists():
            _LOGGER.info("No saved model found, starting fresh")
            return None

        try:
            with model_path.open("rb") as f:
                loaded = pickle.load(f)

                # Check pickle version for migration
                loaded_version = getattr(loaded, "pickle_version", 1)
                if loaded_version < PICKLE_VERSION:
                    _LOGGER.warning(
                        "Model version mismatch (loaded: %d, current: %d). "
                        "Starting fresh to ensure compatibility.",
                        loaded_version,
                        PICKLE_VERSION,
                    )
                    return None

                # Migration code for missing attributes (if version matches)
                if not hasattr(loaded, "use_fallback"):
                    _LOGGER.info("Migrating: Adding use_fallback attribute")
                    loaded.use_fallback = False

                # Legacy patches for old models
                loaded.use_fallback = False
                loaded.min_predictions_for_model = 0

                _LOGGER.info(
                    "Model loaded successfully (version %d)",
                    loaded_version,
                )
                return loaded
        except (pickle.UnpicklingError, EOFError, AttributeError) as err:
            _LOGGER.error(
                "Cannot load model (incompatible format or missing attributes): %s. "
                "Starting fresh.",
                err,
            )
            return None

    # Load controller from disk
    loaded_controller = await hass.async_add_executor_job(_load)

    if loaded_controller:
        # Update configuration parameters from current config
        loaded_controller.update_config(
            min_vorlauf=min_vorlauf,
            max_vorlauf=max_vorlauf,
            learning_rate=learning_rate,
            min_thermische_leistung=min_thermische_leistung,
            max_thermische_leistung=max_thermische_leistung,
        )
    else:
        # Create new controller
        _LOGGER.info("Creating new FlowController")
        loaded_controller = FlowController(
            min_vorlauf=min_vorlauf,
            max_vorlauf=max_vorlauf,
            learning_rate=learning_rate,
            min_thermische_leistung=min_thermische_leistung,
            max_thermische_leistung=max_thermische_leistung,
        )
        loaded_controller.setup(iterations=10)

    return loaded_controller


async def async_save_controller(
    hass: HomeAssistant,
    entry: ConfigEntry,
    controller: FlowController
) -> None:
    """Save controller to disk."""

    model_file_name = f"{DOMAIN}.model.pkl"
    model_path = Path(hass.config.path(model_file_name))

    def _save():
        """Save model to disk."""
        try:
            model_path.parent.mkdir(parents=True, exist_ok=True)

            with model_path.open("wb") as f:
                _LOGGER.debug("Saving model to %s", model_path)
                pickle.dump(controller, f)
        except Exception as err:  # noqa: BLE001
            _LOGGER.error("Error saving model: %s", err)

    await hass.async_add_executor_job(_save)


async def update_listener(
    hass: HomeAssistant, entry: HeatpumpFlowControlConfigEntry
) -> None:
    """Handle options update."""
    _LOGGER.info("update_listener()")
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(
    hass: HomeAssistant, entry: HeatpumpFlowControlConfigEntry
) -> bool:
    """Unload a config entry."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
