"""Config flow for Heatpump Flow Control integration."""

import logging

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.helpers import selector
from homeassistant.helpers.selector import NumberSelectorMode

from .const import (
    DEFAULT_MAX_THERMISCHE_LEISTUNG,
    DEFAULT_MAX_VORLAUF_HI,
    DEFAULT_MAX_VORLAUF_LO,
    DEFAULT_MIN_THERMISCHE_LEISTUNG,
    DEFAULT_MIN_VORLAUF_HI,
    DEFAULT_MIN_VORLAUF_LO,
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)


def get_config_schema(defaults=None):
    """Build config schema with optional defaults."""
    if defaults is None:
        defaults = {}

    return vol.Schema(
        {
            vol.Required(
                "aussen_temp_sensor", default=defaults.get("aussen_temp_sensor")
            ): selector.EntitySelector(selector.EntitySelectorConfig(domain="sensor")),
            vol.Required(
                "raum_ist_sensor", default=defaults.get("raum_ist_sensor")
            ): selector.EntitySelector(selector.EntitySelectorConfig(domain="sensor")),
            vol.Required(
                "raum_soll_sensor", default=defaults.get("raum_soll_sensor")
            ): selector.EntitySelector(
                selector.EntitySelectorConfig(
                    domain=["input_number", "climate", "sensor"]
                )
            ),
            vol.Required(
                "vorlauf_ist_sensor", default=defaults.get("vorlauf_ist_sensor")
            ): selector.EntitySelector(selector.EntitySelectorConfig(domain="sensor")),
            vol.Required(
                "vorlauf_soll_entity", default=defaults.get("vorlauf_soll_entity")
            ): selector.EntitySelector(
                selector.EntitySelectorConfig(
                    domain=["input_number", "number", "climate"]
                )
            ),
            vol.Required(
                "is_heating_entity", default=defaults.get("is_heating_entity")
            ): selector.EntitySelector(selector.EntitySelectorConfig(domain="binary_sensor")),
            vol.Optional(
                "thermische_leistung_sensor", default=defaults.get("thermische_leistung_sensor")
            ): selector.EntitySelector(selector.EntitySelectorConfig(domain="sensor")),
            vol.Optional(
                "min_vorlauf", default=defaults.get("min_vorlauf", 25.0)
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=DEFAULT_MIN_VORLAUF_LO,
                    max=DEFAULT_MIN_VORLAUF_HI,
                    step=1.0,
                    mode=NumberSelectorMode.BOX,
                    unit_of_measurement="°C",
                )
            ),
            vol.Optional(
                "max_vorlauf", default=defaults.get("max_vorlauf", 55.0)
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=DEFAULT_MAX_VORLAUF_LO,
                    max=DEFAULT_MAX_VORLAUF_HI,
                    step=1.0,
                    mode=NumberSelectorMode.BOX,
                    unit_of_measurement="°C",
                )
            ),
            vol.Optional(
                "min_thermische_leistung", default=defaults.get("min_thermische_leistung", DEFAULT_MIN_THERMISCHE_LEISTUNG)
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=0.1,
                    max=10.0,
                    step=0.1,
                    mode=NumberSelectorMode.BOX,
                    unit_of_measurement="kW",
                )
            ),
            vol.Optional(
                "max_thermische_leistung", default=defaults.get("max_thermische_leistung", DEFAULT_MAX_THERMISCHE_LEISTUNG)
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=0.5,
                    max=20.0,
                    step=0.1,
                    mode=NumberSelectorMode.BOX,
                    unit_of_measurement="kW",
                )
            ),
            vol.Optional(
                "update_interval", default=defaults.get("update_interval", 60)
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=1,
                    max=240,
                    step=1,
                    mode=NumberSelectorMode.BOX,
                    unit_of_measurement="Minuten",
                )
            ),
            vol.Optional(
                "learning_rate", default=defaults.get("learning_rate", 0.01)
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=0.001, max=0.1, step=0.001, mode=NumberSelectorMode.BOX
                )
            ),
        }
    )


class FlowControlConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Heatpump Flow Control."""

    VERSION = 1

    async def async_step_user(self, user_input=None):
        """Handle the initial step."""
        errors = {}

        if user_input is not None:
            try:
                return self.async_create_entry(
                    title="Wärmepumpen ML Regelung",
                    data=user_input,
                )
            except Exception as e:  # noqa: BLE001
                _LOGGER.error("Error creating entry: %s", e)
                errors["base"] = "unknown"

        return self.async_show_form(
            step_id="user",
            data_schema=get_config_schema(),
            errors=errors,
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry):
        """Get the options flow for this handler."""
        return FlowControlOptionsFlow(config_entry)


class FlowControlOptionsFlow(config_entries.OptionsFlow):
    """Handle options flow."""

    def __init__(self, config_entry) -> None:
        """Initialize options flow."""
        self._config_entry = config_entry

    async def async_step_init(self, user_input=None):
        """Manage the options."""
        if user_input is not None:
            self.hass.config_entries.async_update_entry(
                self._config_entry, data={**self._config_entry.data, **user_input}
            )
            return self.async_create_entry(title="", data={})

        return self.async_show_form(
            step_id="init",
            data_schema=get_config_schema(defaults=self._config_entry.data),
        )
