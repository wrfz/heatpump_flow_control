"""Config flow for Heatpump Flow Control integration."""

import logging

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.helpers import selector
from homeassistant.helpers.selector import NumberSelectorMode

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)


def get_config_schema(defaults=None):
    """Build config schema with optional defaults."""
    if defaults is None:
        defaults = {}

    # Power sensor dynamisch hinzufügen - nur mit default wenn Wert vorhanden
    power_sensor_value = defaults.get("power_sensor")
    if power_sensor_value and power_sensor_value != "":
        # Wert vorhanden: mit default
        power_sensor_field = vol.Optional(
            "power_sensor", default=power_sensor_value
        )
    else:
        # Kein Wert: ohne default (bleibt leer)
        power_sensor_field = vol.Optional("power_sensor")

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
            vol.Optional(
                "betriebsart_sensor", default=defaults.get("betriebsart_sensor")
            ): selector.EntitySelector(selector.EntitySelectorConfig(domain="sensor")),
            vol.Optional(
                "betriebsart_heizen_wert",
                default=defaults.get("betriebsart_heizen_wert", "Heizen"),
            ): selector.TextSelector(
                selector.TextSelectorConfig(type=selector.TextSelectorType.TEXT)
            ),
            power_sensor_field: selector.EntitySelector(
                selector.EntitySelectorConfig(
                    domain="sensor",
                    multiple=False,
                )
            ),
            vol.Optional(
                "min_vorlauf", default=defaults.get("min_vorlauf", 25.0)
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=28.0,
                    max=32.0,
                    step=1.0,
                    mode=NumberSelectorMode.BOX,
                    unit_of_measurement="°C",
                )
            ),
            vol.Optional(
                "max_vorlauf", default=defaults.get("max_vorlauf", 55.0)
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=35.0,
                    max=40.0,
                    step=1.0,
                    mode=NumberSelectorMode.BOX,
                    unit_of_measurement="°C",
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
            vol.Optional(
                "trend_history_size", default=defaults.get("trend_history_size", 12)
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=6, max=48, step=1, mode=NumberSelectorMode.BOX
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
                # Entferne leere/None power_sensor Werte (optional)
                if "power_sensor" in user_input and not user_input["power_sensor"]:
                    user_input["power_sensor"] = None

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
            # Entferne leere/None power_sensor Werte (optional)
            if "power_sensor" in user_input and not user_input["power_sensor"]:
                user_input["power_sensor"] = None

            self.hass.config_entries.async_update_entry(
                self._config_entry, data={**self._config_entry.data, **user_input}
            )
            return self.async_create_entry(title="", data={})

        return self.async_show_form(
            step_id="init",
            data_schema=get_config_schema(defaults=self._config_entry.data),
        )
