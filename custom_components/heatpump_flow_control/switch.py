"""Switch platform for Heatpump Flow Control integration."""

import logging

from homeassistant.components.switch import SwitchEntity
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.restore_state import RestoreEntity

from . import HeatpumpFlowControlConfigEntry
from .common import get_device_info
from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)

async def async_setup_entry(
    hass: HomeAssistant,
    entry: HeatpumpFlowControlConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the Flow Control switch."""
    _LOGGER.info("switch: setting up")

    async_add_entities([FlowControlActiveSwitch(entry)], True)


class FlowControlActiveSwitch(SwitchEntity, RestoreEntity):
    """Representation of Flow Control Active Switch."""

    def __init__(self, entry: HeatpumpFlowControlConfigEntry) -> None:
        """Initialize the switch."""
        self._entry = entry

        # Basis-Attribute
        self._attr_name = "Heatpump Flow Control Aktiv"
        self._attr_unique_id = f"{DOMAIN}_{entry.entry_id}_aktiv"
        self._attr_is_on = True
        self._attr_icon = "mdi:play-circle"

        self._attr_device_info = get_device_info(entry.entry_id)

    @property
    def _controller(self):
        """Get FlowController from runtime_data."""
        return self._entry.runtime_data

    async def async_added_to_hass(self) -> None:
        """Restore last state."""
        await super().async_added_to_hass()

        last_state = await self.async_get_last_state()
        if last_state is not None:
            self._attr_is_on = last_state.state == "on"
            # Synchronisiere den Controller-Zustand beim Start
            self._controller.enabled = self._attr_is_on
            self._update_icon()

    def _update_icon(self):
        """Update icon based on state."""
        self._attr_icon = "mdi:play-circle" if self._attr_is_on else "mdi:pause-circle"

    @property
    def is_on(self):
        """Return True if entity is on."""
        return self._attr_is_on

    async def async_turn_on(self, **kwargs):
        """Turn the entity on."""
        self._attr_is_on = True
        self._controller.enabled = True
        self._update_icon()
        self.async_write_ha_state()
        _LOGGER.info("async_turn_on()")

    async def async_turn_off(self, **kwargs):
        """Turn the entity off."""
        self._attr_is_on = False
        self._controller.enabled = False
        self._update_icon()
        self.async_write_ha_state()
        _LOGGER.info("async_turn_off()")
