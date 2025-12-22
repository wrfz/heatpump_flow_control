from homeassistant.components.switch import SwitchEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import (
    AddConfigEntryEntitiesCallback,
)
from homeassistant.helpers.restore_state import RestoreEntity
from .const import DOMAIN


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    async_add_entities([FlowControlActiveSwitch(entry)], True)


class FlowControlActiveSwitch(SwitchEntity, RestoreEntity):
    def __init__(self, entry) -> None:
        self._entry = entry
        self._attr_name = "Heatpump Flow Control Aktiv"
        self._attr_unique_id = f"{DOMAIN}_{entry.entry_id}_aktiv"
        self._attr_is_on = True
        self._attr_icon = "mdi:play-circle"

    async def async_added_to_hass(self) -> None:
        """Restore last state."""
        await super().async_added_to_hass()

        last_state = await self.async_get_last_state()
        if last_state is not None:
            self._attr_is_on = last_state.state == "on"
            self._update_icon()

    def _update_icon(self):
        """Update icon based on state."""
        self._attr_icon = "mdi:play-circle" if self._attr_is_on else "mdi:pause-circle"

    @property
    def is_on(self):
        return self._attr_is_on

    async def async_turn_on(self, **kwargs):
        self._attr_is_on = True
        self._update_icon()
        self.async_write_ha_state()

    async def async_turn_off(self, **kwargs):
        self._attr_is_on = False
        self._update_icon()
        self.async_write_ha_state()
