""""Common functions for Heatpump Flow Control integration."""

from homeassistant.helpers.device_registry import DeviceInfo

from .const import DOMAIN


def get_device_info(entry_id: str) -> DeviceInfo:
    """Create DeviceInfo for the integration."""

    return DeviceInfo(
        identifiers={(DOMAIN, entry_id)},
        name="Heatpump Flow Ctron",
        #manufacturer="Custom Integration",
        model="Flow Controller v1",
    )
