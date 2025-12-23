"""Shared types."""

from typing import Any

from homeassistant.helpers.event import dataclass


@dataclass
class SensorValues:
    """Input values for Vorlauf-Soll calculation."""
    aussen_temp: float
    raum_ist: float
    raum_soll: float
    vorlauf_ist: float

@dataclass
class ModelStats:
    """Model statsistics."""
    mae: float
    predictions_count: int
    use_fallback: bool
    history_size: int
    erfahrungen_total: int
    erfahrungen_gelernt: int
    erfahrungen_wartend: int
    reward_learning_enabled: bool

    def get(self, key: str, default: Any = None) -> Any:
        """Simuliert das Verhalten eines Dictionaries."""
        return getattr(self, key, default)
