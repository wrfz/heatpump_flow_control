"""Shared types."""

from collections.abc import Iterator
from dataclasses import asdict, dataclass, fields
from typing import Any


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

@dataclass
class LongtermFeatures:
    """Model longterm features."""
    temp_24h_ago: float = 0.0
    temp_24h_avg: float = 0.0
    temp_3d_avg: float = 0.0
    temp_7d_avg: float = 0.0
    temp_same_hour_avg: float = 0.0
    vorlauf_same_hour_avg: float = 0.0

@dataclass
class PowerFeatures:
    """Model power-related features."""
    power_avg_same_hour: float = 0.0
    power_avg_1h: float = 0.0
    power_avg_3h: float = 0.0
    power_favorable_hours: float = 0.0
@dataclass
class Features(LongtermFeatures, PowerFeatures):
    """Model features."""
    aussen_temp: float = 0.0
    raum_ist: float = 0.0
    raum_soll: float = 0.0
    vorlauf_ist: float = 0.0
    raum_abweichung: float = 0.0
    aussen_trend: float = 0.0
    aussen_trend_kurz: float = 0.0
    aussen_trend_mittel: float = 0.0
    stunde_sin: float = 0.0
    stunde_cos: float = 0.0
    wochentag_sin: float = 0.0
    wochentag_cos: float = 0.0
    # Interaktions-Features
    temp_diff: float = 0.0
    vorlauf_raum_diff: float = 0.0

    def set_long_term_features(self, long_term_features: LongtermFeatures):
        """Set long_term_features attributes to self."""
        for field in fields(long_term_features):
                    value = getattr(long_term_features, field.name)
                    setattr(self, field.name, value)

    def set_power_features(self, power_features: PowerFeatures):
        """Set power_features attributes to self."""
        for field in fields(power_features):
                    value = getattr(power_features, field.name)
                    setattr(self, field.name, value)

    def to_dict(self) -> dict[str, float]:
            """Gibt die Features als Dictionary zurück."""
            return asdict(self)


    def items(self) -> Iterator[tuple[str, float]]:
            """Gibt eine Iterierbare Liste von Schlüssel-Wert-Paaren zurück."""
            return iter(asdict(self).items())
