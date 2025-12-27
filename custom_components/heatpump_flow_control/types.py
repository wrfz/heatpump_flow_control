"""Shared types."""

from collections import deque
from collections.abc import Iterator
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timedelta
from itertools import islice
from typing import Any


@dataclass
class DateTimeTemperatur:
    """Datum-Uhrzeit und Temperatur Paar."""
    timestamp: datetime
    temperature: float

class HistoryBuffer(deque):
    """Eine Deque mit bequemem Zugriff auf das erste und letzte Element."""

    @property
    def first(self) -> DateTimeTemperatur:
        """Gibt das älteste Element zurück oder None, wenn leer."""
        return self[0]

    @property
    def last(self) -> DateTimeTemperatur:
        """Gibt das neueste Element zurück oder None, wenn leer."""
        return self[-1]

    @property
    def prev(self) -> DateTimeTemperatur:
        """Gibt das neueste Element zurück oder None, wenn leer."""
        return self[-2]

    @property
    def is_empty(self) -> bool:
        """Gibt True zurück, wenn der Buffer leer ist."""
        return not self

    def reversed_without_latest(self):
            """Iteriert rückwärts über alle Elemente außer dem neuesten."""
            if len(self) <= 1:
                return iter([])
            # Erstellt einen Iterator, der das letzte Element überspringt
            return reversed(list(islice(self, 0, len(self) - 1)))

    def get_trend(self, hours: float) -> float:
            """Berechnet die Temperaturänderung in Kelvin pro Stunde (K/h).

            Sucht den Punkt, der am nächsten am Zeitfenster liegt.
            Gibt None zurück, wenn nicht genügend Daten vorhanden sind.
            """
            if len(self) < 2:
                return 0.0

            latest = self.last
            # Wir suchen den Referenzpunkt X Stunden vor dem aktuellsten Eintrag
            cutoff = latest.timestamp - timedelta(hours=hours)

            # Suche rückwärts den ersten Eintrag, der älter oder gleich dem Cutoff ist
            reference_entry = self.first # Default auf den ältesten Wert
            for entry in reversed(self):
                if entry.timestamp <= cutoff:
                    reference_entry = entry
                    break

            # Zeitdifferenz berechnen
            time_diff_h = (latest.timestamp - reference_entry.timestamp) / timedelta(hours=1)

            if time_diff_h == 0:
                return 0.0

            temp_diff = latest.temperature - reference_entry.temperature
            return temp_diff / time_diff_h

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

    def get(self, key: str, default: Any = None) -> Any:
        """Simuliert das Verhalten eines Dictionaries."""
        return getattr(self, key, default)

@dataclass
class Trends:
    """Model longterm trends."""

    aussen_trend_1h: float = 0.0
    aussen_trend_2h: float = 0.0
    aussen_trend_3h: float = 0.0
    aussen_trend_6h: float = 0.0

@dataclass
class Features:
    """Model features."""

    aussen_temp: float = 0.0
    raum_ist: float = 0.0
    raum_soll: float = 0.0
    vorlauf_ist: float = 0.0
    raum_abweichung: float = 0.0
    aussen_trend_1h: float = 0.0
    aussen_trend_2h: float = 0.0
    aussen_trend_3h: float = 0.0
    aussen_trend_6h: float = 0.0
    stunde_sin: float = 0.0
    stunde_cos: float = 0.0
    wochentag_sin: float = 0.0
    wochentag_cos: float = 0.0
    # Interaktions-Features
    temp_diff: float = 0.0
    vorlauf_raum_diff: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Gibt die Features als Dictionary zurück."""
        return asdict(self)

    def items(self) -> Iterator[tuple[str, float]]:
        """Gibt eine Iterierbare Liste von Schlüssel-Wert-Paaren zurück."""
        return iter(asdict(self).items())

    def copy(self) -> "Features":
        """Erstellt eine exakte Kopie der aktuellen Features-Instanz."""
        return replace(self)

    def get(self, key: str, default: Any = None) -> Any:
        """Simuliert das Verhalten eines Dictionaries."""
        return getattr(self, key, default)

@dataclass
class Erfahrung:
    """Erfahrung für das Modell."""

    timestamp: datetime
    features: Features
    vorlauf_soll: float
    raum_ist_vorher: float
    raum_soll: float
    gelernt: bool = False
