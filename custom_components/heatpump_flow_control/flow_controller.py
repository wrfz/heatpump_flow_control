"""Machine Learning Controller für Wärmepumpen-Regelung."""

from datetime import datetime, timedelta
import logging
import math
from typing import Any

from river import compose, linear_model, metrics, optim, preprocessing

from .types import (
    DateTimeTemperatur,
    Erfahrung,
    Features,
    HistoryBuffer,
    ModelStats,
    SensorValues,
    TempVorlauf,
    VorlaufSollWeight,
)

# pylint: disable=hass-logger-capital
# ruff: noqa: BLE001

_LOGGER = logging.getLogger(__name__)


class FlowController:
    """Flow Controller für Vorlauf-Temperatur Regelung."""

    def __init__(
        self,
        min_vorlauf: float,
        max_vorlauf: float,
        learning_rate: float,
        trend_history_size: int,
    ) -> None:
        """Initialize the flow controller."""

        self.min_vorlauf = min_vorlauf
        self.max_vorlauf = max_vorlauf
        self.learning_rate = learning_rate
        self.trend_history_size = trend_history_size

        self.use_fallback = True
        self.predictions_count = 0

        self._setup()

    def _setup(
        self,
        *,
        force_fallback: bool = False,
    ) -> None:
        """Initialize the flow controller.

        Args:
            force_fallback: Wenn True, erzwinge Fallback-Modus (z.B. bei Model-Reset)
        """

        # River Online-Learning Model
        self.model = compose.Pipeline(
            preprocessing.StandardScaler(),
            linear_model.LinearRegression(
                optimizer=optim.Adam(self.learning_rate)
            ),
        )

        # ZEITBASIERTE Historie für Trend-Berechnung
        self.aussen_temp_history = HistoryBuffer()  # Kurze Historie für Trends (letzte 2-3 Stunden)
        self.short_history_hours = 3.0  # Kurze Historie: 3 Stunden

        # Langzeit-Historie (1 Woche)
        self.longterm_history_days = 7
        self.aussen_temp_longterm = HistoryBuffer()  # Speichert (timestamp, temp) Tupel
        self.vorlauf_longterm = HistoryBuffer()  # Speichert (timestamp, vorlauf) Tupel
        self.raum_temp_longterm = HistoryBuffer()  # Speichert (timestamp, raum_temp) Tupel

        # Metriken
        self.metric = metrics.MAE()
        self.min_predictions_for_model = 10

        if force_fallback:
            self.use_fallback = True
            _LOGGER.info("Fallback erzwungen (Model-Reset)")

        _LOGGER.info(
            "_setup(): min=%s, max=%s, lr=%s, history=%s",
            self.min_vorlauf,
            self.max_vorlauf,
            self.learning_rate,
            self.trend_history_size,
        )

    def _heizkurve_fallback(self, aussen_temp: float, raum_abweichung: float) -> float:
        """Fallback Heizkurve für Kaltstart.

        Typische Heizkurve: Vorlauf = A - B * Außentemperatur.
        """

        _LOGGER.info("_heizkurve_fallback()")

        max_vorlauf = self.min_vorlauf + (self.max_vorlauf - self.min_vorlauf) // 2

        p0 = TempVorlauf(15.0, self.min_vorlauf)
        p1 = TempVorlauf(-10.0, max_vorlauf)

        # 1. Die mathematische Gerade (funktioniert immer, solange p0.temp != p1.temp)
        vorlauf = p0.vorlauf + (p1.vorlauf - p0.vorlauf) / (p1.temp - p0.temp) * (aussen_temp - p0.temp)

        # Korrektur basierend auf Raum-Abweichung
        if raum_abweichung > 0.5 or raum_abweichung < -0.5:  # Raum zu kalt
            vorlauf += raum_abweichung * 2.0

        v_min = min(p0.vorlauf, p1.vorlauf)
        v_max = max(p0.vorlauf, p1.vorlauf)

        vorlauf = max(v_min, min(vorlauf, v_max))

        _LOGGER.info("_heizkurve_fallback() => %.1f", vorlauf)

        return vorlauf

    def _erstelle_features(
        self,
        sensor_values: SensorValues,
    ) -> Features:
        """Erstellt Feature-Dictionary für das Model.

        Args:
            aussen_temp: Außentemperatur
            raum_ist: Ist-Raumtemperatur
            raum_soll: Soll-Raumtemperatur
            vorlauf_ist: Ist-Vorlauf
        """

        _LOGGER.info("_erstelle_features()")

        now = datetime.now()

        self.aussen_temp_history.append(DateTimeTemperatur(timestamp=now, temperature=sensor_values.aussen_temp))

        # Entferne Einträge die älter als short_history_hours sind
        cutoff_time = now - timedelta(hours=self.short_history_hours)

        while self.aussen_temp_history and self.aussen_temp_history.first.timestamp < cutoff_time:
            self.aussen_temp_history.popleft()

        # Langzeit-Historie aktualisieren (nur alle 30 Min)
        # Entferne alte Einträge basierend auf Tagen statt fester Anzahl
        cutoff_longterm = now - timedelta(days=self.longterm_history_days)

        if (
            not self.aussen_temp_longterm
            or (now - self.aussen_temp_longterm[-1].timestamp).total_seconds() > 1800
        ):
            self.aussen_temp_longterm.append(DateTimeTemperatur(timestamp=now, temperature=sensor_values.aussen_temp))
            self.vorlauf_longterm.append(DateTimeTemperatur(timestamp=now, temperature=sensor_values.vorlauf_ist))
            self.raum_temp_longterm.append(DateTimeTemperatur(timestamp=now, temperature=sensor_values.raum_ist))

            histories = [
                self.aussen_temp_longterm,
                self.vorlauf_longterm,
                self.raum_temp_longterm
            ]

            # Entferne alte Einträge basierend auf Zeit
            for history in histories:
                while history and history[0].timestamp < cutoff_longterm:
                    history.popleft()

        # Raum-Abweichung
        raum_abweichung = sensor_values.raum_soll - sensor_values.raum_ist

        # Tageszeit als zyklisches Feature
        stunde = now.hour + now.minute / 60.0
        stunde_sin = math.sin(2 * math.pi * stunde / 24)
        stunde_cos = math.cos(2 * math.pi * stunde / 24)

        # Wochentag als zyklisches Feature
        wochentag = now.weekday()
        wochentag_sin = math.sin(2 * math.pi * wochentag / 7)
        wochentag_cos = math.cos(2 * math.pi * wochentag / 7)

        return Features(
            aussen_temp = sensor_values.aussen_temp,
            raum_ist = sensor_values.raum_ist,
            raum_soll = sensor_values.raum_soll,
            vorlauf_ist = sensor_values.vorlauf_ist,
            raum_abweichung = raum_abweichung,
            stunde_sin = stunde_sin,
            stunde_cos = stunde_cos,
            wochentag_sin = wochentag_sin,
            wochentag_cos = wochentag_cos,
            # Interaktions-Features
            temp_diff = sensor_values.aussen_temp - sensor_values.raum_ist,
            vorlauf_raum_diff = sensor_values.vorlauf_ist - sensor_values.raum_ist,
        )

    def _bewerte_erfahrung(
        self,
        erfahrung: Erfahrung,
        raum_ist_jetzt: float,
    ) -> VorlaufSollWeight:
        """Berechnet korrigierten Vorlauf."""

        abweichung = abs(raum_ist_jetzt - erfahrung.raum_soll)

        if abweichung < 0.3:
            # War gut!
            korrigierter_vorlauf = erfahrung.vorlauf_soll
            weight = 1.0

        # War schlecht → Korrigiere
        elif abweichung < 0:  # Zu kalt
            korrigierter_vorlauf = erfahrung.vorlauf_soll + abweichung * 3.0
            weight = 3.0

        # Zu warm
        else:
            korrigierter_vorlauf = erfahrung.vorlauf_soll - abweichung * 2.0
            weight = 2.0

        _LOGGER.info("_bewerte_erfahrung() korrigierter_vorlauf=%s, weight=%s", korrigierter_vorlauf, weight)

        return VorlaufSollWeight(vorlauf_soll=korrigierter_vorlauf, weight=weight)

    def lerne_aus_history(
        self,
        historical_state: dict[str, Any],
        current_raum_ist: float,
    ) -> bool:
        """Lernt aus einem historischen HA-Sensor-Zustand."""

        try:
            # Extrahiere Sensor-Werte vom damaligen Zeitpunkt
            timestamp = historical_state['timestamp']
            aussen_temp = historical_state['aussen_temp']
            raum_ist = historical_state['raum_ist']
            raum_soll = historical_state['raum_soll']
            vorlauf_ist = historical_state['vorlauf_ist']
            vorlauf_soll = historical_state['vorlauf_soll']

            # Rekonstruiere Features vom damaligen Zeitpunkt
            raum_abweichung = raum_soll - raum_ist
            stunde = timestamp.hour + timestamp.minute / 60.0
            stunde_sin = math.sin(2 * math.pi * stunde / 24)
            stunde_cos = math.cos(2 * math.pi * stunde / 24)
            wochentag = timestamp.weekday()
            wochentag_sin = math.sin(2 * math.pi * wochentag / 7)
            wochentag_cos = math.cos(2 * math.pi * wochentag / 7)

            features = Features(
                aussen_temp=aussen_temp,
                raum_ist=raum_ist,
                raum_soll=raum_soll,
                vorlauf_ist=vorlauf_ist,
                raum_abweichung=raum_abweichung,
                # Trends können wir nicht rekonstruieren - setze auf 0
                aussen_trend_1h=0.0,
                aussen_trend_2h=0.0,
                aussen_trend_3h=0.0,
                aussen_trend_6h=0.0,
                stunde_sin=stunde_sin,
                stunde_cos=stunde_cos,
                wochentag_sin=wochentag_sin,
                wochentag_cos=wochentag_cos,
                # Interaktions-Features | werden durch Gewichtungen im Model berücksichtigt
                temp_diff=aussen_temp - raum_ist,   # Model muss lernen: "5°C draußen UND 22°C drin → Differenz = -17°C"
                vorlauf_raum_diff=vorlauf_ist - raum_ist,   # Wie viel wärmer ist Vorlauf?
            )

            # Erstelle Erfahrung für Bewertung
            erfahrung = Erfahrung(
                timestamp=timestamp,
                features=features,
                vorlauf_soll=vorlauf_soll,
                raum_ist_vorher=raum_ist,
                raum_soll=raum_soll,
                gelernt=False,
            )

            # Bewerte mit Raum-Ist von 2-6h SPÄTER
            korrigierter_vorlauf_weight = self._bewerte_erfahrung(
                erfahrung=erfahrung, raum_ist_jetzt=current_raum_ist
            )

            _LOGGER.info("lerne_aus_history() features=%s", features)

            self.model.learn_one(
                features.to_dict(),
                korrigierter_vorlauf_weight.vorlauf_soll,
                sample_weight=korrigierter_vorlauf_weight.weight,
            )

        except Exception as e:
            _LOGGER.error("Fehler beim History-Learning: %s", e)
            return False

        else:
            _LOGGER.info("Lernen aus History erfolgreich")
            return True

    def berechne_vorlauf_soll(
        self,
        sensor_values: SensorValues,
    ) -> tuple[float, Features]:
        """Berechnet optimalen Vorlauf-Sollwert.

        Args:
            aussen_temp: Außentemperatur
            raum_ist: Ist-Raumtemperatur
            raum_soll: Soll-Raumtemperatur
            vorlauf_ist: Ist-Vorlauf

        Returns:
            tuple: (vorlauf_soll, features)
        """

        _LOGGER.info("berechne_vorlauf_soll()")

        # Flag um zu tracken ob Model in dieser Berechnung zurückgesetzt wurde
        model_was_reset = False

        features = self._erstelle_features(sensor_values)

        # Während Kaltstart: Heizkurve verwenden
        if (
            self.use_fallback
            and self.predictions_count < self.min_predictions_for_model
        ):
            vorlauf_soll = self._heizkurve_fallback(
                sensor_values.aussen_temp, features.raum_abweichung
            )
            _LOGGER.debug(
                "Verwende Heizkurve (Kaltstart %d/%d): %.1f°C",
                self.predictions_count,
                self.min_predictions_for_model,
                vorlauf_soll,
            )

            if self.predictions_count >= 10:
                _LOGGER.info("Wechsel von Heizkurve zu Model-Modus")
                self.use_fallback = False
        else:
            # Model verwenden
            try:
                vorlauf_soll = self.model.predict_one(features.to_dict())

                # Sanity check: Falls Model unrealistische Werte liefert
                if vorlauf_soll < 15 or vorlauf_soll > 70:
                    _LOGGER.warning(
                        "Model liefert unrealistischen Wert %.1f°C (Features: %s), verwende Heizkurve",
                        vorlauf_soll,
                        {k: round(v, 3) for k, v in features.items()},
                    )

                    # Bei extrem unrealistischen Werten: Model ist korrupt, zurücksetzen
                    if abs(vorlauf_soll) > 1000:
                        _LOGGER.error(
                            "Model-Output extrem unrealistisch (%.0f°C), setze Model zurück",
                            vorlauf_soll,
                        )
                        # Model komplett neu initialisieren
                        self._setup(
                            force_fallback=True,  # Erzwinge Fallback nach Reset
                        )
                        self.predictions_count = 0
                        # Diese Vorhersage war ungültig, zählt nicht mit
                        model_was_reset = True

                    vorlauf_soll = self._heizkurve_fallback(
                        sensor_values.aussen_temp, features.raum_abweichung
                    )
                elif self.use_fallback:
                    _LOGGER.info("Wechsel von Heizkurve zu ML-Model")
                    self.use_fallback = False

            except Exception as e:
                _LOGGER.error("Fehler bei Model-Prediction: %s", e)
                vorlauf_soll = self._heizkurve_fallback(
                    sensor_values.aussen_temp, features.raum_abweichung
                )

        # Begrenzung auf konfigurierten Bereich
        vorlauf_soll = max(self.min_vorlauf, min(self.max_vorlauf, vorlauf_soll))

        # Nur inkrementieren wenn Model nicht gerade zurückgesetzt wurde
        if not model_was_reset:
            self.predictions_count += 1


        if self.predictions_count >= self.min_predictions_for_model:
            self.use_fallback = False

        # History-basiertes Lernen wird von number.py über lerne_aus_history() aufgerufen
        # (Nicht mehr hier, da wir Zugriff auf HA-History brauchen)

        _LOGGER.info(
            "Vorlauf-Soll berechnet: %.1f°C (Außen: %.1f°C, Raum: %.1f/%.1f°C, Trend: [1h: %.2f, 2h: %.2f, 3h: %.2f, 6h: %.2f])",
            vorlauf_soll,
            sensor_values.aussen_temp,
            sensor_values.raum_ist,
            sensor_values.raum_soll,
            features.aussen_trend_1h,
            features.aussen_trend_2h,
            features.aussen_trend_3h,
            features.aussen_trend_6h,
        )

        _LOGGER.info(
            "berechne_vorlauf_soll() min: %.1f°C, max: %.1f°C",
            self.min_vorlauf,
            self.max_vorlauf,
        )  # noqa: HASS_LOGGER_CAPITAL

        return vorlauf_soll, features

    def get_model_statistics(self) -> ModelStats:
        """Gibt Model-Statistiken zurück."""
        # History-basiertes Lernen: Stats kommen direkt aus HA-History
        return ModelStats(
            mae = self.metric.get() if self.predictions_count > 0 else 0.0,
            predictions_count = self.predictions_count,
            use_fallback = self.use_fallback,
            history_size = len(self.aussen_temp_history),
            erfahrungen_total = 0,  # Nicht mehr relevant
            erfahrungen_gelernt = 0,  # Wird in number.py gezählt
            erfahrungen_wartend = 0,  # Nicht mehr relevant
        )

    def reset_model(self) -> None:
        """Setzt das Model zurück (für Neustart)."""
        _LOGGER.info("-> reset_model()")

        self._setup(
            force_fallback=True,  # Erzwinge Fallback nach manuellem Reset
        )
