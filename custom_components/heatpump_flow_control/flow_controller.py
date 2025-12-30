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

# Pickle-Format Version für Migration
# Erhöhe diese Zahl bei inkompatiblen Code-Änderungen
PICKLE_VERSION = 2

# Gewichtung für synthetische Trainingsdaten
# Niedrigerer Wert = echte Daten haben stärkeren Einfluss
# Mit 0.001: Echte Daten dominieren (>80%) nach ~3 Tagen
# Berechnung: 3 Tage × 48 pred/Tag × 2.0 weight = 288 vs 70.200 × 0.001 = 70
SYNTHETIC_WEIGHT = 0.001


class FlowController:
    """Flow Controller für Vorlauf-Temperatur Regelung."""

    # Pickle-Version für Migrations-Check
    pickle_version: int = PICKLE_VERSION

    def __init__(
        self,
        min_vorlauf: float,
        max_vorlauf: float,
        learning_rate: float,
    ) -> None:
        """Initialize the flow controller."""

        self.min_vorlauf = min_vorlauf
        self.max_vorlauf = max_vorlauf
        self.learning_rate = learning_rate

        self.pickle_version = PICKLE_VERSION  # Für Migrations-Check

        # Feature-Liste für zeitversetztes Lernen (statt HA DB)
        self.erfahrungs_liste: list[Erfahrung] = []  # Speichert alle Predictions mit Features
        self.min_reward_hours = 4.0  # Lernen aus Features die 4h alt sind

        self._setup()

    def _setup(
        self,
    ) -> None:
        """Initialize the flow controller."""

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

        # Model mit synthetischen Daten trainieren
        _LOGGER.info("Trainiere Model mit synthetischen Daten")
        self._train_synthetic_data()

        _LOGGER.info(
            "_setup(): min=%s, max=%s, lr=%s",
            self.min_vorlauf,
            self.max_vorlauf,
            self.learning_rate,
        )

    def _heizkurve_fallback(self, aussen_temp: float, raum_abweichung: float) -> float:
        """Heizkurve für synthetisches Training.

        Typische Heizkurve: Vorlauf = A - B * Außentemperatur.
        """

        # Heizkurve: 10°C → 26.4°C bis -15°C → 35°C
        p0 = TempVorlauf(10.0, 26.4) # Aussentemperaur, Vorlauf
        p1 = TempVorlauf(-15.0, 35.0)

        vorlauf = p0.vorlauf + (p1.vorlauf - p0.vorlauf) / (p1.temp - p0.temp) * (aussen_temp - p0.temp)

        # Korrektur basierend auf Raum-Abweichung
        if abs(raum_abweichung) > 0.5:  # Raum zu warm/kalt
            vorlauf += raum_abweichung * 2.0

        v_min = min(p0.vorlauf, p1.vorlauf)
        v_max = max(p0.vorlauf, p1.vorlauf)

        return max(v_min, min(vorlauf, v_max))

    def _train_synthetic_data(self) -> None:
        """Trainiere Model mit synthetischen Daten aus der Heizkurve."""

        temp_min = -15.0
        temp_max = 10.0
        synthetic_count = 0

        _LOGGER.info("Starte synthetisches Training (Temperaturbereich %.1f bis %.1f°C)", temp_min, temp_max)

        # Viele Trainingsdurchläufe für präzises Lernen
        for _ in range(10):
            for t_aussen in range(int(temp_min), int(temp_max) + 1):
                # Breiter Temperaturbereich 16-28°C mit Schwerpunkt auf 22.5°C
                for raum_soll in [16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 21.5, 22.0, 22.5, 22.5, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0]:
                    # Breite Abweichungen für robustes Training (auch große Abweichungen wie im echten Betrieb)
                    for raum_abweichung in [-5.0, -3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0, 5.0]:
                        # Berechne Vorlauf-Soll aus Heizkurve
                        vorlauf_curve = self._heizkurve_fallback(
                            float(t_aussen), raum_abweichung
                        )

                        # Vorlauf_ist variieren: mal zu niedrig, mal genau richtig, mal zu hoch
                        for vorlauf_offset in [-2.0, 0.0, 2.0]:
                            vorlauf_ist_value = vorlauf_curve + vorlauf_offset

                            # Trainiere das Modell mit synthetischen Features
                            synthetic_features = Features(
                                aussen_temp=float(t_aussen),
                                raum_ist=raum_soll + raum_abweichung,
                                raum_soll=raum_soll,
                                vorlauf_ist=vorlauf_ist_value,
                                raum_abweichung=raum_abweichung,
                                aussen_trend_1h=0.0,
                                aussen_trend_2h=0.0,
                                aussen_trend_3h=0.0,
                                aussen_trend_6h=0.0,
                                stunde_sin=0.0,
                                stunde_cos=1.0,
                                wochentag_sin=0.0,
                                wochentag_cos=1.0,
                                temp_diff=float(t_aussen) - (raum_soll + raum_abweichung),
                                vorlauf_raum_diff=vorlauf_ist_value - (raum_soll + raum_abweichung),
                            )

                            self.model.learn_one(
                                synthetic_features.to_dict(),
                                vorlauf_curve,
                                sample_weight=SYNTHETIC_WEIGHT  # Schwächere Gewichtung für schnelle Anpassung
                            )
                            synthetic_count += 1

        _LOGGER.info(
            "Synthetisches Training abgeschlossen: %d Datenpunkte (weight=%.1f)",
            synthetic_count,
            SYNTHETIC_WEIGHT,
        )

    def _berechne_trends(self) -> dict[str, float]:
        """Berechnet zeitnormierte Temperatur-Trends in °C/Stunde.

        Returns:
            Dict mit Trend-Werten (°C/Stunde)
        """
        return {
            "aussen_trend_1h": self.aussen_temp_history.get_trend(hours=1.0),
            "aussen_trend_2h": self.aussen_temp_history.get_trend(hours=2.0),
            "aussen_trend_3h": self.aussen_temp_history.get_trend(hours=3.0),
            "aussen_trend_6h": self.aussen_temp_history.get_trend(hours=6.0),
        }

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

        # Berechne Trends
        #trends = self._berechne_trends()

        # Raum-Abweichung
        raum_abweichung = sensor_values.raum_soll - sensor_values.raum_ist

        # Tageszeit als zyklisches Feature (Default: Mittag)
        stunde = now.hour + now.minute / 60.0
        stunde_sin = math.sin(2 * math.pi * stunde / 24)
        stunde_cos = math.cos(2 * math.pi * stunde / 24)

        # Wochentag als zyklisches Feature (Default: Mittwoch)
        wochentag = now.weekday()
        wochentag_sin = math.sin(2 * math.pi * wochentag / 7)
        wochentag_cos = math.cos(2 * math.pi * wochentag / 7)

        return Features(
            aussen_temp = sensor_values.aussen_temp,
            raum_ist = sensor_values.raum_ist,
            raum_soll = sensor_values.raum_soll,
            vorlauf_ist = sensor_values.vorlauf_ist,
            raum_abweichung = raum_abweichung,
            aussen_trend_1h=0.0,  # Default: kein Trend
            aussen_trend_2h=0.0,
            aussen_trend_3h=0.0,
            aussen_trend_6h=0.0,
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

        # Echte Abweichung (kann positiv oder negativ sein)
        abweichung = erfahrung.raum_soll - raum_ist_jetzt  # Positiv = zu kalt, Negativ = zu warm
        abweichung_abs = abs(abweichung)

        if abweichung_abs < 0.3:
            # War gut! Temperatur fast perfekt erreicht
            korrigierter_vorlauf = erfahrung.vorlauf_soll
            weight = 1.0

        elif abweichung > 0:  # Zu kalt -> Vorlauf war zu niedrig
            korrigierter_vorlauf = erfahrung.vorlauf_soll + abweichung * 3.0
            weight = 3.0

        else:  # Zu warm -> Vorlauf war zu hoch
            korrigierter_vorlauf = erfahrung.vorlauf_soll + abweichung * 2.0  # abweichung ist negativ!
            weight = 2.0

        _LOGGER.info("_bewerte_erfahrung() korrigierter_vorlauf=%s, weight=%s", korrigierter_vorlauf, weight)

        return VorlaufSollWeight(vorlauf_soll=korrigierter_vorlauf, weight=weight)

    def lerne_aus_features(self, current_raum_ist: float) -> dict[str, int]:
        """Lernt aus Features die vor 4h gespeichert wurden.

        Args:
            current_raum_ist: Aktuelle Raumtemperatur (für Bewertung)

        Returns:
            Statistiken über das Lernen
        """
        stats = {"gelernt": 0, "uebersprungen": 0}

        now = datetime.now()

        # Finde Erfahrungen die alt genug sind (4h) und noch nicht gelernt wurden
        for erfahrung in self.erfahrungs_liste:
            if erfahrung.gelernt:
                continue

            # Prüfe ob Erfahrung alt genug ist (4h)
            age_hours = (now - erfahrung.timestamp).total_seconds() / 3600

            if age_hours < self.min_reward_hours:
                stats["uebersprungen"] += 1
                continue

            # Bewerte Erfahrung mit aktueller Raumtemperatur
            try:
                korrigierter_vorlauf_weight = self._bewerte_erfahrung(
                    erfahrung=erfahrung,
                    raum_ist_jetzt=current_raum_ist
                )

                # Lerne mit korrigiertem Vorlauf
                self.model.learn_one(
                    erfahrung.features.to_dict(),
                    korrigierter_vorlauf_weight.vorlauf_soll,
                    sample_weight=korrigierter_vorlauf_weight.weight,
                )

                erfahrung.gelernt = True
                stats["gelernt"] += 1

                _LOGGER.info(
                    "✓ Gelernt aus Erfahrung von vor %.1fh: Vorlauf %.1f°C → %.1f°C (weight=%.1f)",
                    age_hours,
                    erfahrung.vorlauf_soll,
                    korrigierter_vorlauf_weight.vorlauf_soll,
                    korrigierter_vorlauf_weight.weight,
                )

            except Exception as e:
                _LOGGER.error("Fehler beim Lernen aus Feature: %s", e)

        # Cleanup: Entferne gelernte und sehr alte Erfahrungen
        # - Gelernte Erfahrungen werden nicht mehr benötigt
        # - Sehr alte ungelernte Erfahrungen (älter als 7 Tage) ebenfalls entfernen
        max_age_days = 7
        cutoff_cleanup = now - timedelta(days=max_age_days)
        self.erfahrungs_liste = [
            e for e in self.erfahrungs_liste
            if not e.gelernt and e.timestamp > cutoff_cleanup
        ]

        if stats["gelernt"] > 0:
            _LOGGER.info(
                "Feature-Lernen abgeschlossen: %d gelernt, %d übersprungen, %d in Liste",
                stats["gelernt"],
                stats["uebersprungen"],
                len(self.erfahrungs_liste),
            )

        return stats

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

        features = self._erstelle_features(sensor_values)

        # Model für Prediction verwenden
        try:
            raw_prediction = self.model.predict_one(features.to_dict())
            _LOGGER.info(
                "Model-Prediction: %.2f°C (Außen: %.1f°C, Raum-Abw: %.2f°C, Vorlauf-Ist: %.1f°C)",
                raw_prediction,
                sensor_values.aussen_temp,
                features.raum_abweichung,
                sensor_values.vorlauf_ist,
            )
            vorlauf_soll = raw_prediction

            if vorlauf_soll > 1000 or vorlauf_soll < -1000:
                _LOGGER.warning(
                    "Model liefert unrealistischen Wert %.1f°C, clampe auf Grenzen",
                    vorlauf_soll,
                )

        except Exception as e:
            _LOGGER.error("Fehler bei Model-Prediction: %s, verwende Fallback-Heizkurve", e)
            vorlauf_soll = self._heizkurve_fallback(
                sensor_values.aussen_temp, features.raum_abweichung
            )

        # Begrenzung auf konfigurierten Bereich
        vorlauf_soll = max(self.min_vorlauf, min(self.max_vorlauf, vorlauf_soll))

        # Speichere Features für zeitversetztes Lernen (in 4h)
        erfahrung = Erfahrung(
            timestamp=datetime.now(),
            features=features,
            vorlauf_soll=vorlauf_soll,
            raum_ist_vorher=sensor_values.raum_ist,
            raum_soll=sensor_values.raum_soll,
            gelernt=False,
        )
        self.erfahrungs_liste.append(erfahrung)

        # Lerne aus Features die 4h alt sind (Reward-basiertes Lernen)
        self.lerne_aus_features(sensor_values.raum_ist)

        _LOGGER.info(
            "Vorlauf-Soll berechnet: %.1f°C (Außen: %.1f°C, Raum: %.1f/%.1f°C, Vorlauf-Ist: %.1f°C)",
            vorlauf_soll,
            sensor_values.aussen_temp,
            sensor_values.raum_ist,
            sensor_values.raum_soll,
            sensor_values.vorlauf_ist,
        )

        _LOGGER.info(
            "berechne_vorlauf_soll() min: %.1f°C, max: %.1f°C",
            self.min_vorlauf,
            self.max_vorlauf,
        )  # noqa: HASS_LOGGER_CAPITAL

        return vorlauf_soll, features

    def get_model_statistics(self) -> ModelStats:
        """Gibt Model-Statistiken zurück."""
        # Feature-basiertes Lernen
        gelernt = sum(1 for e in self.erfahrungs_liste if e.gelernt)
        wartend = sum(1 for e in self.erfahrungs_liste if not e.gelernt)

        return ModelStats(
            mae = self.metric.get(),
            history_size = len(self.aussen_temp_history),
            erfahrungen_total = len(self.erfahrungs_liste),
            erfahrungen_gelernt = gelernt,
            erfahrungen_wartend = wartend,
        )
