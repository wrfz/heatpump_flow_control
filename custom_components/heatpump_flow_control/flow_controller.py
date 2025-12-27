"""Machine Learning Controller für Wärmepumpen-Regelung."""

from datetime import datetime, timedelta
import logging
import math
import statistics
from typing import Any

from river import compose, linear_model, metrics, optim, preprocessing

from .types import (
    DateTimeTemperatur,
    Erfahrung,
    Features,
    HistoryBuffer,
    ModelStats,
    SensorValues,
    Trends,
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

        self.predictions_count = 0

        # History-basiertes Lernen: Kein ErfahrungsSpeicher mehr nötig
        self.min_reward_hours = 2.0  # Mindeststunden bis Bewertung
        self.max_reward_hours = 6.0  # Maximalstunden bis Bewertung
        self.letztes_lern_timestamp = datetime.now() - timedelta(hours=24)  # Vermeide Doppel-Lernen

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

        # Speichere alten use_fallback Status (für Restart-Persistenz)
        old_use_fallback = getattr(self, "use_fallback", None)

        # River Online-Learning Model
        self.model = compose.Pipeline(
            preprocessing.StandardScaler(),
            linear_model.LinearRegression(optimizer=optim.SGD(self.learning_rate)),
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
        self.predictions_count = 0

        # Kaltstart-Heizkurve (fallback)
        # Intelligente Fallback-Steuerung:
        # 1. force_fallback=True → Erzwinge Fallback (z.B. nach Model-Reset)
        # 2. Erster Init (old_use_fallback=None) → Starte mit Fallback
        # 3. Restart (old_use_fallback!=None) → Behalte alten Status
        if force_fallback:
            self.use_fallback = True
            _LOGGER.info("Fallback erzwungen (Model-Reset)")
        elif old_use_fallback is None:
            self.use_fallback = True  # Erster Start
            _LOGGER.info("Fallback aktiviert (Kaltstart)")
        else:
            self.use_fallback = old_use_fallback  # Restart: behalte Status
            _LOGGER.info("Fallback-Status beibehalten: %s", self.use_fallback)

        self.min_predictions_for_model = 10

        self._ensure_attributes()

        _LOGGER.info(
            "_setup(): min=%s, max=%s, lr=%s, history=%s",
            self.min_vorlauf,
            self.max_vorlauf,
            self.learning_rate,
            self.trend_history_size,
        )

    def _ensure_attributes(self) -> None:
        """Stellt sicher, dass alle Attribute existieren (Backwards Compatibility).

        Wird bei Bedarf aufgerufen wenn Objekt aus Storage geladen wurde.
        """
        if not hasattr(self, "min_reward_hours"):
            self.min_reward_hours = 2.0
        if not hasattr(self, "max_reward_hours"):
            self.max_reward_hours = 6.0
        if not hasattr(self, "letztes_lern_timestamp"):
            self.letztes_lern_timestamp = datetime.now() - timedelta(hours=24)

    def _heizkurve_fallback(self, aussen_temp: float, raum_abweichung: float) -> float:
        """Fallback Heizkurve für Kaltstart.

        Typische Heizkurve: Vorlauf = A - B * Außentemperatur.
        """

        _LOGGER.info("_heizkurve_fallback()")

        # Basis-Heizkurve (anpassbar)
        if aussen_temp <= -10:
            vorlauf = 38.0
        elif aussen_temp >= 15:
            vorlauf = 28.0
        else:
            # Lineare Interpolation
            vorlauf = 38.0 - (aussen_temp + 10) * (13.0 / 28.0)

        # Korrektur basierend auf Raum-Abweichung
        if raum_abweichung > 0.5:  # Raum zu kalt
            vorlauf += raum_abweichung * 3.0
        elif raum_abweichung < -0.5:  # Raum zu warm
            vorlauf += raum_abweichung * 5.0

        _LOGGER.info("_heizkurve_fallback() => %.1f", vorlauf)

        return vorlauf

    def _berechne_trends(self) -> Trends:
        """Berechnet zeitnormierte Temperatur-Trends in °C/Stunde.

        Returns:
            Trends mit °C/Stunde normalisierten Werten
        """

        _LOGGER.info("_berechne_trends()")

        return Trends(
            aussen_trend_1h=self.aussen_temp_history.get_trend(hours=1.0),
            aussen_trend_2h=self.aussen_temp_history.get_trend(hours=2.0),
            aussen_trend_3h=self.aussen_temp_history.get_trend(hours=3.0),
            aussen_trend_6h=self.aussen_temp_history.get_trend(hours=6.0),
        )

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

        trends = self._berechne_trends()

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
            aussen_trend_1h=trends.aussen_trend_1h,
            aussen_trend_2h=trends.aussen_trend_2h,
            aussen_trend_3h=trends.aussen_trend_3h,
            aussen_trend_6h=trends.aussen_trend_6h,
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
        raum_ist: float,
        aussen_trend: float = 0.0,
    ) -> tuple[float, float]:
        """Bewertet eine Erfahrung basierend auf dem Ergebnis.

        Args:
            erfahrung: Die zu bewertende Erfahrung
            raum_ist_jetzt: Aktuelle Raumtemperatur (2-3h nach Entscheidung)
            aussen_trend: Aktueller Außentemperatur-Trend

        Returns:
            tuple: (reward, korrigierter_vorlauf)
                - reward: -1.0 bis +1.0 (wie gut war die Entscheidung)
                - korrigierter_vorlauf: Verbesserter Sollwert falls reward < 0
        """

        _LOGGER.info("_bewerte_erfahrung()")

        raum_soll = erfahrung.raum_soll
        vorlauf_soll = erfahrung.vorlauf_soll

        # Wie groß ist die Abweichung jetzt (2-3h später)?
        abweichung = raum_ist - raum_soll
        abs_abweichung = abs(abweichung)

        # Bewertung berechnen
        if abs_abweichung < 0.3:
            reward = 1.0  # Perfekt!
        elif abs_abweichung < 0.5:
            reward = 0.5  # Gut
        elif abs_abweichung < 0.8:
            reward = 0.0  # OK
        elif abs_abweichung < 1.2:
            reward = -0.5  # Nicht gut
        else:
            reward = -1.0  # Schlecht

        # Trend-Anpassung: Bei starken Trends weniger streng bewerten
        # Wenn Außentemp gefallen ist und Raum zu kalt → War teilweise zu erwarten
        if aussen_trend < -0.5 and abweichung < -0.5:
            reward = max(reward, -0.3)  # Nicht zu negativ bewerten
            _LOGGER.debug(
                "Trend-Anpassung: Außentemp fiel (%.2f), Raum zu kalt war teilweise erwartbar",
                aussen_trend,
            )

        # Wenn Außentemp gestiegen ist und Raum zu warm → War teilweise zu erwarten
        if aussen_trend > 0.5 and abweichung > 0.5:
            reward = max(reward, -0.3)  # Nicht zu negativ bewerten
            _LOGGER.debug(
                "Trend-Anpassung: Außentemp stieg (%.2f), Raum zu warm war teilweise erwartbar",
                aussen_trend,
            )

        # Korrektur berechnen (nur bei schlechtem Ergebnis)
        if reward < 0:
            if abweichung < 0:  # Zu kalt → Vorlauf war zu niedrig
                # Berechne Korrektur: pro 0.5°C Abweichung +2°C Vorlauf
                korrektur = abs(abweichung) * 4.0
                korrigierter_vorlauf_soll = vorlauf_soll + korrektur
            else:  # Zu warm → Vorlauf war zu hoch
                # Berechne Korrektur: pro 0.5°C Abweichung -1.5°C Vorlauf
                # (Weniger aggressiv, da Überhitzung träger reagiert)
                korrektur = abs(abweichung) * 3.0
                korrigierter_vorlauf_soll = vorlauf_soll - korrektur
        else:
            korrigierter_vorlauf_soll = vorlauf_soll

        return reward, korrigierter_vorlauf_soll

    def lerne_aus_history(
        self,
        historical_states: list[dict[str, Any]],
        current_raum_ist: float,
        current_time: datetime,
    ) -> dict[str, int]:
        """Lernt aus historischen HA-Sensor-Werten.

        Args:
            historical_states: Liste von historischen Sensor-Zuständen (2-6h alt)
                Format: [{
                    'timestamp': datetime,
                    'aussen_temp': float,
                    'raum_ist': float,
                    'raum_soll': float,
                    'vorlauf_ist': float,
                    'vorlauf_soll': float,  # Damals berechneter Soll-Wert
                }, ...]
            current_raum_ist: Aktuelle Raumtemperatur (für Reward-Bewertung)
            current_time: Aktuelle Zeit (timezone-aware)

        Returns:
            Statistiken über das Lernen (gelernt_positiv, gelernt_negativ)
        """

        _LOGGER.info("_lerne_aus_history() mit %d historischen Zuständen", len(historical_states))

        stats = {"gelernt_positiv": 0, "gelernt_negativ": 0}

        if not historical_states:
            return stats

        # Aktuellen Trend für Bewertung holen
        trends = self._berechne_trends()
        aussen_trend = trends.aussen_trend_1h

        for state in historical_states:
            try:
                # Extrahiere Sensor-Werte
                timestamp = state['timestamp']
                aussen_temp = state['aussen_temp']
                raum_ist_damals = state['raum_ist']
                raum_soll = state['raum_soll']
                vorlauf_ist = state['vorlauf_ist']
                vorlauf_soll_damals = state['vorlauf_soll']

                # Rekonstruiere Features vom damaligen Zeitpunkt
                # (vereinfachte Version ohne komplette Historie)
                raum_abweichung = raum_soll - raum_ist_damals
                stunde = timestamp.hour + timestamp.minute / 60.0
                stunde_sin = math.sin(2 * math.pi * stunde / 24)
                stunde_cos = math.cos(2 * math.pi * stunde / 24)
                wochentag = timestamp.weekday()
                wochentag_sin = math.sin(2 * math.pi * wochentag / 7)
                wochentag_cos = math.cos(2 * math.pi * wochentag / 7)

                features = Features(
                    aussen_temp=aussen_temp,
                    raum_ist=raum_ist_damals,
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
                    temp_diff=aussen_temp - raum_ist_damals,
                    vorlauf_raum_diff=vorlauf_ist - raum_ist_damals,
                )

                # Erstelle Erfahrung für Bewertung
                erfahrung = Erfahrung(
                    timestamp=timestamp,
                    features=features,
                    vorlauf_soll=vorlauf_soll_damals,
                    raum_ist_vorher=raum_ist_damals,
                    raum_soll=raum_soll,
                    gelernt=False,
                )

                # Bewerte die Erfahrung
                reward, korrigierter_vorlauf = self._bewerte_erfahrung(
                    erfahrung, current_raum_ist, aussen_trend
                )

                if reward >= 0:
                    # Gute Erfahrung: Lerne mit original Vorlauf
                    sample_weight = 1.0 + reward  # 1.0 bis 2.0
                    self.model.learn_one(
                        features.to_dict(),
                        vorlauf_soll_damals,
                        sample_weight=sample_weight,
                    )
                    stats["gelernt_positiv"] += 1

                    _LOGGER.info(
                        "✓ History-Lernen (Reward=%.1f, Weight=%.1f): Vorlauf %.1f°C war gut (von %s)",
                        reward,
                        sample_weight,
                        vorlauf_soll_damals,
                        timestamp.strftime("%H:%M"),
                    )
                else:
                    # Schlechte Erfahrung: Lerne mit korrigiertem Vorlauf
                    sample_weight = abs(reward) * 1.5  # 0.75 bis 1.5
                    self.model.learn_one(
                        features.to_dict(), korrigierter_vorlauf, sample_weight=sample_weight
                    )
                    stats["gelernt_negativ"] += 1

                    _LOGGER.warning(
                        "✗ History-Lernen (Reward=%.1f, Weight=%.1f): Vorlauf %.1f°C → besser %.1f°C (von %s)",
                        reward,
                        sample_weight,
                        vorlauf_soll_damals,
                        korrigierter_vorlauf,
                        timestamp.strftime("%H:%M"),
                    )

            except Exception as e:
                _LOGGER.error("Fehler beim History-Learning für State %s: %s", state.get('timestamp'), e)

        if stats["gelernt_positiv"] + stats["gelernt_negativ"] > 0:
            _LOGGER.info(
                "History-Lernen abgeschlossen: %d positiv, %d negativ (von %d States)",
                stats["gelernt_positiv"],
                stats["gelernt_negativ"],
                len(historical_states),
            )
            self.letztes_lern_timestamp = current_time

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

        # Backwards compatibility: Stelle sicher, dass alle Attribute existieren
        self._ensure_attributes()

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
