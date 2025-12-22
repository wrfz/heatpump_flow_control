"""Machine Learning Controller für Wärmepumpen-Regelung."""

from datetime import datetime
import logging
import math

from river import compose, linear_model, metrics, optim, preprocessing

from .const import (
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_VORLAUF,
    DEFAULT_MIN_VORLAUF,
    DEFAULT_TREND_HISTORY_SIZE,
)

# pylint: disable=hass-logger-capital
# ruff: noqa: BLE001

_LOGGER = logging.getLogger(__name__)


class FlowController:
    """ML Controller für Vorlauf-Temperatur Regelung."""

    def __init__(
        self,
        min_vorlauf: float = DEFAULT_MIN_VORLAUF,
        max_vorlauf: float = DEFAULT_MAX_VORLAUF,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        trend_history_size: int = DEFAULT_TREND_HISTORY_SIZE,
    ) -> None:
        """Initialize the ML controller."""

        self._setup(
            min_vorlauf=min_vorlauf,
            max_vorlauf=max_vorlauf,
            learning_rate=learning_rate,
            trend_history_size=trend_history_size,
        )

    def _setup(
        self,
        *,
        min_vorlauf: float = DEFAULT_MIN_VORLAUF,
        max_vorlauf: float = DEFAULT_MAX_VORLAUF,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        trend_history_size: int = DEFAULT_TREND_HISTORY_SIZE,
    ) -> None:
        """Initialize the ML controller."""

        self.min_vorlauf = min_vorlauf
        self.max_vorlauf = max_vorlauf
        self.trend_history_size = trend_history_size

        # River Online-Learning Model
        self.model = compose.Pipeline(
            preprocessing.StandardScaler(),
            linear_model.LinearRegression(optimizer=optim.SGD(learning_rate)),
        )

        # Historie für Trend-Berechnung (kurz)
        self.aussen_temp_history = []
        self.timestamps = []

        # ERWEITERT: Langzeit-Historie (24 Stunden bei stündlichen Updates = 24 Werte)
        self.longterm_history_size = 24 * 7  # 1 Woche
        self.aussen_temp_longterm = []  # Speichert (timestamp, temp) Tupel
        self.vorlauf_longterm = []  # Speichert (timestamp, vorlauf) Tupel
        self.raum_temp_longterm = []  # Speichert (timestamp, raum_temp) Tupel

        # Metriken
        self.metric = metrics.MAE()
        self.predictions_count = 0

        # Kaltstart-Heizkurve (fallback)
        self.use_fallback = True
        self.min_predictions_for_model = 10

        _LOGGER.info(
            "ML Controller initialisiert: min=%s, max=%s, lr=%s, history=%s, longterm=%s",
            min_vorlauf,
            max_vorlauf,
            learning_rate,
            trend_history_size,
            self.longterm_history_size,
        )

    def _heizkurve_fallback(self, aussen_temp: float, raum_abweichung: float) -> float:
        """Fallback Heizkurve für Kaltstart.

        Typische Heizkurve: Vorlauf = A - B * Außentemperatur.
        """

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

    def _berechne_trends(self) -> dict[str, float]:
        """Berechnet Temperatur-Trends aus Historie."""
        if len(self.aussen_temp_history) < 2:
            return {
                "aussen_trend": 0.0,
                "aussen_trend_kurz": 0.0,
                "aussen_trend_mittel": 0.0,
            }

        # Kurzfristiger Trend (letzte 2 Messungen)
        trend_kurz = self.aussen_temp_history[-1] - self.aussen_temp_history[-2]

        # Mittelfristiger Trend (letzte N/2 Messungen)
        mittel_idx = max(1, len(self.aussen_temp_history) // 2)
        if len(self.aussen_temp_history) >= mittel_idx + 1:
            trend_mittel = (
                self.aussen_temp_history[-1] - self.aussen_temp_history[-mittel_idx]
            ) / mittel_idx
        else:
            trend_mittel = 0.0

        # Langfristiger Trend (über gesamte Historie)
        trend_lang = (self.aussen_temp_history[-1] - self.aussen_temp_history[0]) / len(
            self.aussen_temp_history
        )

        return {
            "aussen_trend": trend_lang,
            "aussen_trend_kurz": trend_kurz,
            "aussen_trend_mittel": trend_mittel,
        }

    def _erstelle_features(
        self,
        aussen_temp: float,
        raum_ist: float,
        raum_soll: float,
        vorlauf_ist: float,
    ) -> dict[str, float]:
        """Erstellt Feature-Dictionary für das Model."""

        now = datetime.now()

        # Temperatur-Historie aktualisieren (kurzfristig)
        self.aussen_temp_history.append(aussen_temp)
        self.timestamps.append(now)

        if len(self.aussen_temp_history) > self.trend_history_size:
            self.aussen_temp_history.pop(0)
            self.timestamps.pop(0)

        # ERWEITERT: Langzeit-Historie aktualisieren (stündlich)
        # Nur hinzufügen wenn > 30 Min seit letztem Eintrag
        if (
            not self.aussen_temp_longterm
            or (now - self.aussen_temp_longterm[-1][0]).total_seconds() > 1800
        ):
            self.aussen_temp_longterm.append((now, aussen_temp))
            self.vorlauf_longterm.append((now, vorlauf_ist))
            self.raum_temp_longterm.append((now, raum_ist))

            # Begrenze auf longterm_history_size
            if len(self.aussen_temp_longterm) > self.longterm_history_size:
                self.aussen_temp_longterm.pop(0)
                self.vorlauf_longterm.pop(0)
                self.raum_temp_longterm.pop(0)

        # Trends berechnen
        trends = self._berechne_trends()

        # Raum-Abweichung
        raum_abweichung = raum_soll - raum_ist

        # Tageszeit als zyklisches Feature
        stunde = now.hour + now.minute / 60.0
        stunde_sin = math.sin(2 * math.pi * stunde / 24)
        stunde_cos = math.cos(2 * math.pi * stunde / 24)

        # Wochentag als zyklisches Feature
        wochentag = now.weekday()
        wochentag_sin = math.sin(2 * math.pi * wochentag / 7)
        wochentag_cos = math.cos(2 * math.pi * wochentag / 7)

        # ERWEITERT: Langzeit-Features berechnen
        longterm_features = self._berechne_longterm_features(now, stunde)

        features = {
            "aussen_temp": aussen_temp,
            "raum_ist": raum_ist,
            "raum_soll": raum_soll,
            "vorlauf_ist": vorlauf_ist,
            "raum_abweichung": raum_abweichung,
            "aussen_trend": trends["aussen_trend"],
            "aussen_trend_kurz": trends["aussen_trend_kurz"],
            "aussen_trend_mittel": trends["aussen_trend_mittel"],
            "stunde_sin": stunde_sin,
            "stunde_cos": stunde_cos,
            "wochentag_sin": wochentag_sin,
            "wochentag_cos": wochentag_cos,
            # Interaktions-Features
            "temp_diff": aussen_temp - raum_ist,
            "vorlauf_raum_diff": vorlauf_ist - raum_ist,
        }

        # Langzeit-Features hinzufügen
        features.update(longterm_features)

        return features

    def _berechne_longterm_features(
        self, now: datetime, current_hour: float
    ) -> dict[str, float]:
        """Berechnet Langzeit-Features aus der Historie.

        Returns:
            Dictionary mit Langzeit-Features
        """
        features = {
            "temp_24h_ago": 0.0,
            "temp_24h_avg": 0.0,
            "temp_7d_avg": 0.0,
            "temp_same_hour_avg": 0.0,
            "vorlauf_same_hour_avg": 0.0,
        }

        if len(self.aussen_temp_longterm) < 2:
            return features

        # Durchschnittstemperatur letzte 24h
        temps_24h = [
            temp
            for ts, temp in self.aussen_temp_longterm
            if (now - ts).total_seconds() < 86400
        ]
        if temps_24h:
            features["temp_24h_avg"] = sum(temps_24h) / len(temps_24h)

        # Durchschnittstemperatur letzte 7 Tage
        if self.aussen_temp_longterm:
            all_temps = [temp for _, temp in self.aussen_temp_longterm]
            features["temp_7d_avg"] = sum(all_temps) / len(all_temps)

        # Temperatur vor 24 Stunden (± 2 Stunden Toleranz)
        for ts, temp in reversed(self.aussen_temp_longterm):
            hours_ago = (now - ts).total_seconds() / 3600
            if 22 <= hours_ago <= 26:  # 24h ± 2h
                features["temp_24h_ago"] = temp
                break

        # Durchschnitt zur gleichen Tageszeit (± 1 Stunde)
        same_hour_temps = []
        same_hour_vorlauf = []

        for i, (ts, temp) in enumerate(self.aussen_temp_longterm):
            ts_hour = ts.hour + ts.minute / 60.0
            hour_diff = abs(ts_hour - current_hour)
            # Berücksichtige Wrap-around (23h und 0h sind nah)
            if hour_diff > 12:
                hour_diff = 24 - hour_diff

            if hour_diff <= 1.5:  # ± 1.5 Stunden
                same_hour_temps.append(temp)
                if i < len(self.vorlauf_longterm):
                    same_hour_vorlauf.append(self.vorlauf_longterm[i][1])

        if same_hour_temps:
            features["temp_same_hour_avg"] = sum(same_hour_temps) / len(same_hour_temps)
        if same_hour_vorlauf:
            features["vorlauf_same_hour_avg"] = sum(same_hour_vorlauf) / len(
                same_hour_vorlauf
            )

        return features

    def berechne_vorlauf_soll(
        self,
        aussen_temp: float,
        raum_ist: float,
        raum_soll: float,
        vorlauf_ist: float,
    ) -> tuple[float, dict[str, float]]:
        """Berechnet optimalen Vorlauf-Sollwert.

        Returns:
            tuple: (vorlauf_soll, features_dict)
        """

        features = self._erstelle_features(
            aussen_temp, raum_ist, raum_soll, vorlauf_ist
        )

        # Während Kaltstart: Heizkurve verwenden
        if (
            self.use_fallback
            and self.predictions_count < self.min_predictions_for_model
        ):
            vorlauf_soll = self._heizkurve_fallback(
                aussen_temp, features["raum_abweichung"]
            )
            _LOGGER.debug(
                "Verwende Heizkurve (Kaltstart %d/%d): %.1f°C",
                self.predictions_count,
                self.min_predictions_for_model,
                vorlauf_soll,
            )
        else:
            # ML-Model verwenden
            try:
                vorlauf_soll = self.model.predict_one(features)

                # Sanity check: Falls Model unrealistische Werte liefert
                if vorlauf_soll < 15 or vorlauf_soll > 70:
                    _LOGGER.warning(
                        "Model liefert unrealistischen Wert %.1f°C, verwende Heizkurve",
                        vorlauf_soll,
                    )
                    vorlauf_soll = self._heizkurve_fallback(
                        aussen_temp, features["raum_abweichung"]
                    )
                elif self.use_fallback:
                    _LOGGER.info("Wechsel von Heizkurve zu ML-Model")
                    self.use_fallback = False

            except Exception as e:
                _LOGGER.error("Fehler bei Model-Prediction: %s", e)
                vorlauf_soll = self._heizkurve_fallback(
                    aussen_temp, features["raum_abweichung"]
                )

        # Begrenzung auf konfigurierten Bereich
        vorlauf_soll = max(self.min_vorlauf, min(self.max_vorlauf, vorlauf_soll))

        self.predictions_count += 1

        _LOGGER.info(
            "Vorlauf-Soll berechnet: %.1f°C (Außen: %.1f°C, Raum: %.1f/%.1f°C, Trend: %.2f)",
            vorlauf_soll,
            aussen_temp,
            raum_ist,
            raum_soll,
            features["aussen_trend"],
        )

        _LOGGER.info(
            "berechne_vorlauf_soll() min: %.1f°C, max: %.1f°C",
            self.min_vorlauf,
            self.max_vorlauf,
        )  # noqa: HASS_LOGGER_CAPITAL

        return vorlauf_soll, features

    def lerne(
        self,
        aussen_temp: float,
        raum_ist: float,
        raum_soll: float,
        vorlauf_ist: float,
        tatsaechlicher_vorlauf: float,
    ) -> None:
        """Lernt aus tatsächlichem Vorlauf-Wert.

        Args:
            aussen_temp: Außentemperatur
            raum_ist: Aktuelle Raumtemperatur
            raum_soll: Soll-Raumtemperatur
            vorlauf_ist: Aktueller Vorlauf
            tatsaechlicher_vorlauf: Der Vorlauf, der tatsächlich eingestellt wurde
        """

        features = self._erstelle_features(
            aussen_temp, raum_ist, raum_soll, vorlauf_ist
        )

        try:
            # Prediction für Metrik
            prediction = self.model.predict_one(features)

            # Model aktualisieren (Online-Learning)
            self.model.learn_one(features, tatsaechlicher_vorlauf)

            # Metrik aktualisieren
            self.metric.update(tatsaechlicher_vorlauf, prediction)

            _LOGGER.debug(
                "Model gelernt: Vorhersage=%.1f°C, Tatsächlich=%.1f°C, MAE=%.2f",
                prediction,
                tatsaechlicher_vorlauf,
                self.metric.get(),
            )

        except Exception as e:
            _LOGGER.error("Fehler beim Learning: %s", e)

    def get_model_stats(self) -> dict[str, float]:
        """Gibt Model-Statistiken zurück."""
        return {
            "mae": self.metric.get() if self.predictions_count > 0 else 0.0,
            "predictions_count": self.predictions_count,
            "use_fallback": self.use_fallback,
            "history_size": len(self.aussen_temp_history),
        }

    def reset_model(self) -> None:
        """Setzt das Model zurück (für Neustart)."""
        _LOGGER.info("-> reset_model()")

        self._setup(
            min_vorlauf=self.min_vorlauf,
            max_vorlauf=self.max_vorlauf,
            learning_rate=0.01,
            trend_history_size=self.trend_history_size,
        )
