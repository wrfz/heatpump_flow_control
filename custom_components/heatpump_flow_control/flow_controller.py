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


class ErfahrungsSpeicher:
    """Speichert Erfahrungen für verzögertes Reward-basiertes Lernen."""

    def __init__(self, max_size: int = 200) -> None:
        """Initialize experience storage."""
        self.erfahrungen = []
        self.max_size = max_size

    def speichere_erfahrung(
        self,
        features: dict[str, float],
        vorlauf_gesetzt: float,
        raum_ist_vorher: float,
        raum_soll: float,
        power_aktuell: float | None = None,
    ) -> None:
        """Speichert eine Erfahrung mit Timestamp.

        Args:
            features: Feature-Dictionary zum Zeitpunkt der Entscheidung
            vorlauf_gesetzt: Der gesetzte Vorlauf-Wert
            raum_ist_vorher: Raumtemperatur zum Zeitpunkt der Entscheidung
            raum_soll: Soll-Raumtemperatur
            power_aktuell: Aktuelle Leistung (positiv=Netzbezug, negativ=Einspeisung)
        """
        erfahrung = {
            "timestamp": datetime.now(),
            "features": features.copy(),
            "vorlauf_gesetzt": vorlauf_gesetzt,
            "raum_ist_vorher": raum_ist_vorher,
            "raum_soll": raum_soll,
            "power_aktuell": power_aktuell,
            "gelernt": False,
        }
        self.erfahrungen.append(erfahrung)

        # Begrenze Größe
        if len(self.erfahrungen) > self.max_size:
            self.erfahrungen.pop(0)

    def hole_lernbare_erfahrungen(
        self, min_stunden: float = 2.0, max_stunden: float = 6.0
    ) -> list:
        """Gibt Erfahrungen zurück, die alt genug sind zum Lernen.

        Args:
            min_stunden: Minimales Alter in Stunden
            max_stunden: Maximales Alter in Stunden

        Returns:
            Liste von lernbaren Erfahrungen
        """
        now = datetime.now()
        lernbar = []

        for erf in self.erfahrungen:
            if erf["gelernt"]:
                continue

            stunden_alt = (now - erf["timestamp"]).total_seconds() / 3600
            if min_stunden <= stunden_alt <= max_stunden:
                lernbar.append(erf)

        return lernbar

    def markiere_gelernt(self, erfahrung: dict) -> None:
        """Markiert eine Erfahrung als gelernt."""
        erfahrung["gelernt"] = True

    def get_stats(self) -> dict[str, int]:
        """Gibt Statistiken über den Speicher zurück."""
        total = len(self.erfahrungen)
        gelernt = sum(1 for e in self.erfahrungen if e["gelernt"])
        ungelernt = total - gelernt

        return {
            "total": total,
            "gelernt": gelernt,
            "ungelernt": ungelernt,
        }


class FlowController:
    """Flow Controller für Vorlauf-Temperatur Regelung."""

    def __init__(
        self,
        min_vorlauf: float = DEFAULT_MIN_VORLAUF,
        max_vorlauf: float = DEFAULT_MAX_VORLAUF,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        trend_history_size: int = DEFAULT_TREND_HISTORY_SIZE,
    ) -> None:
        """Initialize the flow controller."""

        # Initialisiere power-Attribute VOR _setup() für backwards compatibility
        self.power_history = []
        self.power_enabled = False

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
        force_fallback: bool = False,
    ) -> None:
        """Initialize the flow controller.

        Args:
            min_vorlauf: Minimale Vorlauftemperatur
            max_vorlauf: Maximale Vorlauftemperatur
            learning_rate: Learning Rate für das Model
            trend_history_size: Größe der Trend-Historie
            force_fallback: Wenn True, erzwinge Fallback-Modus (z.B. bei Model-Reset)
        """

        # Speichere alten use_fallback Status (für Restart-Persistenz)
        old_use_fallback = getattr(self, "use_fallback", None)

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

        # NEU: Power-Historie (Strompreis/PV-Überschuss)
        self.power_history = []  # Speichert (timestamp, power) Tupel
        self.power_enabled = False  # Wird auf True gesetzt wenn Sensor konfiguriert

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

        # NEU: Erfahrungsspeicher für Reward-basiertes Lernen
        self.erfahrungs_speicher = ErfahrungsSpeicher(max_size=200)
        self.reward_learning_enabled = True  # Kann deaktiviert werden für Tests
        self.min_reward_hours = 2.0  # Mindeststunden bis Bewertung
        self.max_reward_hours = 6.0  # Maximalstunden bis Bewertung

        _LOGGER.info(
            "_setup(): min=%s, max=%s, lr=%s, history=%s, longterm=%s, reward_learning=%s, power=%s",
            min_vorlauf,
            max_vorlauf,
            learning_rate,
            trend_history_size,
            self.longterm_history_size,
            self.reward_learning_enabled,
            self.power_enabled,
        )

    def _ensure_attributes(self) -> None:
        """Stellt sicher, dass alle Attribute existieren (Backwards Compatibility).

        Wird bei Bedarf aufgerufen wenn Objekt aus Storage geladen wurde.
        """
        if not hasattr(self, "power_enabled"):
            self.power_enabled = False
        if not hasattr(self, "power_history"):
            self.power_history = []
        if not hasattr(self, "erfahrungs_speicher"):
            self.erfahrungs_speicher = ErfahrungsSpeicher(max_size=200)
        if not hasattr(self, "reward_learning_enabled"):
            self.reward_learning_enabled = True
        if not hasattr(self, "min_reward_hours"):
            self.min_reward_hours = 2.0
        if not hasattr(self, "max_reward_hours"):
            self.max_reward_hours = 6.0

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
        power_aktuell: float | None = None,
    ) -> dict[str, float]:
        """Erstellt Feature-Dictionary für das Model.

        Args:
            aussen_temp: Außentemperatur
            raum_ist: Ist-Raumtemperatur
            raum_soll: Soll-Raumtemperatur
            vorlauf_ist: Ist-Vorlauf
            power_aktuell: Aktuelle Leistung (positiv=Netzbezug, negativ=Einspeisung)
        """

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

        # NEU: Power-Features berechnen (wenn aktiviert)
        power_features = self._berechne_power_features(now, stunde, power_aktuell)

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

        # Power-Features hinzufügen
        features.update(power_features)

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

    def _berechne_power_features(
        self, now: datetime, current_hour: float, power_aktuell: float | None
    ) -> dict[str, float]:
        """Berechnet Power-basierte Features (PV-Überschuss/Strompreis).

        WICHTIG: Verwendet NICHT den aktuellen Power-Wert als Feature, um
        "falsch herum lernen" zu vermeiden. Stattdessen werden historische
        Durchschnittswerte zur gleichen Tageszeit verwendet.

        Args:
            now: Aktueller Zeitpunkt
            current_hour: Aktuelle Stunde (0-24)
            power_aktuell: Aktueller Leistungswert (nur für Historie-Update)

        Returns:
            Dictionary mit Power-Features
        """
        features = {
            "power_avg_same_hour": 0.0,  # Durchschnitt zur gleichen Tageszeit
            "power_avg_1h": 0.0,  # Durchschnitt letzte 1h
            "power_avg_3h": 0.0,  # Durchschnitt letzte 3h
            "power_favorable_hours": 0.0,  # Anteil günstiger Stunden heute
        }

        if not self.power_enabled or not self.power_history:
            return features

        # Power-Historie aktualisieren (alle 10 Minuten)
        if power_aktuell is not None:
            if (
                not self.power_history
                or (now - self.power_history[-1][0]).total_seconds() > 600
            ):
                self.power_history.append((now, power_aktuell))

                # Begrenze auf longterm_history_size
                if (
                    len(self.power_history) > self.longterm_history_size * 6
                ):  # 10-min Intervalle
                    self.power_history.pop(0)

        # Durchschnitt zur gleichen Tageszeit (± 1.5h) - DAS lernt das Model!
        same_hour_power = []
        for ts, power in self.power_history:
            ts_hour = ts.hour + ts.minute / 60.0
            hour_diff = abs(ts_hour - current_hour)
            if hour_diff > 12:
                hour_diff = 24 - hour_diff

            if hour_diff <= 1.5:  # ± 1.5 Stunden
                same_hour_power.append(power)

        if same_hour_power:
            features["power_avg_same_hour"] = sum(same_hour_power) / len(
                same_hour_power
            )

        # Durchschnitt letzte 1h
        power_1h = [
            power
            for ts, power in self.power_history
            if (now - ts).total_seconds() < 3600
        ]
        if power_1h:
            features["power_avg_1h"] = sum(power_1h) / len(power_1h)

        # Durchschnitt letzte 3h
        power_3h = [
            power
            for ts, power in self.power_history
            if (now - ts).total_seconds() < 10800
        ]
        if power_3h:
            features["power_avg_3h"] = sum(power_3h) / len(power_3h)

        # Anteil günstiger Stunden (negative Werte = Einspeisung) heute
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        power_today = [power for ts, power in self.power_history if ts >= today_start]
        if power_today:
            favorable_count = sum(1 for p in power_today if p < 0)  # Einspeisung
            features["power_favorable_hours"] = favorable_count / len(power_today)

        _LOGGER.debug(
            "Power-Features: same_hour_avg=%.1f, 1h_avg=%.1f, favorable=%.1f%%",
            features["power_avg_same_hour"],
            features["power_avg_1h"],
            features["power_favorable_hours"] * 100,
        )

        return features

    def update_power_sensor(self, power_value: float | None) -> None:
        """Aktiviert/deaktiviert Power-Sensor.

        Args:
            power_value: Aktueller Power-Wert oder None um zu deaktivieren
        """
        if power_value is None:
            self.power_enabled = False
            _LOGGER.info("Power-Sensor deaktiviert")
        else:
            self.power_enabled = True
            _LOGGER.debug("Power-Sensor aktiviert: %.1f W", power_value)

    def _bewerte_erfahrung(
        self,
        erfahrung: dict,
        raum_ist_jetzt: float,
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
        raum_soll = erfahrung["raum_soll"]
        vorlauf_gesetzt = erfahrung["vorlauf_gesetzt"]

        # Wie groß ist die Abweichung jetzt (2-3h später)?
        abweichung = raum_ist_jetzt - raum_soll
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

        # NEU: Power-basierte Reward-Anpassung
        if (
            self.power_enabled
            and "power_aktuell" in erfahrung
            and erfahrung["power_aktuell"] is not None
        ):
            power_damals = erfahrung["power_aktuell"]

            # Bonus: Bei günstigen Strom (Einspeisung/PV-Überschuss) mehr geheizt
            if (
                power_damals < -500 and reward >= 0
            ):  # Einspeisung > 500W und gutes Ergebnis
                reward = min(1.0, reward + 0.2)  # Bonus für PV-Nutzung
                _LOGGER.debug(
                    "Power-Bonus: Bei PV-Überschuss (%.0f W) wurde vorgeheizt - gut!",
                    power_damals,
                )

            # Bonus: Bei teuerem Strom (Netzbezug) weniger geheizt und Raum trotzdem OK
            elif power_damals > 1000 and reward >= 0:  # Netzbezug > 1kW
                # Prüfe ob Vorlauf niedriger war als üblich zur gleichen Tageszeit
                vorlauf_avg_same_hour = erfahrung["features"].get(
                    "vorlauf_same_hour_avg", 0
                )
                if (
                    vorlauf_avg_same_hour > 0
                    and vorlauf_gesetzt < vorlauf_avg_same_hour
                ):
                    reward = min(1.0, reward + 0.15)  # Bonus für Strom-Sparen
                    _LOGGER.debug(
                        "Power-Bonus: Bei Netzbezug (%.0f W) wurde gespart - gut!",
                        power_damals,
                    )

        # Korrektur berechnen (nur bei schlechtem Ergebnis)
        if reward < 0:
            if abweichung < 0:  # Zu kalt → Vorlauf war zu niedrig
                # Berechne Korrektur: pro 0.5°C Abweichung +2°C Vorlauf
                korrektur = abs(abweichung) * 4.0
                korrigierter_vorlauf = vorlauf_gesetzt + korrektur
            else:  # Zu warm → Vorlauf war zu hoch
                # Berechne Korrektur: pro 0.5°C Abweichung -1.5°C Vorlauf
                # (Weniger aggressiv, da Überhitzung träger reagiert)
                korrektur = abs(abweichung) * 3.0
                korrigierter_vorlauf = vorlauf_gesetzt - korrektur
        else:
            korrigierter_vorlauf = vorlauf_gesetzt

        return reward, korrigierter_vorlauf

    def _lerne_aus_erfahrungen(self, raum_ist_jetzt: float) -> dict[str, int]:
        """Lernt aus allen verfügbaren Erfahrungen.

        Args:
            raum_ist_jetzt: Aktuelle Raumtemperatur

        Returns:
            Statistiken über das Lernen (gelernt_positiv, gelernt_negativ)
        """
        if not self.reward_learning_enabled:
            return {"gelernt_positiv": 0, "gelernt_negativ": 0}

        lernbare = self.erfahrungs_speicher.hole_lernbare_erfahrungen(
            min_stunden=self.min_reward_hours, max_stunden=self.max_reward_hours
        )

        stats = {"gelernt_positiv": 0, "gelernt_negativ": 0}

        # Aktuellen Trend für Bewertung holen
        trends = self._berechne_trends()
        aussen_trend = trends.get("aussen_trend", 0.0)

        for erfahrung in lernbare:
            # Bewerte die Erfahrung
            reward, korrigierter_vorlauf = self._bewerte_erfahrung(
                erfahrung, raum_ist_jetzt, aussen_trend
            )

            features = erfahrung["features"]

            try:
                if reward >= 0:
                    # Gute Erfahrung: Lerne mit original Vorlauf
                    # Höheres sample_weight für sehr gute Erfahrungen
                    sample_weight = 1.0 + reward  # 1.0 bis 2.0
                    self.model.learn_one(
                        features,
                        erfahrung["vorlauf_gesetzt"],
                        sample_weight=sample_weight,
                    )

                    stats["gelernt_positiv"] += 1

                    _LOGGER.info(
                        "✓ Reward-Lernen (Reward=%.1f, Weight=%.1f): Vorlauf %.1f°C war gut",
                        reward,
                        sample_weight,
                        erfahrung["vorlauf_gesetzt"],
                    )
                else:
                    # Schlechte Erfahrung: Lerne mit korrigiertem Vorlauf
                    sample_weight = abs(reward) * 1.5  # 0.75 bis 1.5
                    self.model.learn_one(
                        features, korrigierter_vorlauf, sample_weight=sample_weight
                    )

                    stats["gelernt_negativ"] += 1

                    _LOGGER.warning(
                        "✗ Reward-Lernen (Reward=%.1f, Weight=%.1f): Vorlauf %.1f°C → besser %.1f°C",
                        reward,
                        sample_weight,
                        erfahrung["vorlauf_gesetzt"],
                        korrigierter_vorlauf,
                    )

                # Markiere als gelernt
                self.erfahrungs_speicher.markiere_gelernt(erfahrung)

            except Exception as e:
                _LOGGER.error("Fehler beim Reward-Learning: %s", e)

        if stats["gelernt_positiv"] + stats["gelernt_negativ"] > 0:
            _LOGGER.info(
                "Reward-Lernen abgeschlossen: %d positiv, %d negativ (von %d lernbar)",
                stats["gelernt_positiv"],
                stats["gelernt_negativ"],
                len(lernbare),
            )

        return stats

    def berechne_vorlauf_soll(
        self,
        aussen_temp: float,
        raum_ist: float,
        raum_soll: float,
        vorlauf_ist: float,
        power_aktuell: float | None = None,
    ) -> tuple[float, dict[str, float]]:
        """Berechnet optimalen Vorlauf-Sollwert.

        Args:
            aussen_temp: Außentemperatur
            raum_ist: Ist-Raumtemperatur
            raum_soll: Soll-Raumtemperatur
            vorlauf_ist: Ist-Vorlauf
            power_aktuell: Aktuelle Leistung (positiv=Netzbezug, negativ=Einspeisung)

        Returns:
            tuple: (vorlauf_soll, features_dict)
        """
        # Backwards compatibility: Stelle sicher, dass alle Attribute existieren
        self._ensure_attributes()

        # Flag um zu tracken ob Model in dieser Berechnung zurückgesetzt wurde
        model_was_reset = False

        features = self._erstelle_features(
            aussen_temp, raum_ist, raum_soll, vorlauf_ist, power_aktuell
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
                            min_vorlauf=self.min_vorlauf,
                            max_vorlauf=self.max_vorlauf,
                            learning_rate=0.01,  # Default
                            trend_history_size=self.trend_history_size,
                            force_fallback=True,  # Erzwinge Fallback nach Reset
                        )
                        self.predictions_count = 0
                        # Diese Vorhersage war ungültig, zählt nicht mit
                        model_was_reset = True

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

        # Nur inkrementieren wenn Model nicht gerade zurückgesetzt wurde
        if not model_was_reset:
            self.predictions_count += 1

        # NEU: Speichere diese Entscheidung für späteres Reward-Learning
        if self.reward_learning_enabled:
            self.erfahrungs_speicher.speichere_erfahrung(
                features=features,
                vorlauf_gesetzt=vorlauf_soll,
                raum_ist_vorher=raum_ist,
                raum_soll=raum_soll,
                power_aktuell=power_aktuell,
            )

        # NEU: Lerne aus alten Erfahrungen (2-6h alt)
        if self.reward_learning_enabled and self.predictions_count % 2 == 0:
            # Nur jedes 2. Mal ausführen um Performance zu schonen
            lern_stats = self._lerne_aus_erfahrungen(raum_ist_jetzt=raum_ist)
            if lern_stats["gelernt_positiv"] + lern_stats["gelernt_negativ"] > 0:
                _LOGGER.debug("Reward-Learning Stats: %s", lern_stats)

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

    def get_model_stats(self) -> dict[str, float]:
        """Gibt Model-Statistiken zurück."""
        erfahrungs_stats = self.erfahrungs_speicher.get_stats()

        return {
            "mae": self.metric.get() if self.predictions_count > 0 else 0.0,
            "predictions_count": self.predictions_count,
            "use_fallback": self.use_fallback,
            "history_size": len(self.aussen_temp_history),
            "erfahrungen_total": erfahrungs_stats["total"],
            "erfahrungen_gelernt": erfahrungs_stats["gelernt"],
            "erfahrungen_wartend": erfahrungs_stats["ungelernt"],
            "reward_learning_enabled": self.reward_learning_enabled,
        }

    def reset_model(self) -> None:
        """Setzt das Model zurück (für Neustart)."""
        _LOGGER.info("-> reset_model()")

        self._setup(
            min_vorlauf=self.min_vorlauf,
            max_vorlauf=self.max_vorlauf,
            learning_rate=0.01,
            trend_history_size=self.trend_history_size,
            force_fallback=True,  # Erzwinge Fallback nach manuellem Reset
        )
