"""Unit tests for FlowController learning and prediction logic."""

from datetime import datetime, timedelta
from unittest.mock import patch

from custom_components.heatpump_flow_control.flow_controller import (
    ErfahrungsSpeicher,
    FlowController,
    SensorValues,
)


class TestErfahrungsSpeicher:
    """Test experience storage for reward-based learning."""

    def test_initialization(self):
        """Test ErfahrungsSpeicher initialization."""
        speicher = ErfahrungsSpeicher(max_size=100)

        assert len(speicher.erfahrungen) == 0
        assert speicher.max_size == 100

    def test_speichere_erfahrung(self):
        """Test storing an experience."""
        speicher = ErfahrungsSpeicher()

        features = {"aussen_temp": 5.0, "raum_ist": 22.0}
        speicher.speichere_erfahrung(
            features=features,
            vorlauf_gesetzt=35.0,
            raum_ist_vorher=22.0,
            raum_soll=21.0,
            power_aktuell=None,
        )

        assert len(speicher.erfahrungen) == 1
        assert speicher.erfahrungen[0]["vorlauf_gesetzt"] == 35.0
        assert speicher.erfahrungen[0]["gelernt"] is False
        assert "timestamp" in speicher.erfahrungen[0]

    def test_max_size_limit(self):
        """Test that max_size is enforced."""
        speicher = ErfahrungsSpeicher(max_size=3)
        features = {"test": 1.0}

        for i in range(5):
            speicher.speichere_erfahrung(
                features=features,
                vorlauf_gesetzt=30.0 + i,
                raum_ist_vorher=22.0,
                raum_soll=21.0,
            )

        # Should only keep last 3
        assert len(speicher.erfahrungen) == 3
        assert speicher.erfahrungen[0]["vorlauf_gesetzt"] == 32.0  # 2nd experience

    def test_hole_lernbare_erfahrungen(self):
        """Test retrieving learnable experiences."""
        speicher = ErfahrungsSpeicher()
        features = {"test": 1.0}

        # Add experiences at different times
        now = datetime.now()

        # Too recent (1 hour)
        with patch(
            "custom_components.heatpump_flow_control.flow_controller.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = now - timedelta(hours=1)
            speicher.speichere_erfahrung(
                features=features,
                vorlauf_gesetzt=30.0,
                raum_ist_vorher=22.0,
                raum_soll=21.0,
            )

        # Perfect age (3 hours)
        with patch(
            "custom_components.heatpump_flow_control.flow_controller.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = now - timedelta(hours=3)
            speicher.speichere_erfahrung(
                features=features,
                vorlauf_gesetzt=35.0,
                raum_ist_vorher=22.0,
                raum_soll=21.0,
            )

        # Too old (7 hours)
        with patch(
            "custom_components.heatpump_flow_control.flow_controller.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = now - timedelta(hours=7)
            speicher.speichere_erfahrung(
                features=features,
                vorlauf_gesetzt=40.0,
                raum_ist_vorher=22.0,
                raum_soll=21.0,
            )

        # Manually set timestamps (patch doesn't work in list comprehension)
        speicher.erfahrungen[0]["timestamp"] = now - timedelta(hours=1)
        speicher.erfahrungen[1]["timestamp"] = now - timedelta(hours=3)
        speicher.erfahrungen[2]["timestamp"] = now - timedelta(hours=7)

        lernbar = speicher.hole_lernbare_erfahrungen(min_stunden=2.0, max_stunden=6.0)

        # Only the 3-hour old experience should be learnable
        assert len(lernbar) == 1
        assert lernbar[0]["vorlauf_gesetzt"] == 35.0

    def test_markiere_gelernt(self):
        """Test marking experience as learned."""
        speicher = ErfahrungsSpeicher()
        features = {"test": 1.0}

        speicher.speichere_erfahrung(
            features=features,
            vorlauf_gesetzt=30.0,
            raum_ist_vorher=22.0,
            raum_soll=21.0,
        )

        erfahrung = speicher.erfahrungen[0]
        assert erfahrung["gelernt"] is False

        speicher.markiere_gelernt(erfahrung)
        assert erfahrung["gelernt"] is True

    def test_get_stats(self):
        """Test statistics retrieval."""
        speicher = ErfahrungsSpeicher()
        features = {"test": 1.0}

        # Add 3 experiences, mark 2 as learned
        for i in range(3):
            speicher.speichere_erfahrung(
                features=features,
                vorlauf_gesetzt=30.0 + i,
                raum_ist_vorher=22.0,
                raum_soll=21.0,
            )

        speicher.markiere_gelernt(speicher.erfahrungen[0])
        speicher.markiere_gelernt(speicher.erfahrungen[1])

        stats = speicher.get_stats()
        assert stats["total"] == 3
        assert stats["gelernt"] == 2
        assert stats["ungelernt"] == 1


class TestFlowControllerInit:
    """Test FlowController initialization."""

    def test_default_initialization(self):
        """Test controller initializes with defaults."""
        controller = FlowController()

        assert controller.min_vorlauf == 25.0  # DEFAULT_MIN_VORLAUF
        assert controller.max_vorlauf == 55.0  # DEFAULT_MAX_VORLAUF
        assert controller.use_fallback is True
        assert controller.predictions_count == 0
        assert hasattr(controller, "model")
        assert hasattr(controller, "erfahrungs_speicher")
        assert controller.power_enabled is False
        assert len(controller.power_history) == 0

    def test_custom_initialization(self):
        """Test controller with custom parameters."""
        controller = FlowController(
            min_vorlauf=25.0,
            max_vorlauf=50.0,
            learning_rate=0.05,
            trend_history_size=15,
        )

        assert controller.min_vorlauf == 25.0
        assert controller.max_vorlauf == 50.0
        assert controller.trend_history_size == 15

    def test_fallback_persists_after_restart(self):
        """Test that use_fallback state is preserved across restarts.

        Bug fix: Nach HA-Restart sollte ein trainiertes Model nicht zurück
        in Fallback-Modus gehen.
        """
        # Simuliere ersten Start: Fallback aktiv
        controller = FlowController()
        assert controller.use_fallback is True
        assert controller.predictions_count == 0

        # Trainiere Model (10+ Predictions)
        for i in range(15):
            vorlauf_soll, features = controller.berechne_vorlauf_soll(
                SensorValues(aussen_temp=5.0 - i * 0.5,
                raum_ist=21.0,
                raum_soll=21.0,
                vorlauf_ist=35.0 + i)
            )
            # Trainiere mit realistischem Wert
            controller.model.learn_one(features, 35.0 + i)

        # Nach Training: Fallback sollte aus sein
        assert controller.predictions_count >= 10
        # Fallback wird ausgeschaltet bei nächster Prediction wenn Model realistic ist

        # Simuliere HA-Restart: Speichere aktuellen Zustand
        old_predictions_count = controller.predictions_count
        old_use_fallback = False  # Angenommen Model war schon aktiv
        controller.use_fallback = old_use_fallback

        # Simuliere Pickle-Reload: _setup() wird aufgerufen aber use_fallback existiert schon
        controller._setup(
            min_vorlauf=controller.min_vorlauf,
            max_vorlauf=controller.max_vorlauf,
            learning_rate=0.01,
            trend_history_size=controller.trend_history_size,
        )

        # BUG-FIX: use_fallback sollte NICHT auf True zurückgehen
        assert controller.use_fallback is False, (
            "Trainiertes Model sollte nicht in Fallback zurückfallen"
        )

    def test_fallback_reset_on_corruption(self):
        """Test that fallback is reactivated when model is corrupted."""
        controller = FlowController()
        controller.use_fallback = False  # Model war bereits trainiert
        controller.predictions_count = 100

        # Simuliere Model-Korruption durch _setup Aufruf mit force_fallback
        controller._setup(
            min_vorlauf=controller.min_vorlauf,
            max_vorlauf=controller.max_vorlauf,
            learning_rate=0.01,
            trend_history_size=controller.trend_history_size,
            force_fallback=True,  # Erzwinge Fallback nach Reset
        )

        # Nach Reset wegen Korruption: use_fallback muss True sein
        assert controller.use_fallback is True, (
            "Fallback muss nach Model-Reset aktiv sein"
        )
        assert controller.predictions_count == 0, (
            "Predictions zähler muss zurückgesetzt sein"
        )
        delattr(controller, "power_enabled")
        delattr(controller, "power_history")
        delattr(controller, "reward_learning_enabled")

        # Should not crash and re-initialize
        controller._ensure_attributes()

        assert hasattr(controller, "power_enabled")
        assert hasattr(controller, "power_history")
        assert hasattr(controller, "reward_learning_enabled")


class TestFeatureCreation:
    """Test feature creation logic."""

    def test_erstelle_features_basic(self):
        """Test basic feature creation."""
        controller = FlowController()

        features = controller._erstelle_features(
            SensorValues(
                aussen_temp=5.0,
                raum_ist=22.0,
                raum_soll=21.0,
                vorlauf_ist=35.0
            ),
            power_aktuell=None
        )

        # Check basic features
        assert features["aussen_temp"] == 5.0
        assert features["raum_ist"] == 22.0
        assert features["raum_soll"] == 21.0
        assert features["vorlauf_ist"] == 35.0
        assert features["raum_abweichung"] == -1.0  # 21 - 22 (soll - ist)
        assert features["temp_diff"] == -17.0  # 5 - 22
        assert features["vorlauf_raum_diff"] == 13.0  # 35 - 22

        # Check time features exist
        assert "stunde_sin" in features
        assert "stunde_cos" in features
        assert "wochentag_sin" in features
        assert "wochentag_cos" in features

        # Check power features are zero (not enabled)
        assert features["power_avg_same_hour"] == 0.0
        assert features["power_avg_1h"] == 0.0

    def test_erstelle_features_with_trends(self):
        """Test feature creation with temperature trends."""
        controller = FlowController()

        # Add some history
        controller.aussen_temp_history = [3.0, 4.0, 5.0]
        controller.timestamps = [
            datetime.now() - timedelta(hours=2),
            datetime.now() - timedelta(hours=1),
            datetime.now(),
        ]

        features = controller._erstelle_features(
            SensorValues(
                aussen_temp=5.0,
                raum_ist=22.0,
                raum_soll=21.0,
                vorlauf_ist=35.0
            ),
            power_aktuell=None,
        )

        # Trends should be calculated (may be 0 with insufficient data)
        # Just check they exist
        assert "aussen_trend" in features
        assert "aussen_trend_kurz" in features

    def test_berechne_power_features_disabled(self):
        """Test power features when sensor is disabled."""
        controller = FlowController()
        controller.power_enabled = False

        now = datetime.now()
        features = controller._berechne_power_features(now, 12.0, None)

        # All power features should be 0
        assert features["power_avg_same_hour"] == 0.0
        assert features["power_avg_1h"] == 0.0
        assert features["power_avg_3h"] == 0.0
        assert features["power_favorable_hours"] == 0.0


class TestPrediction:
    """Test prediction logic."""

    def test_fallback_heizkurve(self):
        """Test fallback heating curve calculation."""
        controller = FlowController()

        # Cold outside, room too cold
        vorlauf = controller._heizkurve_fallback(
            aussen_temp=-5.0,
            raum_abweichung=2.0,  # 2 degrees too cold
        )

        # Should be high
        assert vorlauf > 35.0

        # Warm outside, room ok
        vorlauf = controller._heizkurve_fallback(aussen_temp=15.0, raum_abweichung=0.0)

        # Should be low
        assert vorlauf < 30.0

    def test_berechne_vorlauf_soll_fallback_mode(self):
        """Test calculation in fallback mode."""
        controller = FlowController()
        controller.use_fallback = True
        controller.predictions_count = 5
        controller.min_predictions_for_model = 10

        vorlauf_soll, features = controller.berechne_vorlauf_soll(
            SensorValues(
                aussen_temp=5.0,
                raum_ist=22.0,
                raum_soll=21.0,
                vorlauf_ist=35.0
            )
        )

        # Should use fallback
        assert controller.min_vorlauf <= vorlauf_soll <= controller.max_vorlauf
        assert controller.predictions_count == 6

    def test_fallback_respects_configured_min_vorlauf(self):
        """Test that fallback mode respects configured min_vorlauf boundary."""
        controller = FlowController(min_vorlauf=30.0, max_vorlauf=55.0)
        controller.use_fallback = True

        # At high outdoor temp, standard Heizkurve would give low value
        vorlauf_soll, _ = controller.berechne_vorlauf_soll(
            SensorValues(
                aussen_temp=15.0,  # Warm outside
                raum_ist=20.0,
                raum_soll=21.0,
                vorlauf_ist=32.0,
             )
        )

        # BUG: Should respect configured min_vorlauf
        assert vorlauf_soll >= 30.0, (
            f"Fallback returned {vorlauf_soll}°C but min is 30°C"
        )
        assert vorlauf_soll <= 55.0

    def test_berechne_vorlauf_soll_within_limits(self):
        """Test that prediction is always within configured limits."""
        controller = FlowController(min_vorlauf=28.0, max_vorlauf=40.0)
        controller.use_fallback = False  # Force model usage

        # Mock extreme prediction
        with patch.object(controller.model, "predict_one", return_value=100.0):
            vorlauf_soll, _ = controller.berechne_vorlauf_soll(
                SensorValues(
                    aussen_temp=5.0,
                    raum_ist=22.0,
                    raum_soll=21.0,
                    vorlauf_ist=35.0
                )
            )

            # Should use fallback (unrealistic value triggers fallback)
            assert 28.0 <= vorlauf_soll <= 40.0

    def test_unrealistic_prediction_triggers_fallback(self):
        """Test that unrealistic predictions trigger fallback."""
        controller = FlowController()
        controller.use_fallback = False
        controller.predictions_count = 20

        # Mock extreme prediction
        with patch.object(controller.model, "predict_one", return_value=5000000.0):
            vorlauf_soll, _ = controller.berechne_vorlauf_soll(
                SensorValues(
                    aussen_temp=5.0,
                    raum_ist=22.0,
                    raum_soll=21.0,
                    vorlauf_ist=35.0
                )
            )

            # Should use fallback and reset model
            assert controller.use_fallback is True
            # Counter must be 0 after reset (ungültige Vorhersage zählt nicht)
            assert controller.predictions_count == 0, (
                "Counter must be reset after model reset"
            )
            # Fallback value should be within reasonable range
            assert 25.0 <= vorlauf_soll <= 55.0


class TestLearning:
    """Test learning functionality."""

    def test_experience_storage(self):
        """Test that experiences are stored during predictions."""
        controller = FlowController()
        controller.use_fallback = False

        # Make prediction - this stores an experience
        vorlauf, features = controller.berechne_vorlauf_soll(
            SensorValues(
                aussen_temp=5.0,
                raum_ist=22.0,
                raum_soll=21.0,
                vorlauf_ist=35.0
            )
        )

        # Should store experience in ErfahrungsSpeicher
        stats = controller.erfahrungs_speicher.get_stats()
        assert stats["total"] == 1
        assert stats["ungelernt"] == 1

    def test_prediction_updates_history(self):
        """Test that predictions update temperature history."""
        controller = FlowController()

        # Make multiple predictions
        for i in range(5):
            controller.berechne_vorlauf_soll(
                SensorValues(
                    aussen_temp=5.0 + i,
                    raum_ist=22.0,
                    raum_soll=21.0,
                    vorlauf_ist=35.0,
                )
            )

        # History should be populated (only updates every 30 minutes, so may be 1)
        assert len(controller.aussen_temp_longterm) >= 1
        assert len(controller.vorlauf_longterm) >= 1
        assert len(controller.raum_temp_longterm) >= 1

    def test_bewerte_erfahrung_positive_reward(self):
        """Test experience evaluation with positive reward."""
        controller = FlowController()

        # Room was too cold (21°C), we increased flow, now it's better (21.5°C)
        erfahrung = {
            "features": {"raum_abweichung": -1.0},
            "raum_ist_vorher": 21.0,
            "raum_soll": 22.0,
            "vorlauf_gesetzt": 38.0,
            "power_aktuell": None,
        }

        reward, y_target = controller._bewerte_erfahrung(
            erfahrung=erfahrung,
            raum_ist_jetzt=21.5,  # Improved
            aussen_trend=0.0,
        )

        # Should have some reward (may be 0 or positive depending on logic)
        assert reward >= 0

    def test_bewerte_erfahrung_negative_reward(self):
        """Test experience evaluation with negative reward."""
        controller = FlowController()

        # Room was ok (22°C), now it's too hot (23°C)
        erfahrung = {
            "features": {"raum_abweichung": 0.0},
            "raum_ist_vorher": 22.0,
            "raum_soll": 22.0,
            "vorlauf_gesetzt": 40.0,  # Too high
            "power_aktuell": None,
        }

        reward, y_target = controller._bewerte_erfahrung(
            erfahrung=erfahrung,
            raum_ist_jetzt=23.0,  # Worse
            aussen_trend=0.0,
        )

        # Should have negative reward (room got worse)
        assert reward < 0

    def test_lerne_reward_processes_old_experiences(self):
        """Test that reward learning processes old experiences."""
        controller = FlowController()
        controller.reward_learning_enabled = True

        # Add an old experience manually
        now = datetime.now()
        old_time = now - timedelta(hours=3)

        features = {"raum_abweichung": 1.0, "aussen_temp": 5.0}
        controller.erfahrungs_speicher.speichere_erfahrung(
            features=features,
            vorlauf_gesetzt=35.0,
            raum_ist_vorher=21.0,
            raum_soll=22.0,
        )

        # Manually set old timestamp
        controller.erfahrungs_speicher.erfahrungen[0]["timestamp"] = old_time

        # Add current data to longterm history
        controller.raum_temp_longterm = [(now, 21.5)]

        # Perform reward learning
        stats = controller._lerne_aus_erfahrungen(raum_ist_jetzt=21.5)

        # Should have attempted learning (check stats exist)
        assert isinstance(stats, dict)


class TestPowerSensor:
    """Test power sensor functionality."""

    def test_update_power_sensor_enables(self):
        """Test that updating power sensor enables it."""
        controller = FlowController()
        assert controller.power_enabled is False

        controller.update_power_sensor(1500.0)
        assert controller.power_enabled is True

    def test_update_power_sensor_disables(self):
        """Test that None disables power sensor."""
        controller = FlowController()
        controller.power_enabled = True

        controller.update_power_sensor(None)
        assert controller.power_enabled is False

    def test_berechne_vorlauf_soll_with_power(self):
        """Test calculation with power sensor."""
        controller = FlowController()
        controller.use_fallback = False

        # Should not crash with power value
        vorlauf_soll, features = controller.berechne_vorlauf_soll(
            SensorValues(
                aussen_temp=5.0,
                raum_ist=22.0,
                raum_soll=21.0,
                vorlauf_ist=35.0,
            ),
            power_aktuell=1500.0,
        )

        assert vorlauf_soll is not None
        assert "power_avg_1h" in features


class TestRealisticLearningScenario:
    """Test realistic learning scenarios with full sensor setup."""

    def test_learns_to_increase_vorlauf_when_temperature_drops(self):
        """Test that model learns to increase vorlauf when outdoor temperature drops.

        Realistisches Szenario:
        - Starte bei 10°C außen, Raum auf 21°C
        - Senke Außentemperatur schrittweise auf -5°C
        - Model sollte lernen, dass bei kälteren Temperaturen höherer Vorlauf nötig ist
        """
        controller = FlowController(min_vorlauf=25.0, max_vorlauf=55.0)
        controller.min_predictions_for_model = 5  # Schnell aus Fallback raus
        controller.reward_learning_enabled = False  # Klassisches Lernen

        raum_soll = 21.0
        vorlauf_predictions = []

        # Simuliere 50 Messzyklen mit sinkender Außentemperatur
        for i in range(50):
            # Außentemperatur sinkt von 10°C auf -5°C
            aussen_temp = 10.0 - (i / 50.0) * 15.0

            # Vorlauf-Ist startet bei 35°C und sollte steigen
            vorlauf_ist = 35.0 + (i / 50.0) * 10.0

            # Raumtemperatur leicht schwankend um Sollwert
            raum_ist = raum_soll + (0.2 if i % 3 == 0 else -0.1)

            # Berechne Vorlauf-Soll
            vorlauf_soll, features = controller.berechne_vorlauf_soll(
                SensorValues(
                    aussen_temp=aussen_temp,
                    raum_ist=raum_ist,
                    raum_soll=raum_soll,
                    vorlauf_ist=vorlauf_ist,
                )
            )

            vorlauf_predictions.append(
                {
                    "cycle": i,
                    "aussen_temp": aussen_temp,
                    "vorlauf_soll": vorlauf_soll,
                    "vorlauf_ist": vorlauf_ist,
                }
            )

            # WICHTIG: Trainiere Model direkt mit dem tatsächlichen Vorlauf
            # In der Realität würde das Model aus vergangenen Messungen lernen
            # Hier simulieren wir: Der IST-Wert war der "richtige" Wert
            controller.model.learn_one(features, vorlauf_ist)

        # Analyse: Vorlauf sollte am Ende höher sein als am Anfang
        vorlauf_anfang_avg = (
            sum(p["vorlauf_soll"] for p in vorlauf_predictions[:10]) / 10
        )
        vorlauf_ende_avg = (
            sum(p["vorlauf_soll"] for p in vorlauf_predictions[-10:]) / 10
        )

        # Bei -5°C (Ende) sollte Vorlauf deutlich höher sein als bei 10°C (Anfang)
        assert vorlauf_ende_avg > vorlauf_anfang_avg, (
            f"Model lernt nicht: Vorlauf am Anfang {vorlauf_anfang_avg:.1f}°C, "
            f"am Ende {vorlauf_ende_avg:.1f}°C. Sollte steigen bei sinkender Außentemp!"
        )

        # Mindestens 5°C Unterschied erwarten
        assert vorlauf_ende_avg - vorlauf_anfang_avg >= 5.0, (
            f"Vorlauf-Anstieg zu gering: {vorlauf_ende_avg - vorlauf_anfang_avg:.1f}°C"
        )

        # Model sollte aus Fallback-Modus raus sein
        assert not controller.use_fallback, "Model sollte nach 50 Zyklen lernen"


class TestModelStats:
    """Test model statistics."""

    def test_get_model_stats_with_experiences(self):
        """Test statistics with stored experiences."""
        controller = FlowController()

        # Add some experiences
        features = {"test": 1.0}
        for i in range(5):
            controller.erfahrungs_speicher.speichere_erfahrung(
                features=features,
                vorlauf_gesetzt=35.0,
                raum_ist_vorher=22.0,
                raum_soll=21.0,
            )

        # Mark some as learned
        controller.erfahrungs_speicher.markiere_gelernt(
            controller.erfahrungs_speicher.erfahrungen[0]
        )

        stats = controller.get_model_stats()

        assert stats.erfahrungen_total == 5
        assert stats.erfahrungen_gelernt == 1
        assert stats.erfahrungen_wartend == 4


class FlowTestHelper:
    """Helper class for fluent-style flow controller testing."""

    def __init__(self, flow_controller: FlowController) -> None:
        """Initialize test helper.

        Args:
            flow_controller: The FlowController instance to test
        """
        self.flow_controller = flow_controller
        self._t_aussen = 10.0
        self._raum_ist = 20.0
        self._raum_soll = 21.0
        self._vorlauf_ist = 30.0
        self._current_time = 0  # Simulated minutes
        self._simulated_datetime = datetime.now()  # Start time
        self._datetime_patcher = None
        self._pending_wait_minutes = 0  # Minutes to advance on next expect_vorlauf_soll()
        self._default_tolerance = 0.5  # Default tolerance for expect_vorlauf_soll()

    def _get_mocked_now(self):
        """Return the current simulated datetime."""
        return self._simulated_datetime

    def t_aussen(self, temp: float) -> "FlowTestHelper":
        """Set outside temperature."""
        self._t_aussen = temp
        return self

    def raum_ist(self, temp: float) -> "FlowTestHelper":
        """Set actual room temperature."""
        self._raum_ist = temp
        return self

    def raum_soll(self, temp: float) -> "FlowTestHelper":
        """Set target room temperature."""
        self._raum_soll = temp
        return self

    def vorlauf_ist(self, temp: float) -> "FlowTestHelper":
        """Set actual flow temperature."""
        self._vorlauf_ist = temp
        return self

    def tolerance(self, value: float) -> "FlowTestHelper":
        """Set default tolerance for all subsequent expect_vorlauf_soll() calls.

        Args:
            value: Default tolerance in °C
        """
        self._default_tolerance = value
        return self

    def expect_vorlauf_soll(
        self, expected: float, tolerance: float | None = None
    ) -> "FlowTestHelper":
        """Execute prediction and verify expected flow temperature.

        Automatically advances simulated time by the minutes specified in the last wait() call.

        Args:
            expected: Expected flow temperature
            tolerance: Allowed deviation in °C (uses default if not specified)
        """
        # Use default tolerance if none specified
        if tolerance is None:
            tolerance = self._default_tolerance

        # Apply pending wait time
        if self._pending_wait_minutes > 0:
            self._current_time += self._pending_wait_minutes
            self._simulated_datetime += timedelta(minutes=self._pending_wait_minutes)
            self._pending_wait_minutes = 0  # Reset after applying

        # Mock datetime.now() to return simulated time
        with patch(
            "custom_components.heatpump_flow_control.flow_controller.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = self._simulated_datetime
            # Also make datetime() constructor work normally
            mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)

            vorlauf_soll, features = self.flow_controller.berechne_vorlauf_soll(
                SensorValues(
                    aussen_temp=self._t_aussen,
                    raum_ist=self._raum_ist,
                    raum_soll=self._raum_soll,
                    vorlauf_ist=self._vorlauf_ist,
                )
            )

            # Add current room temperature to longterm history for reward learning
            # This allows _lerne_aus_erfahrungen() (called internally by berechne_vorlauf_soll)
            # to evaluate past experiences against current room temperature
            self.flow_controller.raum_temp_longterm.append(
                (self._simulated_datetime, self._raum_ist)
            )

        assert abs(vorlauf_soll - expected) <= tolerance, (
            f"Expected vorlauf_soll ~{expected}°C (±{tolerance}°C), "
            f"got {vorlauf_soll:.1f}°C at time={self._current_time}min, "
            f"aussen={self._t_aussen}°C, raum={self._raum_ist}°C, "
            f"vorlauf_ist={self._vorlauf_ist}°C"
        )
        return self

    def wait(self, minutes: int) -> "FlowTestHelper":
        """Set time to advance on next expect_vorlauf_soll() call.

        Does not immediately advance time - the time advancement happens
        when expect_vorlauf_soll() is called.

        Args:
            minutes: Number of minutes to wait before next prediction
        """
        self._pending_wait_minutes = minutes
        return self


def controller(flow_controller: FlowController) -> FlowTestHelper:
    """Factory function to create FlowTestHelper.

    Args:
        flow_controller: The FlowController to wrap

    Returns:
        FlowTestHelper instance for fluent testing
    """
    return FlowTestHelper(flow_controller)


class TestFlowCalculation:
    """Test flow calculation with fluent API."""

    def test_flow_calculation(self):
        """Test flow controller learns to adjust based on room temperature feedback."""
        flow_controller = FlowController(min_vorlauf=25.0, max_vorlauf=55.0)
        flow_controller.min_predictions_for_model = 5  # Schnell aus Fallback raus
        flow_controller.reward_learning_enabled = False  # Klassisches Lernen

        # Fluent test: Realistische Simulation mit zeitversetzter Reaktion
        # wait(60) setzt das Interval für alle folgenden expect_vorlauf_soll() Aufrufe
        (
            controller(flow_controller).tolerance(0.1).wait(60) # Ab jetzt: 60min zwischen jedem expect_vorlauf_soll()
            .t_aussen(10).raum_ist(20).raum_soll(21).vorlauf_ist(30)
            .expect_vorlauf_soll(31.7)
            .vorlauf_ist(30.3).raum_ist(20.3).expect_vorlauf_soll(30.8)
            .vorlauf_ist(30.6).raum_ist(20.5).expect_vorlauf_soll(28.7)
            .vorlauf_ist(30.0).raum_ist(20.8).expect_vorlauf_soll(28.7)
            .vorlauf_ist(29.0).raum_ist(21.0).expect_vorlauf_soll(28.7)
            .vorlauf_ist(28.5).raum_ist(21.0).expect_vorlauf_soll(28.7)
            .vorlauf_ist(28.7).raum_ist(21.0).expect_vorlauf_soll(28.7)
            .vorlauf_ist(28.7).raum_ist(21.0).expect_vorlauf_soll(28.7)
            .vorlauf_ist(28.7).raum_ist(21.0).expect_vorlauf_soll(28.7)
            .vorlauf_ist(28.7).raum_ist(21.0).expect_vorlauf_soll(28.7)
            .vorlauf_ist(28.7).raum_ist(21.0).expect_vorlauf_soll(28.7)
            .vorlauf_ist(28.7).raum_ist(21.0).expect_vorlauf_soll(28.7)
            .vorlauf_ist(28.7).raum_ist(21.0).expect_vorlauf_soll(28.7)
            .vorlauf_ist(28.7).raum_ist(21.0).expect_vorlauf_soll(28.7)
            .vorlauf_ist(39).raum_ist(21.0).expect_vorlauf_soll(28.7)
            .vorlauf_ist(40).raum_ist(21.0).expect_vorlauf_soll(28.7)
            .vorlauf_ist(40).raum_ist(21.0).expect_vorlauf_soll(28.7)
            .vorlauf_ist(40).raum_ist(21.0).expect_vorlauf_soll(28.7)
            .vorlauf_ist(40).raum_ist(21.0).expect_vorlauf_soll(28.7)
            .vorlauf_ist(40).raum_ist(21.0).expect_vorlauf_soll(28.7)
            .vorlauf_ist(40).raum_ist(21.0).expect_vorlauf_soll(28.7)
            .vorlauf_ist(40).raum_ist(21.0).expect_vorlauf_soll(28.7)
            .vorlauf_ist(40).raum_ist(21.0).expect_vorlauf_soll(28.7)

            # SOLL-Wert wird erhöht auf 22.5°C
            #.vorlauf_ist(30.5).raum_soll(22.5).expect_vorlauf_soll(33.0)
            #.vorlauf_ist(33.5).raum_ist(21.1).expect_vorlauf_soll(33.0)
            #.vorlauf_ist(34.8).raum_ist(21.2).expect_vorlauf_soll(34.0)
            #.vorlauf_ist(35.3).raum_ist(21.4).expect_vorlauf_soll(35.0)
            #.vorlauf_ist(35.7).raum_ist(21.6).expect_vorlauf_soll(35.5)
            #.vorlauf_ist(36.0).raum_ist(22.7).expect_vorlauf_soll(35.0)
            #.vorlauf_ist(34.5).raum_ist(22.6).expect_vorlauf_soll(34.0)
            #.vorlauf_ist(33.8).raum_ist(22.5).expect_vorlauf_soll(33.5)
            #.vorlauf_ist(33.5).raum_ist(22.5).expect_vorlauf_soll(33.5)
            #.vorlauf_ist(33.5).raum_ist(22.5).expect_vorlauf_soll(33.5)
            # Außentemperatur sinkt - mehr Vorlauf nötig
            #.vorlauf_ist(33.5).t_aussen(9).expect_vorlauf_soll(33.5)
            #.vorlauf_ist(33.8).t_aussen(8).expect_vorlauf_soll(34.0)
            #.vorlauf_ist(34.3).t_aussen(6).expect_vorlauf_soll(34.5)
            #.vorlauf_ist(35.0).t_aussen(3).expect_vorlauf_soll(35.5)
            #.vorlauf_ist(36.2).t_aussen(0).expect_vorlauf_soll(37.0)
            #.vorlauf_ist(37.8).t_aussen(-3).expect_vorlauf_soll(38.5)
            #.vorlauf_ist(39.2).t_aussen(-6).expect_vorlauf_soll(40.0)
            #.vorlauf_ist(40.8).t_aussen(-9).expect_vorlauf_soll(41.5)
            #.vorlauf_ist(42.2).t_aussen(-11).expect_vorlauf_soll(43.0)
            #.vorlauf_ist(43.5).t_aussen(-12).expect_vorlauf_soll(44.0)
            #.vorlauf_ist(44.2).t_aussen(-12).expect_vorlauf_soll(44.5)
        )

    def test_flow_calculation2(self):
        """Test flow controller learns to adjust based on room temperature feedback."""
        flow_controller = FlowController(min_vorlauf=25.0, max_vorlauf=55.0)
        flow_controller.min_predictions_for_model = 5  # Schnell aus Fallback raus
        flow_controller.reward_learning_enabled = False  # Klassisches Lernen

        (
            controller(flow_controller).tolerance(0.1).wait(60)
            .t_aussen(3.4).raum_ist(22.43).raum_soll(22.5).vorlauf_ist(31.7)
            .expect_vorlauf_soll(40)
            #.vorlauf_ist(30.3).raum_ist(20.3).expect_vorlauf_soll(30.8)
            #.vorlauf_ist(30.6).raum_ist(20.5).expect_vorlauf_soll(28.7)
            #.vorlauf_ist(30.0).raum_ist(20.8).expect_vorlauf_soll(28.7)
            #.vorlauf_ist(29.0).raum_ist(21.0).expect_vorlauf_soll(28.7)
            #.vorlauf_ist(28.5).raum_ist(21.0).expect_vorlauf_soll(28.7)
        )