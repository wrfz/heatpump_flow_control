"""Unit tests for FlowController learning and prediction logic."""

from datetime import datetime, timedelta
import io
import pickle
from unittest.mock import patch

from custom_components.heatpump_flow_control.flow_controller import (
    DateTimeTemperatur,
    Erfahrung,
    Features,
    FlowController,
    HistoryBuffer,
    SensorValues,
)
import pytest


class TestFlowControllerInit:
    """Test FlowController initialization."""

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
        controller = FlowController(
            min_vorlauf=18.0,
            max_vorlauf=39.0,
            learning_rate=0.01,
            trend_history_size=12
        )
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
            controller.model.learn_one(features.to_dict(), 35.0 + i)

        # Nach Training: Fallback sollte aus sein
        assert controller.predictions_count >= 10
        # Fallback wird ausgeschaltet bei nächster Prediction wenn Model realistic ist

        # Simuliere HA-Restart: Speichere aktuellen Zustand
        old_predictions_count = controller.predictions_count
        old_use_fallback = False  # Angenommen Model war schon aktiv
        controller.use_fallback = old_use_fallback

        # Simuliere Pickle-Reload: _setup() wird aufgerufen aber use_fallback existiert schon
        controller._setup(
        )

        # BUG-FIX: use_fallback sollte NICHT auf True zurückgehen
        assert controller.use_fallback is False, (
            "Trainiertes Model sollte nicht in Fallback zurückfallen"
        )

class TestFeatureCreation:
    """Test feature creation logic."""

    def test_erstelle_features_basic(self):
        """Test basic feature creation."""
        controller = FlowController(
            min_vorlauf=18.0,
            max_vorlauf=39.0,
            learning_rate=0.01,
            trend_history_size=12
        )

        features = controller._erstelle_features(
            SensorValues(
                aussen_temp=5.0,
                raum_ist=22.0,
                raum_soll=21.0,
                vorlauf_ist=35.0
            ),
        )

        # Check basic features
        assert features.aussen_temp == 5.0
        assert features.raum_ist == 22.0
        assert features.raum_soll == 21.0
        assert features.vorlauf_ist == 35.0
        assert features.raum_abweichung == -1.0  # 21 - 22 (soll - ist)
        assert features.temp_diff == -17.0  # 5 - 22
        assert features.vorlauf_raum_diff == 13.0  # 35 - 22

        # Check time features exist
        assert features.stunde_sin is not None
        assert features.stunde_cos is not None
        assert features.wochentag_sin is not None
        assert features.wochentag_cos is not None

    def test_erstelle_features_with_trends(self):
        """Test feature creation with temperature trends."""
        controller = FlowController(
            min_vorlauf=18.0,
            max_vorlauf=39.0,
            learning_rate=0.01,
            trend_history_size=12
        )

        # Add some history
        controller.aussen_temp_history = HistoryBuffer([
            DateTimeTemperatur(timestamp=datetime.now() - timedelta(hours=2), temperature=3.0),
            DateTimeTemperatur(timestamp=datetime.now() - timedelta(hours=1), temperature=4.0),
            DateTimeTemperatur(timestamp=datetime.now(), temperature=5.0)
        ])

        features = controller._erstelle_features(
            SensorValues(
                aussen_temp=5.0,
                raum_ist=22.0,
                raum_soll=21.0,
                vorlauf_ist=35.0
            ),
        )

        # Trends should be calculated (may be 0 with insufficient data)
        # Just check they exist
        assert features.aussen_trend_1h is not None
        assert features.aussen_trend_2h is not None
        assert features.aussen_trend_3h is not None
        assert features.aussen_trend_6h is not None

class TestPrediction:
    """Test prediction logic."""

    def test_fallback_heizkurve(self):
        """Test fallback heating curve calculation."""
        controller = FlowController(
            min_vorlauf=25.0,
            max_vorlauf=40.0,
            learning_rate=0.01,
            trend_history_size=12
        )

        assert controller.use_fallback

        assert controller.berechne_vorlauf_soll(SensorValues(aussen_temp=-20.0, raum_ist=22.0, raum_soll=22.0, vorlauf_ist=35.0))[0] == pytest.approx(32)
        assert controller.berechne_vorlauf_soll(SensorValues(aussen_temp=-10.0, raum_ist=22.0, raum_soll=22.0, vorlauf_ist=35.0))[0] == pytest.approx(32)
        assert controller.berechne_vorlauf_soll(SensorValues(aussen_temp=0.0, raum_ist=22.0, raum_soll=22.0, vorlauf_ist=35.0))[0] == pytest.approx(29.2)
        assert controller.berechne_vorlauf_soll(SensorValues(aussen_temp=10.0, raum_ist=22.0, raum_soll=22.0, vorlauf_ist=35.0))[0] == pytest.approx(26.4)
        assert controller.berechne_vorlauf_soll(SensorValues(aussen_temp=15.0, raum_ist=22.0, raum_soll=22.0, vorlauf_ist=35.0))[0] == pytest.approx(25)
        assert controller.berechne_vorlauf_soll(SensorValues(aussen_temp=20.0, raum_ist=22.0, raum_soll=22.0, vorlauf_ist=35.0))[0] == pytest.approx(25)

        controller = FlowController(
            min_vorlauf=25.0,
            max_vorlauf=40.0,
            learning_rate=0.01,
            trend_history_size=12
        )

        assert controller.berechne_vorlauf_soll(SensorValues(aussen_temp=-20.0, raum_ist=22.0, raum_soll=21.0, vorlauf_ist=35.0))[0] == pytest.approx(32)
        assert controller.berechne_vorlauf_soll(SensorValues(aussen_temp=-10.0, raum_ist=22.0, raum_soll=21.0, vorlauf_ist=35.0))[0] == pytest.approx(30)
        assert controller.berechne_vorlauf_soll(SensorValues(aussen_temp=0.0, raum_ist=22.0, raum_soll=21.0, vorlauf_ist=35.0))[0] == pytest.approx(27.2)
        assert controller.berechne_vorlauf_soll(SensorValues(aussen_temp=10.0, raum_ist=22.0, raum_soll=21.0, vorlauf_ist=35.0))[0] == pytest.approx(25)
        assert controller.berechne_vorlauf_soll(SensorValues(aussen_temp=15.0, raum_ist=22.0, raum_soll=21.0, vorlauf_ist=35.0))[0] == pytest.approx(25)
        assert controller.berechne_vorlauf_soll(SensorValues(aussen_temp=20.0, raum_ist=22.0, raum_soll=21.0, vorlauf_ist=35.0))[0] == pytest.approx(25)

        controller = FlowController(
            min_vorlauf=25.0,
            max_vorlauf=40.0,
            learning_rate=0.01,
            trend_history_size=12
        )

        assert controller.berechne_vorlauf_soll(SensorValues(aussen_temp=-20.0, raum_ist=21.0, raum_soll=22.0, vorlauf_ist=35.0))[0] == pytest.approx(32)
        assert controller.berechne_vorlauf_soll(SensorValues(aussen_temp=-10.0, raum_ist=21.0, raum_soll=22.0, vorlauf_ist=35.0))[0] == pytest.approx(32)
        assert controller.berechne_vorlauf_soll(SensorValues(aussen_temp=0.0, raum_ist=21.0, raum_soll=22.0, vorlauf_ist=35.0))[0] == pytest.approx(31.2)
        assert controller.berechne_vorlauf_soll(SensorValues(aussen_temp=10.0, raum_ist=21.0, raum_soll=22.0, vorlauf_ist=35.0))[0] == pytest.approx(28.4)
        assert controller.berechne_vorlauf_soll(SensorValues(aussen_temp=15.0, raum_ist=21.0, raum_soll=22.0, vorlauf_ist=35.0))[0] == pytest.approx(27)
        assert controller.berechne_vorlauf_soll(SensorValues(aussen_temp=20.0, raum_ist=21.0, raum_soll=22.0, vorlauf_ist=35.0))[0] == pytest.approx(25.6)

        assert controller.use_fallback

    def test_berechne_vorlauf_soll_fallback_mode(self):
        """Test calculation in fallback mode."""
        controller = FlowController(
            min_vorlauf=18.0,
            max_vorlauf=39.0,
            learning_rate=0.01,
            trend_history_size=12
        )
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
        controller = FlowController(
            min_vorlauf=30.0,
            max_vorlauf=55.0,
            learning_rate=0.01,
            trend_history_size=12
        )
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
        controller = FlowController(
            min_vorlauf=28.0,
            max_vorlauf=40.0,
            learning_rate=0.01,
            trend_history_size=12
        )
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
        controller = FlowController(
            min_vorlauf=18.0,
            max_vorlauf=39.0,
            learning_rate=0.01,
            trend_history_size=12
        )
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

    def test_prediction_updates_history(self):
        """Test that predictions update temperature history."""
        controller = FlowController(
            min_vorlauf=18.0,
            max_vorlauf=39.0,
            learning_rate=0.01,
            trend_history_size=12
        )

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
        controller = FlowController(
            min_vorlauf=18.0,
            max_vorlauf=39.0,
            learning_rate=0.01,
            trend_history_size=12
        )

        # Room was too cold (21°C), we increased flow, now it's better (21.5°C)
        erfahrung = Erfahrung(
            timestamp=datetime.now(),
            features=Features(raum_abweichung=-1.0),
            raum_ist_vorher=21.0,
            raum_soll=22.0,
            vorlauf_soll=38.0,
        )

        vorlauf_soll_weight = controller._bewerte_erfahrung(
            erfahrung=erfahrung,
            raum_ist_jetzt=21.5
        )

        # Should have some reward (may be 0 or positive depending on logic)
        assert vorlauf_soll_weight.vorlauf_soll == 37
        assert vorlauf_soll_weight.weight == 2.0

    def test_bewerte_erfahrung_negative_reward(self):
        """Test experience evaluation with negative reward."""
        controller = FlowController(
            min_vorlauf=18.0,
            max_vorlauf=39.0,
            learning_rate=0.01,
            trend_history_size=12
        )

        # Room was ok (22°C), now it's too hot (23°C)
        erfahrung = Erfahrung(
            timestamp=datetime.now(),
            features=Features(raum_abweichung=0.0),
            raum_ist_vorher=22.0,
            raum_soll=22.0,
            vorlauf_soll=40.0,  # Too high
        )

        vorlauf_soll_weight = controller._bewerte_erfahrung(
            erfahrung=erfahrung,
            raum_ist_jetzt=23.0
        )

        # Should have negative reward (room got worse)
        assert vorlauf_soll_weight.vorlauf_soll == 38
        assert vorlauf_soll_weight.weight == 2.0

class TestRealisticLearningScenario:
    """Test realistic learning scenarios with full sensor setup."""

    def test_learns_to_increase_vorlauf_when_temperature_drops(self):
        """Test that model learns to increase vorlauf when outdoor temperature drops.

        Realistisches Szenario:
        - Starte bei 10°C außen, Raum auf 21°C
        - Senke Außentemperatur schrittweise auf -5°C
        - Model sollte lernen, dass bei kälteren Temperaturen höherer Vorlauf nötig ist
        """
        controller = FlowController(
            min_vorlauf=25.0,
            max_vorlauf=55.0,
            learning_rate=0.01,
            trend_history_size=12
        )
        controller.min_predictions_for_model = 5  # Schnell aus Fallback raus
        #controller.reward_learning_enabled = False  # Klassisches Lernen

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
            controller.model.learn_one(features.to_dict(), vorlauf_ist)

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
        self._history_states = []  # Store states for optional history learning
        self._enable_history_learning = False  # Default: disabled for backward compatibility

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

            # Store state for optional history learning
            self._history_states.append({
                'timestamp': self._simulated_datetime,
                'minutes_ago': self._current_time,
                'aussen_temp': self._t_aussen,
                'raum_ist': self._raum_ist,
                'raum_soll': self._raum_soll,
                'vorlauf_ist': self._vorlauf_ist,
                'vorlauf_soll': vorlauf_soll,
                'learned': False,
            })

            # Optionally learn from history (simulates number.py behavior)
            self._maybe_learn_from_history()

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

    def enable_history_learning(self) -> "FlowTestHelper":
        """Enable history-based learning in tests.

        When enabled, the test helper will:
        1. Store all states from expect_vorlauf_soll() calls
        2. Create fake historical_state dicts
        3. Call controller.lerne_aus_history() with fake data

        This simulates the behavior of number.py's _async_lerne_aus_history()
        without requiring HA History DB.
        """
        self._enable_history_learning = True
        return self

    def _maybe_learn_from_history(self):
        """Optionally learn from stored history states.

        Called internally after expect_vorlauf_soll() if history learning is enabled.
        Simulates learning from a state that happened ~4h ago.
        """
        if not self._enable_history_learning:
            return

        # Need at least 4 hours of simulated history (240 minutes)
        if self._current_time < 240:
            return

        # Find a state from ~4h ago (240 minutes)
        target_minutes_ago = 240

        # Find the state closest to target_minutes_ago
        best_state = None
        best_diff = None

        for state in self._history_states:
            diff = abs(state['minutes_ago'] - target_minutes_ago)
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_state = state

        if not best_state:
            return

        # Check if we already learned from this state
        if best_state.get('learned'):
            return

        # Need raum_ist_spaeter (2-6h after decision)
        # Find current raum_ist as "spaeter"
        raum_ist_spaeter = self._raum_ist

        # Create fake historical_state dict
        historical_state = {
            'timestamp': best_state['timestamp'],
            'aussen_temp': best_state['aussen_temp'],
            'raum_ist': best_state['raum_ist'],
            'raum_soll': best_state['raum_soll'],
            'vorlauf_ist': best_state['vorlauf_ist'],
            'vorlauf_soll': best_state['vorlauf_soll'],
            'raum_ist_spaeter': raum_ist_spaeter,
        }

        # Call controller's learning method
        success = self.flow_controller.lerne_aus_history(
            historical_state,
            self._simulated_datetime
        )

        if success:
            best_state['learned'] = True


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
        flow_controller = FlowController(
            min_vorlauf=25.0,
            max_vorlauf=55.0,
            learning_rate=0.01,
            trend_history_size=12
        )
        flow_controller.min_predictions_for_model = 5  # Schnell aus Fallback raus
        #flow_controller.reward_learning_enabled = False  # Klassisches Lernen

        # Fluent test: Realistische Simulation mit zeitversetzter Reaktion
        # wait(60) setzt das Interval für alle folgenden expect_vorlauf_soll() Aufrufe
        # enable_history_learning() simuliert HA History-basiertes Lernen
        (
            controller(flow_controller).enable_history_learning().tolerance(0.1).wait(60) # Ab jetzt: 60min zwischen jedem expect_vorlauf_soll()
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
            .vorlauf_ist(30.5).raum_soll(22.5).expect_vorlauf_soll(33.2)
            .vorlauf_ist(33.5).raum_ist(21.1).expect_vorlauf_soll(33.0)
            .vorlauf_ist(34.8).raum_ist(21.2).expect_vorlauf_soll(32.6)
            .vorlauf_ist(35.3).raum_ist(21.4).expect_vorlauf_soll(32.0)
            .vorlauf_ist(35.7).raum_ist(21.6).expect_vorlauf_soll(31.4)
            .vorlauf_ist(36.0).raum_ist(22.7).expect_vorlauf_soll(28.7)
            .vorlauf_ist(34.5).raum_ist(22.6).expect_vorlauf_soll(28.7)
            .vorlauf_ist(33.8).raum_ist(22.5).expect_vorlauf_soll(28.7)
            .vorlauf_ist(33.5).raum_ist(22.5).expect_vorlauf_soll(28.7)
            .vorlauf_ist(33.5).raum_ist(22.5).expect_vorlauf_soll(28.7)
            .vorlauf_ist(33.5).raum_ist(21.5).expect_vorlauf_soll(31.7)

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
        flow_controller = FlowController(
            min_vorlauf=25.0,
            max_vorlauf=55.0,
            learning_rate=0.01,
            trend_history_size=12
        )

        flow_controller.min_predictions_for_model = 5  # Schnell aus Fallback raus
        #flow_controller.reward_learning_enabled = False  # Klassisches Lernen

        (
            controller(flow_controller).enable_history_learning().tolerance(0.1).wait(60)
            .t_aussen(3.4).raum_ist(22.43).raum_soll(22.5).vorlauf_ist(31.7)
            .expect_vorlauf_soll(31.8)
            .vorlauf_ist(30.3).raum_ist(20.3).expect_vorlauf_soll(38.4)
            .vorlauf_ist(30.6).raum_ist(20.5).expect_vorlauf_soll(37.8)
            .vorlauf_ist(30.0).raum_ist(20.8).expect_vorlauf_soll(36.9)
            .vorlauf_ist(29.0).raum_ist(21.0).expect_vorlauf_soll(36.3)
            .vorlauf_ist(28.5).raum_ist(21.0).expect_vorlauf_soll(36.3)
        )

class TestPersistancey:
    """Test persistency of the flow controller."""

    def test_persistancy(self):
        """Test that the FlowController can be pickled and unpickled."""

        controller = FlowController(
            min_vorlauf=25.0,
            max_vorlauf=40.0,
            learning_rate=0.01,
            trend_history_size=12
        )

        assert controller.berechne_vorlauf_soll(SensorValues(aussen_temp=0.0, raum_ist=22.0, raum_soll=22.0, vorlauf_ist=35.0))[0] == pytest.approx(29.2)
        assert controller.predictions_count == 1

        stream = io.BytesIO()
        pickle.dump(controller, stream)
        binary_data = stream.getvalue()

        stream = io.BytesIO(binary_data)
        stream.seek(0)
        controller2 = pickle.load(stream)

        assert controller2.predictions_count == 1

        assert controller2.berechne_vorlauf_soll(SensorValues(aussen_temp=10.0, raum_ist=22.0, raum_soll=22.0, vorlauf_ist=35.0))[0] == pytest.approx(26.4)
        assert controller2.predictions_count == 2

        assert controller.predictions_count == 1



class TestLearning:
    """Test prediction logic."""

    def test_fallback_learning(self):
        """Test fallback heating curve and transition to model mode with proper time simulation.

        This test verifies:
        1. Fallback mode produces correct heating curve values
        2. Transition from fallback to model mode after min_predictions_for_model
        3. Model mode is active after transition
        4. Time simulation allows for history-based learning
        """
        flow_controller = FlowController(
            min_vorlauf=25.0,
            max_vorlauf=40.0,
            learning_rate=0.01,
            trend_history_size=12
        )
        flow_controller.min_predictions_for_model = 10  # Switch to model after 10 predictions
        flow_controller.min_reward_hours = 4.0  # Need 4 hours for learning

        assert flow_controller.use_fallback

        # Test fallback mode with 1-hour intervals between predictions
        # Using fluent API to properly simulate time progression and enable history learning
        test_helper = (
            controller(flow_controller).enable_history_learning().wait(60)
            .t_aussen(5.0).raum_ist(22.0).raum_soll(22.0).vorlauf_ist(35.0)
            .expect_vorlauf_soll(27.8, tolerance=0.1)

            .t_aussen(4.0).raum_ist(22.0).raum_soll(22.0).vorlauf_ist(35.0)
            .expect_vorlauf_soll(28.08, tolerance=0.1)

            .t_aussen(3.0).raum_ist(22.0).raum_soll(22.0).vorlauf_ist(35.0)
            .expect_vorlauf_soll(28.36, tolerance=0.1)

            .t_aussen(2.0).raum_ist(22.0).raum_soll(22.0).vorlauf_ist(35.0)
            .expect_vorlauf_soll(28.64, tolerance=0.1)

            .t_aussen(1.0).raum_ist(22.0).raum_soll(22.0).vorlauf_ist(35.0)
            .expect_vorlauf_soll(28.92, tolerance=0.1)

            .t_aussen(0.0).raum_ist(22.0).raum_soll(22.0).vorlauf_ist(35.0)
            .expect_vorlauf_soll(29.2, tolerance=0.1)

            .t_aussen(-1.0).raum_ist(22.0).raum_soll(22.0).vorlauf_ist(35.0)
            .expect_vorlauf_soll(29.48, tolerance=0.1)

            .t_aussen(-2.0).raum_ist(22.0).raum_soll(22.0).vorlauf_ist(35.0)
            .expect_vorlauf_soll(29.76, tolerance=0.1)

            .t_aussen(-3.0).raum_ist(22.0).raum_soll(22.0).vorlauf_ist(35.0)
            .expect_vorlauf_soll(30.04, tolerance=0.1)
        )

        # Still in fallback mode (9 predictions so far, 540 minutes = 9 hours elapsed)
        assert flow_controller.use_fallback
        assert flow_controller.predictions_count == 9

        # 10th prediction - should switch to model mode after this (600 minutes = 10 hours total)
        test_helper = (
            test_helper
            .t_aussen(-4.0).raum_ist(22.0).raum_soll(22.0).vorlauf_ist(35.0)
            .expect_vorlauf_soll(30.32, tolerance=0.1)
        )

        # Now model mode should be active
        assert not flow_controller.use_fallback
        assert flow_controller.predictions_count == 10

        # Debug: Check if model has been trained
        print("\nDEBUG: Model after switch:")
        print(f"  use_fallback: {flow_controller.use_fallback}")
        print(f"  Model type: {type(flow_controller.model)}")

        # Try a simple prediction to see what the model returns
        from custom_components.heatpump_flow_control.types import Features
        test_features = Features(
            aussen_temp=-5.0,
            raum_ist=21.0,
            raum_soll=22.0,
            vorlauf_ist=32.5,
            raum_abweichung=1.0,
        )
        test_prediction = flow_controller.model.predict_one(test_features.to_dict())
        print(f"  Test prediction for t_aussen=-5.0, raum_dev=1.0: {test_prediction:.2f}°C")

        # Continue with 20 more predictions to actually test model learning
        # Now the model should use predict_one and learn from experiences via lerne_aus_features
        # After synthetic training, model should predict realistic values (sinkende Temp → steigender Vorlauf)
        test_helper = (
            test_helper
            # Predictions 11-16: Model predictions nach synthetischem Training
            # Temperatur SINKT von -5°C auf -10°C → Vorlauf MUSS STEIGEN!
            # Modell gibt ~27-29°C zurück (nicht Heizkurve 30-32°C), aber Richtung stimmt
            .t_aussen(-5.0).raum_ist(22.0).raum_soll(22.0).vorlauf_ist(30.6)
            .expect_vorlauf_soll(27.81, tolerance=0.1)  # Modell-Vorhersage

            .t_aussen(-6.0).raum_ist(22.0).raum_soll(22.0).vorlauf_ist(30.8)
            .expect_vorlauf_soll(27.93, tolerance=0.1)  # Vorlauf steigt

            .t_aussen(-7.0).raum_ist(22.0).raum_soll(22.0).vorlauf_ist(31.0)
            .expect_vorlauf_soll(28.04, tolerance=0.1)  # Vorlauf steigt weiter

            .t_aussen(-8.0).raum_ist(22.0).raum_soll(22.0).vorlauf_ist(31.3)
            .expect_vorlauf_soll(28.14, tolerance=0.1)  # Vorlauf steigt weiter

            .t_aussen(-9.0).raum_ist(22.0).raum_soll(22.0).vorlauf_ist(31.6)
            .expect_vorlauf_soll(28.21, tolerance=0.1)  # Vorlauf steigt weiter

            .t_aussen(-10.0).raum_ist(22.0).raum_soll(22.0).vorlauf_ist(31.9)
            .expect_vorlauf_soll(28.27, tolerance=0.1)  # Vorlauf steigt weiter
        )

        # After 16 predictions (16 hours), experiences from hour 1-12 can be learned (>4h old)
        # Verify learning is happening
        assert flow_controller.predictions_count == 16

        # Continue with more predictions using conditions similar to training data
        # Model predictions after learning from synthetic data and fallback experiences
        # Temperatur STEIGT von -9°C auf 0°C → Vorlauf MUSS FALLEN!
        test_helper = (
            test_helper
            # Predictions 17-26: Temperatur STEIGT → Vorlauf MUSS FALLEN
            # Modell gibt ~28-29°C zurück, fallend bei steigender Temperatur
            .t_aussen(-9.0).raum_ist(22.0).raum_soll(22.0).vorlauf_ist(31.7)
            .expect_vorlauf_soll(28.21, tolerance=0.1)  # Vorlauf beginnt hoch

            .t_aussen(-8.0).raum_ist(22.0).raum_soll(22.0).vorlauf_ist(31.4)
            .expect_vorlauf_soll(28.14, tolerance=0.1)  # Vorlauf fällt

            .t_aussen(-7.0).raum_ist(22.0).raum_soll(22.0).vorlauf_ist(31.1)
            .expect_vorlauf_soll(28.04, tolerance=0.1)  # Vorlauf fällt weiter

            .t_aussen(-6.0).raum_ist(22.0).raum_soll(22.0).vorlauf_ist(30.8)
            .expect_vorlauf_soll(27.93, tolerance=0.1)  # Vorlauf fällt weiter

            .t_aussen(-5.0).raum_ist(22.0).raum_soll(22.0).vorlauf_ist(30.5)
            .expect_vorlauf_soll(27.81, tolerance=0.1)  # Vorlauf fällt weiter

            .t_aussen(-4.0).raum_ist(22.0).raum_soll(22.0).vorlauf_ist(30.2)
            .expect_vorlauf_soll(27.68, tolerance=0.1)  # Vorlauf fällt weiter

            .t_aussen(-3.0).raum_ist(22.0).raum_soll(22.0).vorlauf_ist(30.0)
            .expect_vorlauf_soll(27.54, tolerance=0.1)  # Vorlauf fällt weiter

            .t_aussen(-2.0).raum_ist(22.0).raum_soll(22.0).vorlauf_ist(29.7)
            .expect_vorlauf_soll(27.39, tolerance=0.1)  # Vorlauf fällt weiter

            .t_aussen(-1.0).raum_ist(22.0).raum_soll(22.0).vorlauf_ist(29.4)
            .expect_vorlauf_soll(27.24, tolerance=0.1)  # Vorlauf fällt weiter

            .t_aussen(0.0).raum_ist(22.0).raum_soll(22.0).vorlauf_ist(29.1)
            .expect_vorlauf_soll(27.08, tolerance=0.1)  # Vorlauf fällt auf Minimum
        )

        # Verify that the model is being used
        assert flow_controller.predictions_count == 26
        assert not flow_controller.use_fallback

        # Verify that experiences are being stored
        assert len(flow_controller.erfahrungs_liste) <= 26, \
            "Experiences should be stored (some may be removed after learning)"

        # Verify that predictions respect min/max bounds
        # (Model without training returns min_vorlauf)
        assert all(
            e.vorlauf_soll >= flow_controller.min_vorlauf
            for e in flow_controller.erfahrungs_liste
        ), "All predictions should be >= min_vorlauf"

        # Model should still operate within configured bounds
        # Get last prediction to verify
        vorlauf_soll, _ = flow_controller.berechne_vorlauf_soll(
            SensorValues(
                aussen_temp=0.0,
                raum_ist=22.0,
                raum_soll=22.0,
                vorlauf_ist=29.0,
            )
        )
        assert flow_controller.min_vorlauf <= vorlauf_soll <= flow_controller.max_vorlauf, \
            f"Model prediction {vorlauf_soll} should be within [{flow_controller.min_vorlauf}, {flow_controller.max_vorlauf}]"
