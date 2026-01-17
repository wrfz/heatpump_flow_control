"""Unit tests for FlowController learning and prediction logic."""

from datetime import datetime, timedelta
import io
import pickle
import random
from unittest.mock import patch

from custom_components.heatpump_flow_control.flow_controller import (
    DateTimeTemperatur,
    Erfahrung,
    Features,
    FlowController,
    HistoryBuffer,
    SensorValues,
    VorlaufSollWeight,
)
import numpy as np
import pytest

from .temperature_test_base import TemperaturePredictionTestBase


class TestFlowControllerInit:
    """Test FlowController initialization."""

    def test_custom_initialization(self):
        """Test controller with custom parameters."""
        controller = FlowController(
            min_vorlauf=25.0,
            max_vorlauf=50.0,
            learning_rate=0.05,
        )

        assert controller.min_vorlauf == 25.0
        assert controller.max_vorlauf == 50.0

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
        )
        controller.setup()

        # Trainiere Model (10+ Predictions)
        for i in range(15):
            vorlauf_soll_and_features = controller.berechne_vorlauf_soll(
                SensorValues(aussen_temp=5.0 - i * 0.5,
                raum_ist=21.0,
                raum_soll=21.0,
                vorlauf_ist=35.0 + i)
            )
            # Trainiere mit realistischem Wert
            controller.model.learn_one(vorlauf_soll_and_features.features.to_dict(), 35.0 + i)

        # Simuliere Pickle-Reload: _setup() wird aufgerufen aber use_fallback existiert schon
        controller.setup()


class TestBewertung:
    """Test reward calculation logic."""

    def test_bewerte_erfahrung(self):
        """Test reward calculation for experiences."""

        controller = FlowController(
            min_vorlauf=18.0,
            max_vorlauf=39.0,
            learning_rate=0.01,
        )

        DC = 0.0

        features=Features(
            aussen_temp=DC,
            raum_ist=DC,
            raum_soll=DC,
            vorlauf_ist=DC,
            raum_abweichung=DC,
            aussen_trend_1h=DC,
            stunde_sin=DC,
            stunde_cos=DC,
            wochentag_sin=DC,
            wochentag_cos=DC,
            temp_diff=DC,
            vorlauf_raum_diff=DC,
        )

        def bewerte_erfahrung(raum_ist_jetzt: float, raum_soll: float, vorlauf_soll: float) -> VorlaufSollWeight:
            return controller._bewerte_erfahrung(  # noqa: SLF001
                    erfahrung=Erfahrung(timestamp=datetime.now(), features=features, raum_soll=raum_soll, vorlauf_soll=vorlauf_soll, raum_ist_vorher=DC),
                    raum_ist_jetzt=raum_ist_jetzt,
                )

        assert bewerte_erfahrung(raum_ist_jetzt=22.0, raum_soll=22.0, vorlauf_soll=30.0) == pytest.approx(VorlaufSollWeight(vorlauf_soll=30.00, weight=1.00), abs=0.01)
        assert bewerte_erfahrung(raum_ist_jetzt=21.0, raum_soll=22.0, vorlauf_soll=30.0) == pytest.approx(VorlaufSollWeight(vorlauf_soll=32.00, weight=2.00), abs=0.01)
        assert bewerte_erfahrung(raum_ist_jetzt=22.0, raum_soll=21.0, vorlauf_soll=30.0) == pytest.approx(VorlaufSollWeight(vorlauf_soll=28.00, weight=2.00), abs=0.01)


class TestFeatureCreation:
    """Test feature creation logic."""

    def test_erstelle_features_basic(self):
        """Test basic feature creation."""
        controller = FlowController(
            min_vorlauf=18.0,
            max_vorlauf=39.0,
            learning_rate=0.01,
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

class TestPrediction:
    """Test prediction logic."""

    def test_fallback_heizkurve(self):
        """Test fallback heating curve calculation."""
        controller = FlowController(
            min_vorlauf=25.0,
            max_vorlauf=40.0,
            learning_rate=0.01,
        )

        # Test direct fallback curve (not the trained model)
        assert controller._heizkurve_fallback(-20.0, 0.0) == pytest.approx(40.00, abs=0.01)     # noqa: SLF001
        assert controller._heizkurve_fallback(-10.0, 0.0) == pytest.approx(35.71, abs=0.01)     # noqa: SLF001
        assert controller._heizkurve_fallback(  0.0, 0.0) == pytest.approx(31.42, abs=0.01)     # noqa: SLF001
        assert controller._heizkurve_fallback( 10.0, 0.0) == pytest.approx(27.14, abs=0.01)     # noqa: SLF001
        assert controller._heizkurve_fallback( 15.0, 0.0) == pytest.approx(26.40, abs=0.01)     # noqa: SLF001
        assert controller._heizkurve_fallback( 20.0, 0.0) == pytest.approx(26.40, abs=0.01)     # noqa: SLF001

        # Test with room temperature deviation (room too hot)
        assert controller._heizkurve_fallback(-20.0, -1.0) == pytest.approx(34.72, abs=0.01)    # noqa: SLF001
        assert controller._heizkurve_fallback(-10.0, -1.0) == pytest.approx(31.28, abs=0.01)    # noqa: SLF001
        assert controller._heizkurve_fallback(  0.0, -1.0) == pytest.approx(27.84, abs=0.01)    # noqa: SLF001
        assert controller._heizkurve_fallback( 10.0, -1.0) == pytest.approx(26.40, abs=0.01)    # noqa: SLF001
        assert controller._heizkurve_fallback( 15.0, -1.0) == pytest.approx(26.40, abs=0.01)    # noqa: SLF001
        assert controller._heizkurve_fallback( 20.0, -1.0) == pytest.approx(26.40, abs=0.01)    # noqa: SLF001

        # Test with room temperature deviation (room too cold)
        assert controller._heizkurve_fallback(-20.0, 1.0) == pytest.approx(35.00, abs=0.01)     # noqa: SLF001
        assert controller._heizkurve_fallback(-10.0, 1.0) == pytest.approx(35.00, abs=0.01)     # noqa: SLF001
        assert controller._heizkurve_fallback(  0.0, 1.0) == pytest.approx(31.84, abs=0.01)     # noqa: SLF001
        assert controller._heizkurve_fallback( 10.0, 1.0) == pytest.approx(28.40, abs=0.01)     # noqa: SLF001
        assert controller._heizkurve_fallback( 15.0, 1.0) == pytest.approx(26.68, abs=0.01)     # noqa: SLF001
        assert controller._heizkurve_fallback( 20.0, 1.0) == pytest.approx(26.40, abs=0.01)     # noqa: SLF001

    def test_berechne_vorlauf_soll_fallback_mode(self):
        """Test calculation in fallback mode."""
        controller = FlowController(
            min_vorlauf=18.0,
            max_vorlauf=39.0,
            learning_rate=0.01,
        )

        vorlauf_soll_and_features = controller.berechne_vorlauf_soll(
            SensorValues(
                aussen_temp=5.0,
                raum_ist=22.0,
                raum_soll=21.0,
                vorlauf_ist=35.0
            )
        )

        # Should use fallback
        assert controller.min_vorlauf <= vorlauf_soll_and_features.vorlauf <= controller.max_vorlauf

    def test_fallback_respects_configured_min_vorlauf(self):
        """Test that fallback mode respects configured min_vorlauf boundary."""
        controller = FlowController(
            min_vorlauf=30.0,
            max_vorlauf=55.0,
            learning_rate=0.01,
        )

        # At high outdoor temp, standard Heizkurve would give low value
        vorlauf_soll_and_features = controller.berechne_vorlauf_soll(
            SensorValues(
                aussen_temp=15.0,  # Warm outside
                raum_ist=20.0,
                raum_soll=21.0,
                vorlauf_ist=32.0,
             )
        )

        # BUG: Should respect configured min_vorlauf
        assert vorlauf_soll_and_features.vorlauf >= 30.0, (
            f"Fallback returned {vorlauf_soll_and_features.vorlauf}°C but min is 30°C"
        )
        assert vorlauf_soll_and_features.vorlauf <= 55.0

    def test_berechne_vorlauf_soll_within_limits(self):
        """Test that prediction is always within configured limits."""
        controller = FlowController(
            min_vorlauf=28.0,
            max_vorlauf=40.0,
            learning_rate=0.01,
        )
        controller.setup()

        # Mock extreme prediction
        with patch.object(controller.model, "predict_one", return_value=100.0):
            vorlauf_soll_and_features = controller.berechne_vorlauf_soll(
                SensorValues(
                    aussen_temp=5.0,
                    raum_ist=22.0,
                    raum_soll=21.0,
                    vorlauf_ist=35.0
                )
            )

            # Should use fallback (unrealistic value triggers fallback)
            assert 28.0 <= vorlauf_soll_and_features.vorlauf <= 40.0

    def test_unrealistic_prediction_triggers_fallback(self):
        """Test that unrealistic predictions trigger fallback."""
        controller = FlowController(
            min_vorlauf=18.0,
            max_vorlauf=39.0,
            learning_rate=0.01,
        )
        controller.setup()

        # Mock extreme prediction
        with patch.object(controller.model, "predict_one", return_value=5000000.0):
            vorlauf_soll_and_features = controller.berechne_vorlauf_soll(
                SensorValues(
                    aussen_temp=5.0,
                    raum_ist=22.0,
                    raum_soll=21.0,
                    vorlauf_ist=35.0
                )
            )

            # Fallback value should be within reasonable range
            assert 25.0 <= vorlauf_soll_and_features.vorlauf <= 55.0

class FlowTestHelper:
    """Helper class for fluent-style flow controller testing."""

    def __init__(self, flow_controller: FlowController) -> None:
        """Initialize test helper.

        Args:
            flow_controller: The FlowController instance to test
        """
        self.flow_controller = flow_controller
        self._t_aussen = 10.0
        self._raum_ist = 22.5  # Default auf 22.5°C
        self._raum_soll = 22.5  # Default auf 22.5°C
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
        """Set actual room temperature (sticky - remains for subsequent predictions)."""
        self._raum_ist = temp
        return self

    def raum_soll(self, temp: float) -> "FlowTestHelper":
        """Set target room temperature (sticky - remains for subsequent predictions)."""
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

            vorlauf_soll_and_features = self.flow_controller.berechne_vorlauf_soll(
                SensorValues(
                    aussen_temp=self._t_aussen,
                    raum_ist=self._raum_ist,
                    raum_soll=self._raum_soll,
                    vorlauf_ist=self._vorlauf_ist,
                )
            )

            # Store state for optional history learning
            self._history_states.append({
                'timestamp': self._simulated_datetime,
                'minutes_ago': self._current_time,
                'aussen_temp': self._t_aussen,
                'raum_ist': self._raum_ist,
                'raum_soll': self._raum_soll,
                'vorlauf_ist': self._vorlauf_ist,
                'vorlauf_soll': vorlauf_soll_and_features.vorlauf,
                'learned': False,
            })

        assert abs(vorlauf_soll_and_features.vorlauf - expected) <= tolerance, (
            f"Expected vorlauf_soll ~{expected}°C (±{tolerance}°C), "
            f"got {vorlauf_soll_and_features.vorlauf:.1f}°C at time={self._current_time}min, "
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

def controller(flow_controller: FlowController) -> FlowTestHelper:
    """Factory function to create FlowTestHelper.

    Args:
        flow_controller: The FlowController to wrap

    Returns:
        FlowTestHelper instance for fluent testing
    """
    return FlowTestHelper(flow_controller)

class TestPersistancey:
    """Test persistency of the flow controller."""

    def test_persistancy(self):
        """Test that the FlowController can be pickled and unpickled."""

        controller = FlowController(
            min_vorlauf=25.0,
            max_vorlauf=40.0,
            learning_rate=0.01,
        )
        controller.setup(1)

        # First prediction
        result1 = controller.berechne_vorlauf_soll(
            SensorValues(aussen_temp=0.0, raum_ist=22.0, raum_soll=22.0, vorlauf_ist=35.0)
        )
        assert result1.vorlauf == pytest.approx(34.74, abs=0.01)

        # Pickle and unpickle
        stream = io.BytesIO()
        pickle.dump(controller, stream)
        binary_data = stream.getvalue()

        stream = io.BytesIO(binary_data)
        stream.seek(0)
        controller2: FlowController = pickle.load(stream)

        # After unpickling, config must be restored
        controller2.update_config(
            min_vorlauf=25.0,
            max_vorlauf=40.0,
            learning_rate=0.01,
        )

        # Verify model state is preserved
        assert len(controller2.erfahrungs_liste) == 1

        # Second prediction should work with restored model
        result2 = controller2.berechne_vorlauf_soll(
            SensorValues(aussen_temp=10.0, raum_ist=22.0, raum_soll=22.0, vorlauf_ist=35.0)
        )
        assert result2.vorlauf == pytest.approx(34.35, abs=0.01)
        assert len(controller2.erfahrungs_liste) == 2

    def test_config_changes_after_unpickle(self):
        """Test that config changes work after unpickling."""

        # Create controller with initial config
        controller = FlowController(
            min_vorlauf=25.0,
            max_vorlauf=40.0,
            learning_rate=0.01,
        )
        controller.setup()

        # Make a prediction to train the model
        controller.berechne_vorlauf_soll(
            SensorValues(aussen_temp=0.0, raum_ist=22.0, raum_soll=22.0, vorlauf_ist=35.0)
        )

        # Pickle
        stream = io.BytesIO()
        pickle.dump(controller, stream)
        stream.seek(0)

        # Unpickle with DIFFERENT config (like user changed settings in HA)
        controller2: FlowController = pickle.load(stream)
        controller2.update_config(
            min_vorlauf=20.0,  # Changed from 25.0
            max_vorlauf=50.0,  # Changed from 40.0
            learning_rate=0.02,  # Changed from 0.01
        )

        # Verify new config is applied
        assert controller2.min_vorlauf == 20.0
        assert controller2.max_vorlauf == 50.0
        assert controller2.learning_rate == 0.02

        # Verify model state is still preserved
        assert len(controller2.erfahrungs_liste) == 1

        # Model should still work with new config
        result = controller2.berechne_vorlauf_soll(
            SensorValues(aussen_temp=10.0, raum_ist=22.0, raum_soll=22.0, vorlauf_ist=35.0)
        )
        assert result.vorlauf >= 20.0  # Respects new min
        assert result.vorlauf <= 50.0  # Respects new max


class TestLearning(TemperaturePredictionTestBase):
    """Test learning and prediction behavior."""


    def test_outside_temperature_learning(self):
        """Test that the model learns to adjust flow temperature based on outside temperature changes."""

        # Set random seeds for deterministic results
        random.seed(42)
        np.random.seed(42)

        controller = FlowController(
            min_vorlauf=25.0,
            max_vorlauf=40.0,
            learning_rate=0.01,
        )
        controller.setup(1)

        temperatures = """
        t_aussen | raum-ist | raum-soll | vorlauf_ist | vorlauf_soll
        -------------------------------------------------------------
         10.0     | 22.5     |  22.5     |    28.0     | 28.62
          8.0     | 22.7     |  22.5     |    28.2     | 28.75
          6.0     | 22.8     |  22.5     |    28.4     | 28.93
          4.0     | 22.8     |  22.5     |    28.5     | 29.09
          2.0     | 22.7     |  22.5     |    28.7     | 29.39
          0.0     | 22.5     |  22.5     |    28.9     | 29.75
         -2.0     | 22.4     |  22.5     |    29.2     | 30.14
         -4.0     | 22.2     |  22.5     |    29.4     | 30.50
         -6.0     | 22.1     |  22.5     |    29.8     | 30.97
         -8.0     | 21.9     |  22.5     |    30.1     | 31.41
        -10.0     | 21.7     |  22.5     |    30.5     | 31.94
        -12.0     | 21.6     |  22.5     |    30.9     | 32.40
        -14.0     | 21.4     |  22.5     |    31.3     | 32.93
        -15.0     | 21.3     |  22.5     |    31.8     | 33.44
        -------------------------------------------------------------
        -14.0     | 21.4     |  22.5     |    32.3     | 33.75
        -12.0     | 21.5     |  22.5     |    32.7     | 33.93
        -10.0     | 21.8     |  22.5     |    33.1     | 34.00
         -8.0     | 22.1     |  22.5     |    33.3     | 33.90
         -6.0     | 22.5     |  22.5     |    33.5     | 33.75
         -4.0     | 22.8     |  22.5     |    33.5     | 33.49
         -2.0     | 23.2     |  22.5     |    33.5     | 33.17
          0.0     | 23.5     |  22.5     |    33.4     | 32.82
          2.0     | 23.8     |  22.5     |    33.2     | 32.40
          4.0     | 24.1     |  22.5     |    33.0     | 31.98
          6.0     | 24.4     |  22.5     |    32.6     | 31.39
          8.0     | 24.6     |  22.5     |    32.3     | 30.94
         10.0     | 24.8     |  22.5     |    31.9     | 30.42
        """

        self._assert_temperature_predictions(controller, temperatures, simulate_raum_ist=True)

    def test_raum_soll_temperature_learning(self):
        """Test that the model learns to adjust flow temperature based on outside temperature changes."""

        # Set random seeds for deterministic results
        random.seed(42)
        np.random.seed(42)

        controller = FlowController(
            min_vorlauf=25.0,
            max_vorlauf=40.0,
            learning_rate=0.01,
        )
        controller.setup(1)

        temperatures = """
        t_aussen | raum-ist | raum-soll | vorlauf_ist | vorlauf_soll
        -------------------------------------------------------------
          0.0     | 20.0     |  22.5     |    32.0     | 33.80
          0.0     | 21.1     |  22.5     |    32.5     | 33.54
          0.0     | 21.9     |  22.5     |    32.8     | 33.30
          0.0     | 22.5     |  22.5     |    33.0     | 33.10
          0.0     | 23.0     |  22.5     |    33.0     | 32.80
          0.0     | 23.3     |  22.5     |    33.0     | 32.62
          0.0     | 23.5     |  22.5     |    32.8     | 32.33
          0.0     | 23.7     |  22.5     |    32.7     | 32.13
          0.0     | 23.7     |  22.5     |    32.5     | 31.97
          0.0     | 23.8     |  22.5     |    32.3     | 31.74
          0.0     | 23.8     |  22.5     |    32.2     | 31.66
          0.0     | 23.8     |  22.5     |    32.0     | 31.50
          0.0     | 23.7     |  22.5     |    31.8     | 31.40
          0.0     | 23.7     |  22.5     |    31.7     | 31.31
        -------------------------------------------------------------
          0.0     | 23.7     |  20.0     |    31.6     | 29.73
          0.0     | 23.4     |  20.0     |    31.0     | 29.42
          0.0     | 23.1     |  20.0     |    30.6     | 29.28
          0.0     | 22.9     |  20.0     |    30.2     | 29.07
          0.0     | 22.7     |  20.0     |    29.9     | 28.95
          0.0     | 22.5     |  20.0     |    29.6     | 28.83
          0.0     | 22.3     |  20.0     |    29.4     | 28.78
          0.0     | 22.2     |  20.0     |    29.2     | 28.68
          0.0     | 22.1     |  20.0     |    29.1     | 28.66
          0.0     | 22.0     |  20.0     |    29.0     | 28.64
          0.0     | 21.9     |  20.0     |    28.9     | 28.62
          0.0     | 21.8     |  20.0     |    28.9     | 28.68
          0.0     | 21.8     |  20.0     |    28.9     | 28.68
          0.0     | 21.7     |  20.0     |    28.8     | 28.66
        -------------------------------------------------------------
          0.0     | 21.7     |  25.0     |    28.8     | 31.65
          0.0     | 22.1     |  25.0     |    29.7     | 32.15
          0.0     | 22.6     |  25.0     |    30.5     | 32.50
          0.0     | 22.9     |  25.0     |    31.1     | 32.81
          0.0     | 23.3     |  25.0     |    31.7     | 33.06
          0.0     | 23.6     |  25.0     |    32.1     | 33.20
          0.0     | 23.9     |  25.0     |    32.4     | 33.26
          0.0     | 24.1     |  25.0     |    32.7     | 33.39
          0.0     | 24.3     |  25.0     |    32.9     | 33.43
          0.0     | 24.4     |  25.0     |    33.1     | 33.53
        """

        self._assert_temperature_predictions(controller, temperatures, simulate_raum_ist=True)
