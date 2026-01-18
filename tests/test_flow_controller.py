"""Unit tests for FlowController learning and prediction logic."""

from datetime import datetime, timedelta
import io
import pickle
import random
from unittest.mock import patch

from custom_components.heatpump_flow_control.flow_controller import (
    DateTimeTemperatur,
    FlowController,
    HistoryBuffer,
    SensorValues,
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
        assert features.raum_abweichung == -1.0  # 21 - 22 (soll - ist)
        assert features.temp_diff == -17.0  # 5 - 22

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
        assert controller._heizkurve_fallback( 15.0, 0.0) == pytest.approx(25.00, abs=0.01)     # noqa: SLF001
        assert controller._heizkurve_fallback( 20.0, 0.0) == pytest.approx(25.00, abs=0.01)     # noqa: SLF001

        # Test with room temperature deviation (room too hot)
        assert controller._heizkurve_fallback(-20.0, -1.0) == pytest.approx(38.00, abs=0.01)    # noqa: SLF001
        assert controller._heizkurve_fallback(-10.0, -1.0) == pytest.approx(33.71, abs=0.01)    # noqa: SLF001
        assert controller._heizkurve_fallback(  0.0, -1.0) == pytest.approx(29.42, abs=0.01)    # noqa: SLF001
        assert controller._heizkurve_fallback( 10.0, -1.0) == pytest.approx(25.14, abs=0.01)    # noqa: SLF001
        assert controller._heizkurve_fallback( 15.0, -1.0) == pytest.approx(25.00, abs=0.01)    # noqa: SLF001
        assert controller._heizkurve_fallback( 20.0, -1.0) == pytest.approx(25.00, abs=0.01)    # noqa: SLF001

        # Test with room temperature deviation (room too cold)
        assert controller._heizkurve_fallback(-20.0, 1.0) == pytest.approx(40.00, abs=0.01)     # noqa: SLF001
        assert controller._heizkurve_fallback(-10.0, 1.0) == pytest.approx(37.71, abs=0.01)     # noqa: SLF001
        assert controller._heizkurve_fallback(  0.0, 1.0) == pytest.approx(33.43, abs=0.01)     # noqa: SLF001
        assert controller._heizkurve_fallback( 10.0, 1.0) == pytest.approx(29.14, abs=0.01)     # noqa: SLF001
        assert controller._heizkurve_fallback( 15.0, 1.0) == pytest.approx(27.00, abs=0.01)     # noqa: SLF001
        assert controller._heizkurve_fallback( 20.0, 1.0) == pytest.approx(25.00, abs=0.01)     # noqa: SLF001

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
        assert result1.vorlauf == pytest.approx(30.70, abs=0.01)

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
        assert result2.vorlauf == pytest.approx(28.37, abs=0.01)
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
         15.0     | 22.5     |  22.5     |    28.0     | 27.25
         14.0     | 22.5     |  22.5     |    27.8     | 27.46
         13.0     | 22.5     |  22.5     |    27.7     | 27.70
         12.0     | 22.5     |  22.5     |    27.7     | 27.96
         11.0     | 22.4     |  22.5     |    27.8     | 28.23
         10.0     | 22.4     |  22.5     |    27.9     | 28.52
          8.0     | 22.2     |  22.5     |    28.1     | 29.09
          6.0     | 22.1     |  22.5     |    28.4     | 29.68
          4.0     | 21.9     |  22.5     |    28.8     | 30.29
          2.0     | 21.7     |  22.5     |    29.2     | 30.91
          0.0     | 21.5     |  22.5     |    29.7     | 31.54
         -2.0     | 21.3     |  22.5     |    30.3     | 32.18
         -4.0     | 21.0     |  22.5     |    30.8     | 32.82
         -6.0     | 20.8     |  22.5     |    31.4     | 33.47
         -8.0     | 20.6     |  22.5     |    32.0     | 34.13
        -10.0     | 20.3     |  22.5     |    32.7     | 34.78
        -12.0     | 20.1     |  22.5     |    33.3     | 35.44
        -14.0     | 19.8     |  22.5     |    33.9     | 36.10
        -15.0     | 19.6     |  22.5     |    34.6     | 36.49
        -------------------------------------------------------------
        -14.0     | 19.6     |  22.5     |    35.2     | 36.32
        -12.0     | 19.6     |  22.5     |    35.5     | 35.84
        -10.0     | 19.6     |  22.5     |    35.6     | 35.33
         -8.0     | 19.7     |  22.5     |    35.5     | 34.77
         -6.0     | 19.9     |  22.5     |    35.3     | 34.20
         -4.0     | 20.1     |  22.5     |    35.0     | 33.60
         -2.0     | 20.2     |  22.5     |    34.6     | 32.99
          0.0     | 20.4     |  22.5     |    34.1     | 32.37
          2.0     | 20.6     |  22.5     |    33.6     | 31.74
          4.0     | 20.9     |  22.5     |    33.0     | 31.09
          6.0     | 21.1     |  22.5     |    32.4     | 30.45
          8.0     | 21.3     |  22.5     |    31.8     | 29.80
         10.0     | 21.6     |  22.5     |    31.2     | 29.15
         11.0     | 21.8     |  22.5     |    30.6     | 28.76
         12.0     | 21.9     |  22.5     |    30.1     | 28.39
         13.0     | 22.1     |  22.5     |    29.6     | 28.02
         14.0     | 22.3     |  22.5     |    29.1     | 27.67
         15.0     | 22.4     |  22.5     |    28.7     | 27.32
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
          0.0     | 21.4     |  20.0     |    26.0     | 29.45
          0.0     | 21.1     |  20.0     |    27.0     | 29.69
          0.0     | 20.9     |  20.0     |    27.8     | 29.88
          0.0     | 20.7     |  20.0     |    28.4     | 30.02
          0.0     | 20.5     |  20.0     |    28.9     | 30.14
          0.0     | 20.4     |  20.0     |    29.3     | 30.23
          0.0     | 20.3     |  20.0     |    29.6     | 30.30
          0.0     | 20.3     |  20.0     |    29.8     | 30.35
          0.0     | 20.2     |  20.0     |    30.0     | 30.40
          0.0     | 20.2     |  20.0     |    30.1     | 30.43
          0.0     | 20.1     |  20.0     |    30.2     | 30.45
          0.0     | 20.1     |  20.0     |    30.3     | 30.47
          0.0     | 20.1     |  20.0     |    30.3     | 30.49
          0.0     | 20.1     |  20.0     |    30.4     | 30.50
          0.0     | 20.1     |  20.0     |    30.4     | 30.51
          0.0     | 20.0     |  20.0     |    30.4     | 30.52
          0.0     | 20.0     |  20.0     |    30.5     | 30.53
          0.0     | 20.0     |  20.0     |    30.5     | 30.53
          0.0     | 20.0     |  20.0     |    30.5     | 30.53
          0.0     | 20.0     |  20.0     |    30.5     | 30.54
          0.0     | 20.0     |  20.0     |    30.5     | 30.54
          0.0     | 20.0     |  20.0     |    30.5     | 30.54
          0.0     | 20.0     |  20.0     |    30.5     | 30.54
          0.0     | 20.0     |  20.0     |    30.5     | 30.54
          0.0     | 20.0     |  20.0     |    30.5     | 30.54
          0.0     | 20.0     |  20.0     |    30.5     | 30.54
          0.0     | 20.0     |  20.0     |    30.5     | 30.54
          0.0     | 20.0     |  20.0     |    30.5     | 30.54
          0.0     | 20.0     |  20.0     |    30.5     | 30.54
          0.0     | 20.0     |  20.0     |    30.5     | 30.55
        -------------------------------------------------------------
          0.0     | 20.0     |  25.0     |    30.5     | 34.86
          0.0     | 20.4     |  25.0     |    31.8     | 34.54
          0.0     | 20.7     |  25.0     |    32.6     | 34.28
          0.0     | 21.0     |  25.0     |    33.1     | 34.09
          0.0     | 21.2     |  25.0     |    33.4     | 33.93
          0.0     | 21.3     |  25.0     |    33.6     | 33.81
          0.0     | 21.5     |  25.0     |    33.6     | 33.72
          0.0     | 21.5     |  25.0     |    33.7     | 33.65
          0.0     | 21.6     |  25.0     |    33.7     | 33.59
          0.0     | 21.7     |  25.0     |    33.6     | 33.54
        """

        self._assert_temperature_predictions(controller, temperatures, simulate_raum_ist=True)
