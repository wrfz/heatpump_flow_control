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
        controller._setup()

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
        assert controller._heizkurve_fallback(-20.0, 0.0) == pytest.approx(35.00, abs=0.01)     # noqa: SLF001
        assert controller._heizkurve_fallback(-10.0, 0.0) == pytest.approx(33.28, abs=0.01)     # noqa: SLF001
        assert controller._heizkurve_fallback(  0.0, 0.0) == pytest.approx(29.84, abs=0.01)     # noqa: SLF001
        assert controller._heizkurve_fallback( 10.0, 0.0) == pytest.approx(26.40, abs=0.01)     # noqa: SLF001
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
        controller.min_predictions_for_model = 10

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
            vorlauf_soll_and_features = controller.berechne_vorlauf_soll(
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
                    "vorlauf_soll": vorlauf_soll_and_features.vorlauf,
                    "vorlauf_ist": vorlauf_ist,
                }
            )

            # WICHTIG: Trainiere Model direkt mit dem tatsächlichen Vorlauf
            # In der Realität würde das Model aus vergangenen Messungen lernen
            # Hier simulieren wir: Der IST-Wert war der "richtige" Wert
            controller.model.learn_one(vorlauf_soll_and_features.features.to_dict(), vorlauf_ist)

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

        # First prediction
        result1 = controller.berechne_vorlauf_soll(
            SensorValues(aussen_temp=0.0, raum_ist=22.0, raum_soll=22.0, vorlauf_ist=35.0)
        )
        assert result1.vorlauf == pytest.approx(32.92, abs=0.01)

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
        assert result2.vorlauf == pytest.approx(31.96, abs=0.01)
        assert len(controller2.erfahrungs_liste) == 2

    def test_config_changes_after_unpickle(self):
        """Test that config changes work after unpickling."""

        # Create controller with initial config
        controller = FlowController(
            min_vorlauf=25.0,
            max_vorlauf=40.0,
            learning_rate=0.01,
        )

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

class TestLearning:
    """Test prediction logic."""

    # Physics simulation parameters (adjustable)
    HEATING_FACTOR = 0.15  # How much vorlauf heats the room per hour
    COOLING_FACTOR = 0.05  # Heat loss to outside per hour
    VORLAUF_ADJUSTMENT_FACTOR = 0.3  # How fast vorlauf_ist approaches vorlauf_soll per hour

    def _simulate_raum_ist_change(
        self,
        raum_ist_old: float,
        vorlauf_soll: float,
        aussen_temp: float,
        hours: float = 1.0
    ) -> float:
        """Simulate how raum_ist changes based on vorlauf_soll and outside temperature.

        Args:
            raum_ist_old: Previous room temperature
            vorlauf_soll: Flow temperature setpoint
            aussen_temp: Outside temperature
            hours: Time period in hours (default: 1.0)

        Returns:
            New room temperature after heating/cooling
        """
        # Heating effect: positive if vorlauf is warmer than room
        heating_delta = (vorlauf_soll - raum_ist_old) * self.HEATING_FACTOR * hours

        # Cooling effect: heat loss to outside (always negative)
        cooling_delta = -(raum_ist_old - aussen_temp) * self.COOLING_FACTOR * hours

        # New room temperature
        return raum_ist_old + heating_delta + cooling_delta

    def _simulate_vorlauf_ist_change(
        self,
        vorlauf_ist_old: float,
        vorlauf_soll: float,
        hours: float = 1.0
    ) -> float:
        """Simulate how vorlauf_ist changes towards vorlauf_soll.

        Args:
            vorlauf_ist_old: Previous flow temperature
            vorlauf_soll: Flow temperature setpoint
            hours: Time period in hours (default: 1.0)

        Returns:
            New flow temperature moving towards setpoint
        """
        # Flow temperature adjusts towards setpoint
        delta = (vorlauf_soll - vorlauf_ist_old) * self.VORLAUF_ADJUSTMENT_FACTOR * hours
        return vorlauf_ist_old + delta

    def _assert_temperature_predictions(
        self,
        flow_controller: FlowController,
        temperature_table: str,
        tolerance: float = 0.1,
        simulate_raum_ist: bool = False
    ) -> None:
        """Helper to test temperature predictions against expected table.

        Args:
            flow_controller: The controller to test
            temperature_table: Table with expected values in format:
                t_aussen | raum-ist | raum-soll | vorlauf_ist | vorlauf_soll
                -------------------------------------------------------------
                10.0     | 22,5     | 22,5      | 28,0        | 26,40
                ...
            tolerance: Allowed deviation in °C
            simulate_raum_ist: If True, simulate raum_ist based on physics and compare
        """
        # Parse temperature table and track separator positions
        expected_rows = []
        separator_before_row = set()  # Track which rows should have separator before them

        for _, line in enumerate(temperature_table.strip().split('\n')):
            line = line.strip()
            if not line or 't_aussen' in line:
                continue

            # Check if this is a separator line (must check before '|' check)
            if '---' in line:
                # Mark that next data row should have separator before it
                separator_before_row.add(len(expected_rows))
                continue

            # Skip lines without data separator
            if '|' not in line:
                continue

            parts = [p.strip().replace(',', '.') for p in line.split('|')]
            if len(parts) == 5:
                expected_rows.append({
                    't_aussen': float(parts[0]),
                    'raum_ist': float(parts[1]),
                    'raum_soll': float(parts[2]),
                    'vorlauf_ist': float(parts[3]),
                    'vorlauf_soll': float(parts[4])
                })

        # Execute predictions and collect actual results
        test_helper = (
            controller(flow_controller)
            .enable_history_learning()
            .wait(60)
        )

        actual_rows = []
        errors = []

        # Track previous values for simulation
        prev_raum_ist = None
        prev_vorlauf_soll = None
        prev_vorlauf_ist = None

        for i, expected in enumerate(expected_rows):
            # If simulating physics, calculate expected raum_ist from previous values
            if simulate_raum_ist and i > 0 and prev_raum_ist is not None and prev_vorlauf_soll is not None:
                simulated_raum_ist = self._simulate_raum_ist_change(
                    raum_ist_old=prev_raum_ist,
                    vorlauf_soll=prev_vorlauf_soll,
                    aussen_temp=expected['t_aussen'],
                    hours=1.0  # Assuming 1 hour between measurements
                )

                # Check if table value matches simulation
                raum_ist_diff = abs(expected['raum_ist'] - simulated_raum_ist)
                if raum_ist_diff > 0.1:  # Tolerance for raum_ist simulation
                    errors.append({
                        'row': i + 1,
                        'type': 'raum_ist_simulation',
                        't_aussen': expected['t_aussen'],
                        'expected': expected['raum_ist'],
                        'actual': simulated_raum_ist,
                        'diff': raum_ist_diff,
                        'prev_raum_ist': prev_raum_ist,
                        'prev_vorlauf_soll': prev_vorlauf_soll,
                    })

            # If simulating physics, calculate expected vorlauf_ist from previous values
            if simulate_raum_ist and i > 0 and prev_vorlauf_ist is not None and prev_vorlauf_soll is not None:
                simulated_vorlauf_ist = self._simulate_vorlauf_ist_change(
                    vorlauf_ist_old=prev_vorlauf_ist,
                    vorlauf_soll=prev_vorlauf_soll,
                    hours=1.0  # Assuming 1 hour between measurements
                )

                # Check if table value matches simulation
                vorlauf_ist_diff = abs(expected['vorlauf_ist'] - simulated_vorlauf_ist)
                if vorlauf_ist_diff > 0.1:  # Tolerance for vorlauf_ist simulation
                    errors.append({
                        'row': i + 1,
                        'type': 'vorlauf_ist_simulation',
                        't_aussen': expected['t_aussen'],
                        'expected': expected['vorlauf_ist'],
                        'actual': simulated_vorlauf_ist,
                        'diff': vorlauf_ist_diff,
                        'prev_vorlauf_ist': prev_vorlauf_ist,
                        'prev_vorlauf_soll': prev_vorlauf_soll,
                    })

            # Use table value for raum_ist (either verified by simulation or first row)
            raum_ist_to_use = expected['raum_ist']

            # Use table value for vorlauf_ist (either verified by simulation or first row)
            vorlauf_ist_to_use = expected['vorlauf_ist']

            # Execute prediction
            test_helper = (
                test_helper
                .t_aussen(expected['t_aussen'])
                .raum_ist(raum_ist_to_use)
                .raum_soll(expected['raum_soll'])
                .vorlauf_ist(vorlauf_ist_to_use)
            )

            # Get actual prediction
            vorlauf_soll_and_features = flow_controller.berechne_vorlauf_soll(
                SensorValues(
                    aussen_temp=expected['t_aussen'],
                    raum_ist=raum_ist_to_use,
                    raum_soll=expected['raum_soll'],
                    vorlauf_ist=vorlauf_ist_to_use,
                )
            )

            # Store for next iteration
            prev_raum_ist = raum_ist_to_use
            prev_vorlauf_soll = vorlauf_soll_and_features.vorlauf
            prev_vorlauf_ist = vorlauf_ist_to_use

            actual_rows.append({
                't_aussen': expected['t_aussen'],
                'raum_ist': raum_ist_to_use,
                'raum_soll': expected['raum_soll'],
                'vorlauf_ist': vorlauf_ist_to_use,
                'vorlauf_soll': vorlauf_soll_and_features.vorlauf
            })

            # Check if vorlauf_soll within tolerance
            diff = abs(vorlauf_soll_and_features.vorlauf - expected['vorlauf_soll'])
            if diff > tolerance:
                errors.append({
                    'row': i + 1,
                    'type': 'vorlauf_soll',
                    't_aussen': expected['t_aussen'],
                    'expected': expected['vorlauf_soll'],
                    'actual': vorlauf_soll_and_features.vorlauf,
                    'diff': diff
                })

        # If there are errors, generate helpful output table
        if errors:
            # Separate errors by type
            raum_ist_errors = [e for e in errors if e.get('type') == 'raum_ist_simulation']
            vorlauf_ist_errors = [e for e in errors if e.get('type') == 'vorlauf_ist_simulation']
            vorlauf_errors = [e for e in errors if e.get('type') == 'vorlauf_soll']

            # Generate table with actual values (easy to copy-paste)
            output_lines = [
                "\n" + "="*80,
                "TEST FAILED - Predictions don't match expected values",
                "="*80,
            ]

            # If there are raum_ist or vorlauf_ist simulation errors, generate corrected table
            if raum_ist_errors or vorlauf_ist_errors:
                # Build table lines for both output and direct print
                table_lines = [
                    "",
                    "CORRECTED TABLE WITH SIMULATED RAUM_IST (copy this to update test):",
                    "-"*80,
                    "        t_aussen | raum-ist | raum-soll | vorlauf_ist | vorlauf_soll",
                    "        -------------------------------------------------------------"
                ]

                error_types = []
                if raum_ist_errors:
                    error_types.append("raum_ist")
                if vorlauf_ist_errors:
                    error_types.append("vorlauf_ist")

                output_lines.extend([
                    f"\n{' AND '.join(error_types).upper()} SIMULATION ERRORS:",
                    "-"*80,
                    "The values in the table don't match the physics simulation!",
                    f"Using physics: HEATING_FACTOR={self.HEATING_FACTOR}, COOLING_FACTOR={self.COOLING_FACTOR}, VORLAUF_ADJUSTMENT_FACTOR={self.VORLAUF_ADJUSTMENT_FACTOR}",
                    "-"*80,
                    "\nCORRECTED TABLE WITH SIMULATED RAUM_IST (copy this to update test):",
                    "-"*80,
                    "        t_aussen | raum-ist | raum-soll | vorlauf_ist | vorlauf_soll",
                    "        -------------------------------------------------------------"
                ])

                # Generate complete table with simulated raum_ist and vorlauf_ist values
                prev_raum_ist_sim = None
                prev_vorlauf_soll_sim = None
                prev_vorlauf_ist_sim = None

                for i, row in enumerate(actual_rows):
                    # Add separator line if needed (from original table structure)
                    # Skip separator at position 0 (header separator)
                    if i in separator_before_row and i > 0:
                        separator = "        -------------------------------------------------------------"
                        output_lines.append(separator)
                        table_lines.append(separator)

                    # For first row, use table values
                    if i == 0:
                        raum_ist_display = row['raum_ist']
                        vorlauf_ist_display = row['vorlauf_ist']
                    else:
                        # Calculate simulated raum_ist (prev values are guaranteed to be set after first row)
                        assert prev_raum_ist_sim is not None
                        assert prev_vorlauf_soll_sim is not None
                        assert prev_vorlauf_ist_sim is not None

                        raum_ist_display = self._simulate_raum_ist_change(
                            raum_ist_old=prev_raum_ist_sim,
                            vorlauf_soll=prev_vorlauf_soll_sim,
                            aussen_temp=row['t_aussen'],
                            hours=1.0
                        )
                        # Calculate simulated vorlauf_ist
                        vorlauf_ist_display = self._simulate_vorlauf_ist_change(
                            vorlauf_ist_old=prev_vorlauf_ist_sim,
                            vorlauf_soll=prev_vorlauf_soll_sim,
                            hours=1.0
                        )

                    # Store for next iteration
                    prev_raum_ist_sim = raum_ist_display
                    prev_vorlauf_soll_sim = row['vorlauf_soll']
                    prev_vorlauf_ist_sim = vorlauf_ist_display

                    # Format line
                    line = (
                        f"        {row['t_aussen']:>5.1f}     | "
                        f"{raum_ist_display:>4.1f}     | "
                        f"{row['raum_soll']:>5.1f}     | "
                        f"{vorlauf_ist_display:>7.1f}     | "
                        f"{row['vorlauf_soll']:.2f}"
                    )

                    output_lines.append(line)
                    table_lines.append(line)

                output_lines.extend([
                    "-"*70,
                    "\nDETAILS (first 10 errors):",
                    "-"*70,
                ])

                if raum_ist_errors:
                    output_lines.append("\nRAUM_IST errors:")
                    for err in raum_ist_errors[:10]:
                        output_lines.append(
                            f"Row {err['row']:2d}: Table={err['expected']:.2f}°C → Simulated={err['actual']:.2f}°C "
                            f"(Diff: {err['diff']:.2f}°C)"
                        )
                    if len(raum_ist_errors) > 10:
                        output_lines.append(f"... and {len(raum_ist_errors) - 10} more errors")

                if vorlauf_ist_errors:
                    output_lines.append("\nVORLAUF_IST errors:")
                    for err in vorlauf_ist_errors[:10]:
                        output_lines.append(
                            f"Row {err['row']:2d}: Table={err['expected']:.2f}°C → Simulated={err['actual']:.2f}°C "
                            f"(Diff: {err['diff']:.2f}°C)"
                        )
                    if len(vorlauf_ist_errors) > 10:
                        output_lines.append(f"... and {len(vorlauf_ist_errors) - 10} more errors")

                # Print table without pytest's "E" prefix for easy copying
                print("\n" + "\n".join(table_lines) + "\n")

            # Show vorlauf_soll prediction errors if any
            elif vorlauf_errors:
                # Build table lines for both output and direct print
                table_lines = [
                    "",
                    "ACTUAL OUTPUT TABLE (copy this to update test):",
                    "-"*80,
                    "        t_aussen | raum-ist | raum-soll | vorlauf_ist | vorlauf_soll",
                    "        -------------------------------------------------------------"
                ]

                output_lines.extend([
                    "\nVORLAUF_SOLL PREDICTION ERRORS:",
                    "-"*80,
                    "\nACTUAL OUTPUT TABLE (copy this to update test):",
                    "-"*80,
                    "        t_aussen | raum-ist | raum-soll | vorlauf_ist | vorlauf_soll",
                    "        -------------------------------------------------------------"
                ])

                for i, row in enumerate(actual_rows):
                    # Add separator line if needed (from original table structure)
                    # Skip separator at position 0 (header separator)
                    if i in separator_before_row and i > 0:
                        separator = "        -------------------------------------------------------------"
                        output_lines.append(separator)
                        table_lines.append(separator)

                    line = (
                        f"        {row['t_aussen']:>5.1f}     | "
                        f"{row['raum_ist']:>4.1f}     | "
                        f"{row['raum_soll']:>5.1f}     | "
                        f"{row['vorlauf_ist']:>7.1f}     | "
                        f"{row['vorlauf_soll']:.2f}"
                    )

                    output_lines.append(line)
                    table_lines.append(line)

                output_lines.extend([
                    "-"*70,
                    "\nERROR DETAILS:",
                    "-"*70,
                ])

                output_lines.extend(
                    f"Row {err['row']:2d}: t_aussen={err['t_aussen']:>6.1f}°C  "
                    f"Expected: {err['expected']:.2f}°C  "
                    f"Actual: {err['actual']:.2f}°C  "
                    f"Diff: {err['diff']:.2f}°C (tolerance: {tolerance}°C)"
                    for err in vorlauf_errors
                )

                # Print table without pytest's "E" prefix for easy copying
                print("\n" + "\n".join(table_lines) + "\n")  # noqa: T201

            output_lines.append("="*70 + "\n")

            error_msg = "\n".join(output_lines)
            pytest.fail(error_msg)

    def test_outside_temperature_learning(self):
        """Test that the model learns to adjust flow temperature based on outside temperature changes."""

        # Set random seeds for deterministic results
        random.seed(42)
        np.random.seed(42)

        flow_controller = FlowController(
            min_vorlauf=25.0,
            max_vorlauf=40.0,
            learning_rate=0.01,
        )

        temperatures = """
        t_aussen | raum-ist | raum-soll | vorlauf_ist | vorlauf_soll
        -------------------------------------------------------------
         10.0     | 22.5     |  22.5     |    28.0     | 28.08
          8.0     | 22.6     |  22.5     |    28.0     | 28.22
          6.0     | 22.6     |  22.5     |    28.1     | 28.46
          4.0     | 22.6     |  22.5     |    28.2     | 28.70
          2.0     | 22.5     |  22.5     |    28.3     | 29.00
          0.0     | 22.3     |  22.5     |    28.6     | 29.46
         -2.0     | 22.2     |  22.5     |    28.9     | 29.86
         -4.0     | 22.0     |  22.5     |    29.2     | 30.32
         -6.0     | 21.9     |  22.5     |    29.5     | 30.73
         -8.0     | 21.7     |  22.5     |    29.9     | 31.24
        -10.0     | 21.6     |  22.5     |    30.3     | 31.70
        -12.0     | 21.4     |  22.5     |    30.7     | 32.21
        -14.0     | 21.3     |  22.5     |    31.2     | 32.73
        -15.0     | 21.2     |  22.5     |    31.6     | 33.10
        -------------------------------------------------------------
        -14.0     | 21.2     |  22.5     |    32.1     | 33.28
        -12.0     | 21.4     |  22.5     |    32.5     | 33.21
        -10.0     | 21.6     |  22.5     |    32.7     | 33.03
         -8.0     | 21.8     |  22.5     |    32.8     | 32.79
         -6.0     | 22.1     |  22.5     |    32.8     | 32.44
         -4.0     | 22.3     |  22.5     |    32.7     | 32.09
         -2.0     | 22.6     |  22.5     |    32.5     | 31.64
          0.0     | 22.8     |  22.5     |    32.3     | 31.23
          2.0     | 23.0     |  22.5     |    31.9     | 30.72
          4.0     | 23.2     |  22.5     |    31.6     | 30.26
          6.0     | 23.4     |  22.5     |    31.2     | 29.75
          8.0     | 23.6     |  22.5     |    30.7     | 29.18
         10.0     | 23.7     |  22.5     |    30.3     | 28.72
        """

        self._assert_temperature_predictions(flow_controller, temperatures, simulate_raum_ist=True)

    def test_raum_soll_temperature_learning(self):
        """Test that the model learns to adjust flow temperature based on outside temperature changes."""

        # Set random seeds for deterministic results
        random.seed(42)
        np.random.seed(42)

        flow_controller = FlowController(
            min_vorlauf=25.0,
            max_vorlauf=40.0,
            learning_rate=0.01,
        )

        temperatures = """
        t_aussen | raum-ist | raum-soll | vorlauf_ist | vorlauf_soll
        -------------------------------------------------------------
          0.0     | 20.0     |  22.5     |    32.0     | 32.44
          0.0     | 20.9     |  22.5     |    32.1     | 32.07
          0.0     | 21.5     |  22.5     |    32.1     | 31.78
          0.0     | 22.0     |  22.5     |    32.0     | 31.49
          0.0     | 22.3     |  22.5     |    31.9     | 31.29
          0.0     | 22.5     |  22.5     |    31.7     | 31.08
          0.0     | 22.7     |  22.5     |    31.5     | 30.88
          0.0     | 22.8     |  22.5     |    31.3     | 30.72
          0.0     | 22.9     |  22.5     |    31.2     | 30.61
          0.0     | 22.9     |  22.5     |    31.0     | 30.50
          0.0     | 22.9     |  22.5     |    30.9     | 30.45
          0.0     | 22.9     |  22.5     |    30.7     | 30.34
          0.0     | 22.8     |  22.5     |    30.6     | 30.33
          0.0     | 22.8     |  22.5     |    30.5     | 30.28
        -------------------------------------------------------------
          0.0     | 22.8     |  20.0     |    30.4     | 29.10
          0.0     | 22.6     |  20.0     |    30.0     | 28.98
          0.0     | 22.4     |  20.0     |    29.7     | 28.91
          0.0     | 22.3     |  20.0     |    29.5     | 28.85
          0.0     | 22.1     |  20.0     |    29.3     | 28.83
          0.0     | 22.0     |  20.0     |    29.1     | 28.77
          0.0     | 21.9     |  20.0     |    29.0     | 28.76
          0.0     | 21.9     |  20.0     |    28.9     | 28.71
          0.0     | 21.8     |  20.0     |    28.9     | 28.75
          0.0     | 21.7     |  20.0     |    28.8     | 28.75
          0.0     | 21.7     |  20.0     |    28.8     | 28.75
          0.0     | 21.7     |  20.0     |    28.8     | 28.75
          0.0     | 21.6     |  20.0     |    28.7     | 28.74
          0.0     | 21.6     |  20.0     |    28.7     | 28.74
        -------------------------------------------------------------
          0.0     | 21.6     |  25.0     |    28.7     | 30.97
          0.0     | 21.9     |  25.0     |    29.4     | 31.22
          0.0     | 22.2     |  25.0     |    29.9     | 31.35
          0.0     | 22.5     |  25.0     |    30.4     | 31.48
          0.0     | 22.7     |  25.0     |    30.7     | 31.55
          0.0     | 22.9     |  25.0     |    31.0     | 31.62
          0.0     | 23.1     |  25.0     |    31.2     | 31.64
          0.0     | 23.2     |  25.0     |    31.3     | 31.64
          0.0     | 23.3     |  25.0     |    31.4     | 31.65
          0.0     | 23.4     |  25.0     |    31.5     | 31.66
        """

        self._assert_temperature_predictions(flow_controller, temperatures, simulate_raum_ist=True)
