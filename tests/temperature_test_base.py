"""Base class for temperature prediction tests with physics simulation."""

from __future__ import annotations

from copy import replace
from dataclasses import dataclass

from custom_components.heatpump_flow_control.flow_controller import (
    FlowController,
    SensorValues,
)
import pytest


@dataclass
class Row:
    """Expected row data for temperature table."""

    t_aussen: float
    raum_ist: float
    raum_soll: float
    vorlauf_ist: float
    vorlauf_soll: float

    def copy(self) -> Row:
        """Erstellt eine echte Kopie des Objekts."""
        return replace(self)

@dataclass
class Error:
    """Error information for temperature prediction tests."""

    row: int
    type: str
    t_aussen: float
    expected: float
    actual: float
    diff: float
    prev_raum_ist: float | None = None
    prev_vorlauf_soll: float | None = None
    prev_vorlauf_ist: float | None = None


class TemperaturePredictionTestBase:
    """Base class for temperature prediction tests with physics simulation."""

    # Physics simulation parameters - adjusted to match realistic heating behavior
    # With these values, a room at 22.5°C with vorlauf=31.4°C and outside=0°C stays stable
    HEATING_FACTOR = 0.095  # How much vorlauf heats the room per hour (reduced for realism)
    COOLING_FACTOR = 0.05  # Heat loss to outside per hour (increased for better insulation)
    VORLAUF_ADJUSTMENT_FACTOR = 0.3  # How fast vorlauf_ist approaches vorlauf_soll per hour

    def _simulate_raum_ist_change(
        self,
        raum_ist_old: float,
        vorlauf_soll: float,
        aussen_temp: float,
        hours: float
    ) -> float:
        """Simulate how raum_ist changes based on vorlauf_soll and outside temperature.

        Args:
            raum_ist_old: Previous room temperature
            vorlauf_soll: Flow temperature setpoint
            aussen_temp: Outside temperature
            hours: Time period in hours

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
        hours: float
    ) -> float:
        """Simulate how vorlauf_ist changes towards vorlauf_soll.

        Args:
            vorlauf_ist_old: Previous flow temperature
            vorlauf_soll: Flow temperature setpoint
            hours: Time period in hours

        Returns:
            New flow temperature moving towards setpoint
        """
        # Flow temperature adjusts towards setpoint
        delta = (vorlauf_soll - vorlauf_ist_old) * self.VORLAUF_ADJUSTMENT_FACTOR * hours
        return vorlauf_ist_old + delta

    def _parse_temperature_table(self, temperature_table: str) -> tuple[list[Row], set[int]]:
        """Parse temperature table string into rows and separator positions.

        Args:
            temperature_table: Table string with temperature data

        Returns:
            Tuple of (expected_rows, separator_before_row)
        """
        expected_rows: list[Row] = []
        separator_before_row = set()

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
                expected_rows.append(Row(
                    t_aussen = float(parts[0]),
                    raum_ist = float(parts[1]),
                    raum_soll = float(parts[2]),
                    vorlauf_ist = float(parts[3]),
                    vorlauf_soll = float(parts[4])
                ))

        return expected_rows, separator_before_row

    def _validate_physics_simulation(
        self,
        expected: Row,
        prev_raum_ist: float | None,
        prev_vorlauf_soll: float | None,
        prev_vorlauf_ist: float | None,
        row_index: int
    ) -> tuple[list[Error], float, float]:
        """Validate that table values match physics simulation.

        Args:
            expected: Expected row data
            prev_raum_ist: Previous room temperature
            prev_vorlauf_soll: Previous flow setpoint
            prev_vorlauf_ist: Previous actual flow temperature
            row_index: Current row index

        Returns:
            Tuple of (errors, raum_ist_to_use, vorlauf_ist_to_use)
        """
        errors = list[Error]()
        raum_ist_to_use = expected.raum_ist
        vorlauf_ist_to_use = expected.vorlauf_ist

        # Skip first row (no previous values)
        if (row_index == 0 or
            prev_raum_ist is None or
            prev_vorlauf_soll is None or
            prev_vorlauf_ist is None):
            return errors, raum_ist_to_use, vorlauf_ist_to_use

        # Validate raum_ist against simulation
        simulated_raum_ist = self._simulate_raum_ist_change(
            raum_ist_old=prev_raum_ist,
            vorlauf_soll=prev_vorlauf_soll,
            aussen_temp=expected.t_aussen,
            hours=1.0
        )

        raum_ist_diff = abs(expected.raum_ist - simulated_raum_ist)
        if raum_ist_diff > 0.1:
            errors.append(Error(
                row=row_index + 1,
                type='raum_ist_simulation',
                t_aussen=expected.t_aussen,
                expected=expected.raum_ist,
                actual=simulated_raum_ist,
                diff=raum_ist_diff,
                prev_raum_ist=prev_raum_ist,
                prev_vorlauf_soll=prev_vorlauf_soll,
            ))

        # Validate vorlauf_ist against simulation
        simulated_vorlauf_ist = self._simulate_vorlauf_ist_change(
            vorlauf_ist_old=prev_vorlauf_ist,
            vorlauf_soll=prev_vorlauf_soll,
            hours=1.0
        )

        vorlauf_ist_diff = abs(expected.vorlauf_ist - simulated_vorlauf_ist)
        if vorlauf_ist_diff > 0.1:
            errors.append(Error(
                row=row_index + 1,
                type='vorlauf_ist_simulation',
                t_aussen=expected.t_aussen,
                expected=expected.vorlauf_ist,
                actual=simulated_vorlauf_ist,
                diff=vorlauf_ist_diff,
                prev_vorlauf_ist=prev_vorlauf_ist,
                prev_vorlauf_soll=prev_vorlauf_soll,
            ))

        return errors, raum_ist_to_use, vorlauf_ist_to_use

    def _execute_prediction(
        self,
        flow_controller: FlowController,
        expected: Row
    ) -> float:
        """Execute a single prediction.

        Args:
            flow_controller: The controller to test
            expected: Expected values for this prediction

        Returns:
            Predicted vorlauf_soll value
        """
        result = flow_controller.berechne_vorlauf_soll(
            SensorValues(
                aussen_temp=expected.t_aussen,
                raum_ist=expected.raum_ist,
                raum_soll=expected.raum_soll,
                vorlauf_ist=expected.vorlauf_ist,
            )
        )
        return result.vorlauf

    def _check_prediction_accuracy(
        self,
        actual_vorlauf: float,
        expected_vorlauf: float,
        tolerance: float,
        row_index: int,
        t_aussen: float
    ) -> Error | None:
        """Check if prediction is within tolerance.

        Returns:
            Error dict if outside tolerance, None otherwise
        """
        diff = abs(actual_vorlauf - expected_vorlauf)
        if diff > tolerance:
            return Error(
                row=row_index + 1,
                type='vorlauf_soll',
                t_aussen=t_aussen,
                expected=expected_vorlauf,
                actual=actual_vorlauf,
                diff=diff
            )
        return None

    def _execute_all_predictions(
        self,
        flow_controller: FlowController,
        expected_rows: list[Row],
        tolerance: float,
        simulate_raum_ist: bool
    ) -> tuple[list[Row], list[Error]]:
        """Execute all predictions and collect results.

        Returns:
            Tuple of (actual_rows, errors)
        """
        actual_rows = list[Row]()
        errors = list[Error]()

        prev_raum_ist = None
        prev_vorlauf_soll = None
        prev_vorlauf_ist = None

        for i, expected in enumerate(expected_rows):
            # Validate physics simulation if enabled
            if simulate_raum_ist:
                validation_errors, raum_ist_to_use, vorlauf_ist_to_use = self._validate_physics_simulation(
                    expected, prev_raum_ist, prev_vorlauf_soll, prev_vorlauf_ist, i
                )
                errors.extend(validation_errors)
            else:
                raum_ist_to_use = expected.raum_ist
                vorlauf_ist_to_use = expected.vorlauf_ist

            # Update expected with validated values
            expected_validated = expected.copy()
            expected_validated.raum_ist = raum_ist_to_use
            expected_validated.vorlauf_ist = vorlauf_ist_to_use

            # Execute prediction
            actual_vorlauf = self._execute_prediction(flow_controller, expected_validated)

            # Store results
            actual_rows.append(Row(
                t_aussen=expected.t_aussen,
                raum_ist=raum_ist_to_use,
                raum_soll=expected.raum_soll,
                vorlauf_ist=vorlauf_ist_to_use,
                vorlauf_soll=actual_vorlauf
            ))

            # Check prediction accuracy
            error = self._check_prediction_accuracy(
                actual_vorlauf, expected.vorlauf_soll, tolerance, i, expected.t_aussen
            )
            if error:
                errors.append(error)

            # Store for next iteration
            prev_raum_ist = raum_ist_to_use
            prev_vorlauf_soll = actual_vorlauf
            prev_vorlauf_ist = vorlauf_ist_to_use

        return actual_rows, errors

    def _stabilize_table(
        self,
        flow_controller: FlowController,
        initial_rows: list[Row],
        separator_before_row: set[int],
        max_iterations: int = 10
    ) -> tuple[list[Row], int]:
        """Stabilize table through iterative simulation until convergence.

        Args:
            flow_controller: Controller to use for predictions
            initial_rows: Initial row configuration (t_aussen, raum_soll)
            separator_before_row: Set of row indices that have separators
            max_iterations: Maximum iterations before giving up

        Returns:
            Tuple of (stabilized_rows, iterations_needed)
        """
        converged = False
        iteration = 0
        prev_iteration_rows = None
        current_rows = initial_rows

        while not converged and iteration < max_iterations:
            iteration += 1
            new_rows = []

            prev_raum_ist = current_rows[0].raum_ist
            prev_vorlauf_ist = current_rows[0].vorlauf_ist
            prev_vorlauf_soll: float | None = None

            for i, row in enumerate(current_rows):
                # For first row, use provided initial values
                if i == 0:
                    raum_ist = row.raum_ist
                    vorlauf_ist = row.vorlauf_ist
                elif prev_vorlauf_soll is not None:
                    # Simulate based on previous row's vorlauf_soll
                    raum_ist = self._simulate_raum_ist_change(
                        prev_raum_ist, prev_vorlauf_soll, row.t_aussen, 1.0
                    )
                    vorlauf_ist = self._simulate_vorlauf_ist_change(
                        prev_vorlauf_ist, prev_vorlauf_soll, 1.0
                    )
                else:
                    # Fallback for second row (should not happen, but safe)
                    raum_ist = row.raum_ist
                    vorlauf_ist = row.vorlauf_ist

                # Get new prediction with simulated values
                result = flow_controller.berechne_vorlauf_soll(
                    SensorValues(
                        aussen_temp=row.t_aussen,
                        raum_ist=raum_ist,
                        raum_soll=row.raum_soll,
                        vorlauf_ist=vorlauf_ist,
                    )
                )
                vorlauf_soll = result.vorlauf

                new_rows.append(Row(
                    t_aussen=row.t_aussen,
                    raum_ist=raum_ist,
                    raum_soll=row.raum_soll,
                    vorlauf_ist=vorlauf_ist,
                    vorlauf_soll=vorlauf_soll
                ))

                # Store for next row
                prev_raum_ist = raum_ist
                prev_vorlauf_ist = vorlauf_ist
                prev_vorlauf_soll = vorlauf_soll

            # Check convergence
            if prev_iteration_rows:
                max_diff = 0.0
                for curr, prev in zip(new_rows, prev_iteration_rows, strict=True):
                    diff = max(
                        abs(curr.raum_ist - prev.raum_ist),
                        abs(curr.vorlauf_ist - prev.vorlauf_ist),
                        abs(curr.vorlauf_soll - prev.vorlauf_soll)
                    )
                    max_diff = max(max_diff, diff)

                if max_diff < 0.01:
                    converged = True

            prev_iteration_rows = new_rows
            current_rows = new_rows

        return current_rows, iteration

    def _format_simulation_error_output(
        self,
        flow_controller: FlowController,
        actual_rows: list[Row],
        errors: list[Error],
        separator_before_row: set[int]
    ) -> list[str]:
        """Format output for simulation errors.

        Returns:
            List of output lines
        """
        output_lines = []

        raum_ist_errors = [e for e in errors if e.type == 'raum_ist_simulation']
        vorlauf_ist_errors = [e for e in errors if e.type == 'vorlauf_ist_simulation']

        error_types = []
        if raum_ist_errors:
            error_types.append("raum_ist")
        if vorlauf_ist_errors:
            error_types.append("vorlauf_ist")

        # Stabilize table internally before displaying
        stabilized_rows, iterations = self._stabilize_table(
            flow_controller, actual_rows, separator_before_row, max_iterations=10
        )

        output_lines.extend([
            f"\n{' AND '.join(error_types).upper()} SIMULATION ERRORS:",
            "-"*80,
            "The values in the table don't match the physics simulation!",
            f"Using physics: HEATING_FACTOR={self.HEATING_FACTOR}, COOLING_FACTOR={self.COOLING_FACTOR}, VORLAUF_ADJUSTMENT_FACTOR={self.VORLAUF_ADJUSTMENT_FACTOR}",
            f"Table stabilized after {iterations} iteration(s)",
            "-"*80,
            "\nCORRECTED TABLE WITH SIMULATED VALUES (copy this to update test):",
            "-"*80,
        ])

        # Display stabilized table
        output_lines.extend([
            "        t_aussen | raum-ist | raum-soll | vorlauf_ist | vorlauf_soll",
            "        -------------------------------------------------------------"
        ])

        for i, row in enumerate(stabilized_rows):
            if i in separator_before_row and i > 0:
                output_lines.append("        -------------------------------------------------------------")

            line = (
                f"        {row.t_aussen:>5.1f}     | "
                f"{row.raum_ist:>4.1f}     | "
                f"{row.raum_soll:>5.1f}     | "
                f"{row.vorlauf_ist:>7.1f}     | "
                f"{row.vorlauf_soll:.2f}"
            )
            output_lines.append(line)

        output_lines.extend(["-"*70, "\nDETAILS (first 10 errors):", "-"*70])

        if raum_ist_errors:
            output_lines.append("\nRAUM_IST errors:")
            output_lines.extend([
                f"Row {err.row:2d}: Table={err.expected:.2f}°C → "
                f"Simulated={err.actual:.2f}°C (Diff: {err.diff:.2f}°C)"
                for err in raum_ist_errors[:10]
            ])
            if len(raum_ist_errors) > 10:
                output_lines.append(f"... and {len(raum_ist_errors) - 10} more errors")

        if vorlauf_ist_errors:
            output_lines.append("\nVORLAUF_IST errors:")
            output_lines.extend([
                f"Row {err.row:2d}: Table={err.expected:.2f}°C → "
                f"Simulated={err.actual:.2f}°C (Diff: {err.diff:.2f}°C)"
                for err in vorlauf_ist_errors[:10]
            ])
            if len(vorlauf_ist_errors) > 10:
                output_lines.append(f"... and {len(vorlauf_ist_errors) - 10} more errors")

        return output_lines

    def _format_prediction_error_output(
        self,
        flow_controller: FlowController,
        actual_rows: list[Row],
        errors: list[Error],
        separator_before_row: set[int],
        tolerance: float
    ) -> list[str]:
        """Format output for prediction errors.

        Returns:
            List of output lines
        """
        # Stabilize table internally before displaying
        stabilized_rows, iterations = self._stabilize_table(
            flow_controller, actual_rows, separator_before_row, max_iterations=10
        )

        output_lines = [
            "\nVORLAUF_SOLL PREDICTION ERRORS:",
            "-"*80,
            f"Table stabilized after {iterations} iteration(s)",
            "\nSTABLE OUTPUT TABLE (copy this to update test):",
            "-"*80,
            "        t_aussen | raum-ist | raum-soll | vorlauf_ist | vorlauf_soll",
            "        -------------------------------------------------------------"
        ]

        for i, row in enumerate(stabilized_rows):
            if i in separator_before_row and i > 0:
                output_lines.append("        -------------------------------------------------------------")

            line = (
                f"        {row.t_aussen:>5.1f}     | "
                f"{row.raum_ist:>4.1f}     | "
                f"{row.raum_soll:>5.1f}     | "
                f"{row.vorlauf_ist:>7.1f}     | "
                f"{row.vorlauf_soll:.2f}"
            )
            output_lines.append(line)

        output_lines.extend(["-"*70, "\nERROR DETAILS:", "-"*70])

        output_lines.extend([
            f"Row {err.row:2d}: t_aussen={err.t_aussen:>6.1f}°C  "
            f"Expected: {err.expected:.2f}°C  "
            f"Actual: {err.actual:.2f}°C  "
            f"Diff: {err.diff:.2f}°C (tolerance: {tolerance}°C)"
            for err in errors
        ])

        return output_lines

    def _assert_temperature_predictions(
        self,
        flow_controller: FlowController,
        temperature_table: str,
        tolerance: float = 0.1,
        simulate_raum_ist: bool = False
    ) -> None:
        """Test temperature predictions against expected table.

        Args:
            flow_controller: The controller to test
            temperature_table: Table with expected values
            tolerance: Allowed deviation in °C
            simulate_raum_ist: If True, validate physics simulation
        """
        # Parse input table
        expected_rows, separator_before_row = self._parse_temperature_table(temperature_table)

        # Execute predictions
        actual_rows, errors = self._execute_all_predictions(
            flow_controller, expected_rows, tolerance, simulate_raum_ist
        )

        # Report errors if any
        if not errors:
            return

        output_lines = [
            "\n" + "="*80,
            "TEST FAILED - Predictions don't match expected values",
            "="*80,
        ]

        raum_ist_errors = [e for e in errors if e.type == 'raum_ist_simulation']
        vorlauf_ist_errors = [e for e in errors if e.type == 'vorlauf_ist_simulation']
        vorlauf_errors = [e for e in errors if e.type == 'vorlauf_soll']

        # Format appropriate error output
        if raum_ist_errors or vorlauf_ist_errors:
            error_output = self._format_simulation_error_output(
                flow_controller, actual_rows, errors, separator_before_row
            )
            output_lines.extend(error_output)

            # Also print table for easy copying
            table_start = next(i for i, line in enumerate(error_output)
                             if "CORRECTED TABLE" in line)
            print("\n".join(error_output[table_start:]))  # noqa: T201

        elif vorlauf_errors:
            error_output = self._format_prediction_error_output(
                flow_controller, actual_rows, vorlauf_errors, separator_before_row, tolerance
            )
            output_lines.extend(error_output)

            # Also print table for easy copying
            table_start = next(i for i, line in enumerate(error_output)
                             if "STABLE OUTPUT TABLE" in line)
            print("\n".join(error_output[table_start:]))  # noqa: T201

        output_lines.append("="*70 + "\n")
        pytest.fail("\n".join(output_lines))
