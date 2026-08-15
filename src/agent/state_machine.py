"""Deterministic application of planned exchanges to booking state."""

from copy import deepcopy
from typing import Any

from src.models.enums import DeviationType, StateChangeSource, StateOperation
from src.models.plan import PlannedExchange, PlannedStateChange
from src.models.state import BookingState, SELECTED_SERVICES_FIELD


class StateTransitionError(ValueError):
    """A planned exchange cannot be safely applied to booking state."""


class StateMachine:
    MAX_SERVICES = 4

    def apply(self, exchange: PlannedExchange, state: BookingState) -> None:
        """Apply one exchange atomically, leaving original_values untouched."""

        if exchange.deviation is DeviationType.D1 and exchange.state_changes:
            raise StateTransitionError("D1 must not change booking state")

        candidate = deepcopy(state.current_values)
        for change in exchange.state_changes:
            self._apply_change(candidate, change)

        state.current_values.clear()
        state.current_values.update(candidate)
        if exchange.deviation is not None:
            state.used_deviations.append(exchange.deviation)

    def _apply_change(self, values: dict[str, Any], change: PlannedStateChange) -> None:
        if change.operation is StateOperation.REPLACE:
            values[change.field_name] = deepcopy(change.value)
            return
        if change.operation is StateOperation.ADD:
            if change.field_name != SELECTED_SERVICES_FIELD:
                raise StateTransitionError("ADD is only supported for selected_services")
            self._add_service(values, change)
            return
        raise StateTransitionError(f"unsupported state operation: {change.operation}")

    def _add_service(self, values: dict[str, Any], change: PlannedStateChange) -> None:
        services = values.get(SELECTED_SERVICES_FIELD)
        if not isinstance(services, list):
            raise StateTransitionError("selected_services must exist as a list in current_values")
        if change.value in services:
            raise StateTransitionError("service is already selected")
        if len(services) >= self.MAX_SERVICES:
            raise StateTransitionError("a booking may contain at most four services")
        services.append(deepcopy(change.value))
