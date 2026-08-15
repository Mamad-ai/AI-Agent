"""Validation of planned state transitions and final booking state."""

from src.agent.state_machine import StateMachine, StateTransitionError
from src.models.enums import (
    DeviationType,
    PhaseVariant,
    StateChangeSource,
    StateOperation,
    ValidationScope,
    ValidationSeverity,
    ValidationSource,
)
from src.models.plan import PlannedExchange, TurnPlan
from src.models.state import BookingState
from src.models.template import ConversationTemplate, thaw_value
from src.models.validation import ValidationIssue, ValidationReport


class StateValidator:
    def __init__(self, state_machine: StateMachine | None = None) -> None:
        self._state_machine = state_machine or StateMachine()

    def validate(
        self, plan: TurnPlan, template: ConversationTemplate, actual: BookingState
    ) -> ValidationReport:
        issues: list[ValidationIssue] = []
        if thaw_value(actual.original_values) != thaw_value(template.original_values):
            self._add(issues, "ORIGINAL_VALUES_CHANGED", "original template values changed")

        expected = BookingState(template.original_values, thaw_value(template.original_values))
        for exchange in plan.exchanges:
            self._validate_exchange_rules(exchange, issues)
            try:
                self._state_machine.apply(exchange, expected)
            except StateTransitionError as error:
                self._add(issues, "INVALID_STATE_TRANSITION", str(error), exchange.exchange_number)

        if actual.current_values != expected.current_values:
            self._add(issues, "CURRENT_STATE_MISMATCH", "current_values do not match planned changes")
        if actual.used_deviations != expected.used_deviations:
            self._add(
                issues,
                "USED_DEVIATIONS_MISMATCH",
                "used_deviations do not match explicit planned deviations",
            )
        services = actual.current_values.get("selected_services", ())
        if isinstance(services, (list, tuple)) and len(services) > StateMachine.MAX_SERVICES:
            self._add(issues, "TOO_MANY_SERVICES", "booking contains more than four services")
        return ValidationReport(self.__class__.__name__, not issues, tuple(issues))

    def _validate_exchange_rules(
        self, exchange: PlannedExchange, issues: list[ValidationIssue]
    ) -> None:
        deviation_changes = tuple(
            item for item in exchange.state_changes if item.source is StateChangeSource.DEVIATION
        )
        phase_changes = tuple(
            item for item in exchange.state_changes if item.source is StateChangeSource.PHASE_VARIANT
        )
        deviation = exchange.deviation
        if deviation is DeviationType.D1 and exchange.state_changes:
            self._add(issues, "D1_CHANGES_STATE", "D1 must not change state", exchange.exchange_number)
        elif deviation is DeviationType.D2 and not (
            len(deviation_changes) == 1
            and deviation_changes[0].operation is StateOperation.REPLACE
        ):
            self._add(issues, "INVALID_D2_STATE", "D2 requires one REPLACE", exchange.exchange_number)
        elif deviation is DeviationType.D3 and not (
            len(deviation_changes) == 1
            and deviation_changes[0].operation is StateOperation.ADD
            and deviation_changes[0].field_name == "selected_services"
        ):
            self._add(issues, "INVALID_D3_STATE", "D3 requires one service ADD", exchange.exchange_number)
        elif deviation is DeviationType.D5 and not (
            len(deviation_changes) >= 2
            and all(item.operation is StateOperation.REPLACE for item in deviation_changes)
            and len({item.field_name for item in deviation_changes}) >= 2
        ):
            self._add(
                issues, "INVALID_D5_STATE", "D5 requires distinct REPLACE changes", exchange.exchange_number
            )

        if exchange.phase_variant is PhaseVariant.P4_2 and not (
            len(phase_changes) == 1 and phase_changes[0].operation is StateOperation.REPLACE
        ):
            self._add(issues, "INVALID_P4_2_STATE", "P4.2 requires one phase REPLACE", exchange.exchange_number)
        if exchange.phase_variant is PhaseVariant.P4_3 and not (
            len(phase_changes) >= 2
            and all(item.operation is StateOperation.REPLACE for item in phase_changes)
            and len({item.field_name for item in phase_changes}) >= 2
        ):
            self._add(
                issues,
                "INVALID_P4_3_STATE",
                "P4.3 requires distinct phase REPLACE changes",
                exchange.exchange_number,
            )

    @staticmethod
    def _add(issues: list[ValidationIssue], code: str, message: str, number: int | None = None) -> None:
        issues.append(
            ValidationIssue(
                code,
                message,
                ValidationSeverity.HARD,
                ValidationSource.DETERMINISTIC,
                ValidationScope.STATE,
                exchange_number=number,
            )
        )
