"""Validation of structured P4 summaries against current booking state."""

from typing import Any, Mapping

from src.agent.state_machine import StateMachine, StateTransitionError
from src.models.dialogue import ConversationExchange, Dialogue
from src.models.enums import Phase, ValidationScope, ValidationSeverity, ValidationSource
from src.models.plan import PlannedExchange, TurnPlan
from src.models.state import BookingState
from src.models.template import ConversationTemplate, thaw_value
from src.models.validation import ValidationIssue, ValidationReport


class SummaryValidator:
    def __init__(self, state_machine: StateMachine | None = None) -> None:
        self._state_machine = state_machine or StateMachine()

    def validate(
        self, dialogue: Dialogue, plan: TurnPlan, template: ConversationTemplate
    ) -> ValidationReport:
        issues: list[ValidationIssue] = []
        state = BookingState(template.original_values, thaw_value(template.original_values))
        for planned in plan.exchanges:
            try:
                self._state_machine.apply(planned, state)
            except StateTransitionError:
                continue
            number = planned.exchange_number
            if planned.phase is Phase.P4 and number <= len(dialogue.exchanges):
                issues.extend(
                    self.validate_exchange(
                        dialogue.exchanges[number - 1], planned, state.current_values
                    ).issues
                )
        return ValidationReport(self.__class__.__name__, not issues, tuple(issues))

    def validate_exchange(
        self,
        generated: ConversationExchange,
        planned: PlannedExchange,
        expected_current_values: Mapping[str, Any],
    ) -> ValidationReport:
        """Validate one P4 summary against Python's expected post-exchange state."""

        issues: list[ValidationIssue] = []
        if planned.phase is Phase.P4:
            summary = thaw_value(generated.summary_facts)
            expected = thaw_value(expected_current_values)
            if summary != expected:
                issues.append(
                    ValidationIssue(
                        "P4_SUMMARY_STATE_MISMATCH",
                        "P4 structured summary must equal current booking state",
                        ValidationSeverity.HARD,
                        ValidationSource.DETERMINISTIC,
                        ValidationScope.EXCHANGE,
                        exchange_number=planned.exchange_number,
                        expected=expected,
                        observed=summary,
                    )
                )
        return ValidationReport(f"{self.__class__.__name__}.exchange", not issues, tuple(issues))
