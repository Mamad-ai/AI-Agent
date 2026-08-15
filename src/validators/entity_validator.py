"""Validation of structured facts; natural-language semantics are out of scope."""

from typing import Any, Mapping

from src.models.dialogue import ConversationExchange, Dialogue
from src.models.enums import ValidationScope, ValidationSeverity, ValidationSource
from src.models.plan import PlannedExchange, TurnPlan
from src.models.state import BookingState, SELECTED_SERVICES_FIELD
from src.models.template import ConversationTemplate, thaw_value
from src.models.validation import ValidationIssue, ValidationReport


class EntityValidator:
    def validate(
        self,
        dialogue: Dialogue,
        plan: TurnPlan,
        template: ConversationTemplate,
        state: BookingState,
    ) -> ValidationReport:
        issues: list[ValidationIssue] = []
        expressed_in_dialogue: set[str] = set()
        for number, planned in enumerate(plan.exchanges, start=1):
            if number > len(dialogue.exchanges):
                continue
            generated = dialogue.exchanges[number - 1]
            issues.extend(self.validate_exchange(generated, planned).issues)
            expressed_in_dialogue.update(
                self.correctly_expressed_fields(generated, planned, template.required_fields)
            )

        for field_name in template.required_fields:
            if isinstance(field_name, str) and (
                field_name not in state.current_values
                or self.is_structurally_missing(state.current_values[field_name])
            ):
                self._add(
                    issues,
                    "MISSING_REQUIRED_VALUE",
                    f"required value {field_name!r} is missing from current state",
                    field_name=field_name,
                )
            elif isinstance(field_name, str) and field_name not in expressed_in_dialogue:
                self._add(
                    issues,
                    "REQUIRED_FIELD_NOT_EXPRESSED",
                    f"required value {field_name!r} was not expressed in structured facts",
                    field_name=field_name,
                )

        services = state.current_values.get(SELECTED_SERVICES_FIELD, ())
        if isinstance(services, (list, tuple)):
            if len(services) > 4:
                self._add(issues, "TOO_MANY_SERVICES", "booking contains more than four services")
            if template.available_services:
                for service in services:
                    if service not in template.available_services:
                        self._add(
                            issues,
                            "SERVICE_NOT_AVAILABLE",
                            f"service {service!r} is not in validation metadata",
                            field_name=SELECTED_SERVICES_FIELD,
                        )
        return ValidationReport(self.__class__.__name__, not issues, tuple(issues))

    def validate_exchange(
        self, generated: ConversationExchange, planned: PlannedExchange
    ) -> ValidationReport:
        """Validate structured fact metadata for one generated exchange."""

        issues: list[ValidationIssue] = []
        self._check_fact_contract(
            planned.required_customer_facts,
            planned.allowed_customer_facts,
            generated.customer_facts,
            planned.exchange_number,
            "customer",
            issues,
        )
        self._check_fact_contract(
            planned.required_assistant_facts,
            planned.allowed_assistant_facts,
            generated.assistant_facts,
            planned.exchange_number,
            "assistant",
            issues,
        )
        return ValidationReport(f"{self.__class__.__name__}.exchange", not issues, tuple(issues))

    def correctly_expressed_fields(
        self,
        generated: ConversationExchange,
        planned: PlannedExchange,
        required_fields: Any,
    ) -> set[str]:
        """Find non-empty required fields matching an allowed planned value."""

        expressed: set[str] = set()
        fact_sets = (
            (generated.customer_facts, planned.allowed_customer_facts),
            (generated.assistant_facts, planned.allowed_assistant_facts),
        )
        for field_name in required_fields:
            if not isinstance(field_name, str):
                continue
            for observed, allowed in fact_sets:
                if (
                    field_name in observed
                    and field_name in allowed
                    and not self.is_structurally_missing(observed[field_name])
                    and thaw_value(observed[field_name]) == thaw_value(allowed[field_name])
                ):
                    expressed.add(field_name)
                    break
        return expressed

    def _check_fact_contract(
        self,
        expected: Mapping[str, Any],
        allowed: Mapping[str, Any],
        observed: Mapping[str, Any],
        number: int,
        owner: str,
        issues: list[ValidationIssue],
    ) -> None:
        for field_name, expected_value in expected.items():
            if field_name not in observed:
                self._add(
                    issues,
                    "MISSING_STRUCTURED_FACT",
                    f"{owner} fact {field_name!r} is missing",
                    number,
                    field_name,
                )
            elif thaw_value(observed[field_name]) != thaw_value(expected_value):
                self._add(
                    issues,
                    "STRUCTURED_FACT_CHANGED",
                    f"{owner} fact {field_name!r} changed",
                    number,
                    field_name,
                )

        for field_name, observed_value in observed.items():
            if field_name not in allowed:
                self._add(
                    issues,
                    "UNPLANNED_STRUCTURED_FACT",
                    f"{owner} fact {field_name!r} is not allowed by the plan",
                    number,
                    field_name,
                )
            elif thaw_value(observed_value) != thaw_value(allowed[field_name]):
                self._add(
                    issues,
                    "ALLOWED_FACT_VALUE_CHANGED",
                    f"{owner} fact {field_name!r} differs from its allowed value",
                    number,
                    field_name,
                )

    @staticmethod
    def is_structurally_missing(value: Any) -> bool:
        return value is None or value == "" or (
            isinstance(value, (list, tuple)) and len(value) == 0
        )

    @staticmethod
    def _add(
        issues: list[ValidationIssue],
        code: str,
        message: str,
        number: int | None = None,
        field_name: str | None = None,
    ) -> None:
        issues.append(
            ValidationIssue(
                code,
                message,
                ValidationSeverity.HARD,
                ValidationSource.DETERMINISTIC,
                ValidationScope.EXCHANGE if number else ValidationScope.STATE,
                exchange_number=number,
                field_name=field_name,
            )
        )
