"""Deterministic validation of completed conversation templates."""

from collections import Counter
from typing import Any, Iterable, Optional

from src.models.enums import (
    DeviationType,
    Phase,
    PhaseVariant,
    StateChangeSource,
    StateOperation,
    ValidationScope,
    ValidationSeverity,
    ValidationSource,
)
from src.models.state import SELECTED_SERVICES_FIELD
from src.models.template import (
    ConversationTemplate,
    DeviationAssignment,
    TemplateExchangeRequirements,
    TemplateStateChange,
    thaw_value,
)
from src.models.validation import ValidationIssue, ValidationReport


class TemplateValidator:
    """Validate structure and hard P/D rules without mutating the template."""

    _P3_BOOKING_FIELDS = frozenset(
        {"booking_intent", "intent", "selected_services", "services", "day", "time"}
    )
    _STATEFUL_DEVIATION_PHASES = frozenset({Phase.P3, Phase.P4, Phase.P5})
    _ALLOWED_P1_P2_LINKS = frozenset(
        {
            (PhaseVariant.P1_2, PhaseVariant.P2_2),
            (PhaseVariant.P1_2, PhaseVariant.P2_3),
            (PhaseVariant.P1_2, PhaseVariant.P2_4),
            (PhaseVariant.P1_3, PhaseVariant.P2_1),
            (PhaseVariant.P1_3, PhaseVariant.P2_2),
            (PhaseVariant.P1_3, PhaseVariant.P2_3),
            (PhaseVariant.P1_3, PhaseVariant.P2_4),
            (PhaseVariant.P1_4, PhaseVariant.P2_2),
            (PhaseVariant.P1_4, PhaseVariant.P2_3),
        }
    )
    _MAX_DEVIATIONS = 2
    _MAX_SERVICES = 4

    def validate(self, template: ConversationTemplate) -> ValidationReport:
        issues: list[ValidationIssue] = []
        self._validate_structure(template, issues)
        self._validate_required_source_data(template, issues)
        self._validate_exchange_rules(template, issues)
        return ValidationReport(
            validator_name=self.__class__.__name__,
            passed=not issues,
            issues=tuple(issues),
        )

    def _validate_structure(
        self, template: ConversationTemplate, issues: list[ValidationIssue]
    ) -> None:
        if template.total_turns <= 0:
            self._issue(issues, "INVALID_TOTAL_TURNS", "total_turns must be greater than zero")
        if len(template.phase_by_exchange) != template.total_turns:
            self._issue(issues, "PHASE_COUNT_MISMATCH", "exactly one phase is required per exchange")
        if len(template.phase_variant_by_exchange) != template.total_turns:
            self._issue(
                issues,
                "PHASE_VARIANT_COUNT_MISMATCH",
                "exactly one phase variant is required per exchange",
            )
        for number, (phase, variant) in enumerate(
            zip(template.phase_by_exchange, template.phase_variant_by_exchange), start=1
        ):
            if not variant.belongs_to(phase):
                self._issue(
                    issues,
                    "PHASE_VARIANT_MISMATCH",
                    f"{variant.value} does not belong to {phase.value}",
                    number,
                )

        self._validate_numbered_items(
            template.deviations,
            template.total_turns,
            "deviation",
            issues,
        )
        if len(template.deviations) > self._MAX_DEVIATIONS:
            self._issue(
                issues,
                "TOO_MANY_DEVIATIONS",
                "a conversation may contain at most two deviations",
            )
        for deviation in template.deviations:
            if 1 <= deviation.exchange_number <= len(template.phase_by_exchange):
                actual_phase = template.phase_by_exchange[deviation.exchange_number - 1]
                if deviation.phase is not actual_phase:
                    self._issue(
                        issues,
                        "DEVIATION_PHASE_MISMATCH",
                        "deviation phase does not match its exchange phase",
                        deviation.exchange_number,
                    )
                if (
                    deviation.deviation_type is not DeviationType.D1
                    and actual_phase not in self._STATEFUL_DEVIATION_PHASES
                ):
                    self._issue(
                        issues,
                        "INVALID_DEVIATION_PLACEMENT",
                        f"{deviation.deviation_type.value} is not allowed in {actual_phase.value}",
                        deviation.exchange_number,
                    )
        self._validate_numbered_items(
            template.mlp_expectations,
            template.total_turns,
            "optional semantic-action hint",
            issues,
        )
        self._validate_numbered_items(
            template.planned_semantic_actions,
            template.total_turns,
            "planned semantic action",
            issues,
        )
        self._validate_p1_p2_links(template, issues)

    def _validate_p1_p2_links(
        self, template: ConversationTemplate, issues: list[ValidationIssue]
    ) -> None:
        pairs = zip(
            template.phase_by_exchange,
            template.phase_variant_by_exchange,
        )
        phase_plan = tuple(pairs)
        for index in range(len(phase_plan) - 1):
            phase, variant = phase_plan[index]
            next_phase, next_variant = phase_plan[index + 1]
            if (
                phase is Phase.P1
                and next_phase is Phase.P2
                and (variant, next_variant) not in self._ALLOWED_P1_P2_LINKS
            ):
                self._issue(
                    issues,
                    "INVALID_P1_P2_LINK",
                    f"{variant.value} may not transition to {next_variant.value}",
                    index + 2,
                )
        self._validate_numbered_items(
            template.exchange_requirements,
            template.total_turns,
            "exchange requirements",
            issues,
            require_every_exchange=True,
        )

    def _validate_numbered_items(
        self,
        items: Iterable[Any],
        total_turns: int,
        label: str,
        issues: list[ValidationIssue],
        require_every_exchange: bool = False,
    ) -> None:
        numbers = [item.exchange_number for item in items]
        for number in numbers:
            if number < 1 or number > total_turns:
                self._issue(
                    issues,
                    "INVALID_EXCHANGE_REFERENCE",
                    f"{label} references invalid exchange {number}",
                    number if number > 0 else None,
                )
        duplicates = [number for number, count in Counter(numbers).items() if count > 1]
        for number in duplicates:
            self._issue(
                issues,
                "DUPLICATE_EXCHANGE_ASSIGNMENT",
                f"multiple {label} entries reference exchange {number}",
                number,
            )
        if require_every_exchange and set(numbers) != set(range(1, total_turns + 1)):
            self._issue(
                issues,
                "INCOMPLETE_EXCHANGE_ASSIGNMENT",
                f"{label} must contain exactly one entry per exchange",
            )

    def _validate_required_source_data(
        self, template: ConversationTemplate, issues: list[ValidationIssue]
    ) -> None:
        valid_names = [item for item in template.required_fields if isinstance(item, str)]
        if len(set(valid_names)) != len(valid_names):
            self._issue(issues, "DUPLICATE_REQUIRED_FIELD", "required_fields contains duplicates")
        for field_name in template.required_fields:
            if not isinstance(field_name, str) or not field_name.strip():
                self._issue(
                    issues, "INVALID_REQUIRED_FIELD", "required_fields must contain non-empty strings"
                )
            elif field_name not in template.original_values:
                self._issue(
                    issues,
                    "MISSING_SOURCE_VALUE",
                    f"required source-of-truth field {field_name!r} is missing",
                    field_name=field_name,
                )

    def _validate_exchange_rules(
        self, template: ConversationTemplate, issues: list[ValidationIssue]
    ) -> None:
        deviations = {item.exchange_number: item for item in template.deviations}
        requirements = {item.exchange_number: item for item in template.exchange_requirements}
        selected = template.original_values.get(SELECTED_SERVICES_FIELD, ())
        selected_services = list(selected) if isinstance(selected, (list, tuple)) else []
        if len(selected_services) > self._MAX_SERVICES:
            self._issue(
                issues,
                "TOO_MANY_SERVICES",
                "a booking may contain at most four selected services",
                field_name=SELECTED_SERVICES_FIELD,
            )

        for number in range(1, template.total_turns + 1):
            if number > len(template.phase_by_exchange) or number > len(
                template.phase_variant_by_exchange
            ):
                continue
            phase = template.phase_by_exchange[number - 1]
            variant = template.phase_variant_by_exchange[number - 1]
            deviation = deviations.get(number)
            requirement = requirements.get(number)
            changes = requirement.state_changes if requirement else ()
            phase_changes = tuple(
                change for change in changes if change.source is StateChangeSource.PHASE_VARIANT
            )
            deviation_changes = tuple(
                change for change in changes if change.source is StateChangeSource.DEVIATION
            )

            if requirement is not None:
                self._validate_fact_contract(number, requirement, issues)
            self._validate_phase_variant_rules(number, phase, variant, phase_changes, issues)
            if phase is Phase.P3:
                self._validate_p3(number, requirement, phase_changes, deviation, issues)
            if deviation:
                self._validate_deviation(
                    number,
                    deviation.deviation_type,
                    changes,
                    deviation_changes,
                    selected_services,
                    issues,
                )
            elif deviation_changes:
                self._issue(
                    issues,
                    "DEVIATION_CHANGE_WITHOUT_DEVIATION",
                    "deviation-sourced state change has no assigned deviation",
                    number,
                )

    def _validate_fact_contract(
        self,
        number: int,
        requirement: TemplateExchangeRequirements,
        issues: list[ValidationIssue],
    ) -> None:
        fact_sets = (
            (
                "customer",
                requirement.required_customer_facts,
                requirement.allowed_customer_facts,
            ),
            (
                "assistant",
                requirement.required_assistant_facts,
                requirement.allowed_assistant_facts,
            ),
        )
        for owner, required, allowed in fact_sets:
            for field_name, required_value in required.items():
                if field_name not in allowed or thaw_value(allowed[field_name]) != thaw_value(
                    required_value
                ):
                    self._issue(
                        issues,
                        "REQUIRED_FACT_NOT_ALLOWED",
                        f"required {owner} fact {field_name!r} must be allowed with the same value",
                        number,
                        field_name,
                    )
    def _validate_phase_variant_rules(
        self,
        number: int,
        phase: Phase,
        variant: PhaseVariant,
        changes: tuple[TemplateStateChange, ...],
        issues: list[ValidationIssue],
    ) -> None:
        if phase is not Phase.P4:
            return
        if variant is PhaseVariant.P4_1 and changes:
            self._issue(issues, "P4_1_HAS_CORRECTION", "P4.1 must not correct state", number)
        elif variant is PhaseVariant.P4_2:
            if len(changes) != 1 or any(
                change.operation is not StateOperation.REPLACE for change in changes
            ):
                self._issue(
                    issues,
                    "INVALID_P4_2_CORRECTION",
                    "P4.2 requires exactly one phase-variant REPLACE",
                    number,
                )
        elif variant is PhaseVariant.P4_3:
            if len(changes) < 2 or any(
                change.operation is not StateOperation.REPLACE for change in changes
            ) or len({change.field_name for change in changes}) < 2:
                self._issue(
                    issues,
                    "INVALID_P4_3_CORRECTIONS",
                    "P4.3 requires phase-variant REPLACE changes for at least two distinct fields",
                    number,
                )

    def _validate_p3(
        self,
        number: int,
        requirement: Optional[TemplateExchangeRequirements],
        phase_changes: tuple[TemplateStateChange, ...],
        deviation: Optional[DeviationAssignment],
        issues: list[ValidationIssue],
    ) -> None:
        for change in phase_changes:
            if change.field_name in self._P3_BOOKING_FIELDS:
                self._issue(
                    issues,
                    "P3_INTRODUCES_BOOKING_STATE",
                    f"P3 may not introduce {change.field_name!r} without a deviation",
                    number,
                    change.field_name,
                )
        has_stateful_deviation = (
            deviation is not None and deviation.deviation_type.changes_booking_state
        )
        if requirement is not None and not has_stateful_deviation:
            introduced = self._P3_BOOKING_FIELDS.intersection(
                requirement.required_customer_facts.keys()
            )
            for field_name in sorted(introduced):
                self._issue(
                    issues,
                    "P3_INTRODUCES_BOOKING_FACT",
                    f"P3 may not introduce {field_name!r} without a state-changing deviation",
                    number,
                    field_name,
                )

    def _validate_deviation(
        self,
        number: int,
        deviation: DeviationType,
        all_changes: tuple[TemplateStateChange, ...],
        changes: tuple[TemplateStateChange, ...],
        selected_services: list[Any],
        issues: list[ValidationIssue],
    ) -> None:
        if deviation is DeviationType.D1:
            if all_changes:
                self._issue(issues, "D1_CHANGES_STATE", "D1 must not change booking state", number)
            return
        if deviation is DeviationType.D2:
            valid = len(changes) == 1 and changes[0].operation is StateOperation.REPLACE
            if not valid:
                self._issue(
                    issues, "INVALID_D2_CHANGE", "D2 requires exactly one deviation REPLACE", number
                )
        elif deviation is DeviationType.D3:
            valid = (
                len(changes) == 1
                and changes[0].operation is StateOperation.ADD
                and changes[0].field_name == SELECTED_SERVICES_FIELD
            )
            if not valid:
                self._issue(
                    issues,
                    "INVALID_D3_CHANGE",
                    "D3 requires exactly one deviation ADD to selected_services",
                    number,
                )
            else:
                service = changes[0].value
                if service in selected_services:
                    self._issue(
                        issues,
                        "DUPLICATE_D3_SERVICE",
                        "D3 may not add an already selected service",
                        number,
                        SELECTED_SERVICES_FIELD,
                    )
                elif len(selected_services) >= self._MAX_SERVICES:
                    self._issue(
                        issues,
                        "TOO_MANY_SERVICES",
                        "D3 would exceed the maximum of four selected services",
                        number,
                        SELECTED_SERVICES_FIELD,
                    )
                else:
                    selected_services.append(service)
        elif deviation is DeviationType.D5:
            valid = len(changes) >= 2 and all(
                change.operation is StateOperation.REPLACE for change in changes
            )
            distinct_fields = {change.field_name for change in changes}
            valid = valid and len(distinct_fields) >= 2
            if not valid:
                self._issue(
                    issues,
                    "INVALID_D5_CHANGES",
                    "D5 requires deviation REPLACE changes for at least two distinct fields",
                    number,
                )

    @staticmethod
    def _issue(
        issues: list[ValidationIssue],
        code: str,
        message: str,
        exchange_number: Optional[int] = None,
        field_name: Optional[str] = None,
    ) -> None:
        issues.append(
            ValidationIssue(
                code=code,
                message=message,
                severity=ValidationSeverity.HARD,
                source=ValidationSource.DETERMINISTIC,
                scope=ValidationScope.TEMPLATE,
                exchange_number=exchange_number,
                field_name=field_name,
            )
        )
