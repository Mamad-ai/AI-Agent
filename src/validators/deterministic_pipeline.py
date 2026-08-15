"""Stable, non-mutating deterministic validation pipeline."""

from dataclasses import dataclass
from typing import Any, Mapping, Tuple

from src.models.dialogue import ConversationExchange, Dialogue
from src.models.enums import ValidationSeverity
from src.models.plan import PlannedExchange, TurnPlan
from src.models.state import BookingState
from src.models.template import ConversationTemplate
from src.models.validation import ValidationIssue, ValidationReport
from src.validators.entity_validator import EntityValidator
from src.validators.mlp_structure_validator import MLPStructureValidator
from src.validators.state_validator import StateValidator
from src.validators.structure_validator import StructureValidator
from src.validators.summary_validator import SummaryValidator


@dataclass(frozen=True)
class DeterministicValidationResult:
    reports: Tuple[ValidationReport, ...]

    @property
    def issues(self) -> Tuple[ValidationIssue, ...]:
        return tuple(issue for report in self.reports for issue in report.issues)

    @property
    def has_hard_failures(self) -> bool:
        return any(issue.severity is ValidationSeverity.HARD for issue in self.issues)

    @property
    def passed(self) -> bool:
        return all(report.passed for report in self.reports) and not self.has_hard_failures

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "has_hard_failures": self.has_hard_failures,
            "reports": [report.to_dict() for report in self.reports],
            "issues": [issue.to_dict() for issue in self.issues],
        }


class DeterministicValidationPipeline:
    def __init__(self, mlp_taxonomy: dict[str, frozenset[str]]) -> None:
        self._structure = StructureValidator()
        self._state = StateValidator()
        self._entity = EntityValidator()
        self._mlp = MLPStructureValidator(mlp_taxonomy)
        self._summary = SummaryValidator()

    def validate_exchange(
        self,
        generated: ConversationExchange,
        planned: PlannedExchange,
        expected_current_values: Mapping[str, Any],
    ) -> DeterministicValidationResult:
        """Validate an exchange before it is accepted for continued generation."""

        return DeterministicValidationResult(
            (
                self._entity.validate_exchange(generated, planned),
                self._mlp.validate_exchange(generated, planned),
                self._summary.validate_exchange(
                    generated, planned, expected_current_values
                ),
            )
        )

    def correctly_expressed_required_fields(
        self,
        generated: ConversationExchange,
        planned: PlannedExchange,
        required_fields: Any,
    ) -> set[str]:
        return self._entity.correctly_expressed_fields(generated, planned, required_fields)

    def validate(
        self,
        dialogue: Dialogue,
        plan: TurnPlan,
        template: ConversationTemplate,
        state: BookingState,
    ) -> DeterministicValidationResult:
        reports = (
            self._structure.validate(dialogue, plan, template),
            self._state.validate(plan, template, state),
            self._entity.validate(dialogue, plan, template, state),
            self._mlp.validate(dialogue, plan),
            self._summary.validate(dialogue, plan, template),
        )
        return DeterministicValidationResult(reports)
