"""Structural MLP validation without interpreting natural-language text."""

from typing import AbstractSet, Mapping

from src.models.dialogue import ConversationExchange, Dialogue
from src.models.enums import ValidationScope, ValidationSeverity, ValidationSource
from src.models.plan import PlannedExchange, TurnPlan
from src.models.validation import ValidationIssue, ValidationReport


class MLPStructureValidator:
    def __init__(self, taxonomy: Mapping[str, AbstractSet[str]]) -> None:
        self._taxonomy = {mlp1: frozenset(mlp2_values) for mlp1, mlp2_values in taxonomy.items()}

    def validate(self, dialogue: Dialogue, plan: TurnPlan) -> ValidationReport:
        issues: list[ValidationIssue] = []
        for number, planned in enumerate(plan.exchanges, start=1):
            if number > len(dialogue.exchanges):
                continue
            generated = dialogue.exchanges[number - 1]
            issues.extend(self.validate_exchange(generated, planned).issues)
        return ValidationReport(self.__class__.__name__, not issues, tuple(issues))

    def validate_exchange(
        self, generated: ConversationExchange, planned: PlannedExchange
    ) -> ValidationReport:
        issues: list[ValidationIssue] = []
        number = planned.exchange_number
        mlp1 = generated.customer_mlp1
        mlp2 = generated.customer_mlp2
        if not mlp1 or mlp1 not in self._taxonomy:
            self._add(
                issues,
                "UNKNOWN_MLP1",
                "customer MLP1 is missing or not present in the taxonomy",
                number,
            )
        else:
            allowed_mlp2 = self._taxonomy[mlp1]
            if not allowed_mlp2 and mlp2 is not None:
                self._add(
                    issues,
                    "MLP2_FORBIDDEN",
                    "selected MLP1 requires customer_mlp2 to be None",
                    number,
                )
            elif allowed_mlp2 and mlp2 is None:
                self._add(
                    issues,
                    "MLP2_REQUIRED",
                    "selected MLP1 requires an MLP2 label",
                    number,
                )
            elif allowed_mlp2 and mlp2 not in allowed_mlp2:
                self._add(
                    issues,
                    "INVALID_MLP2_FOR_MLP1",
                    "customer MLP2 is not allowed under selected MLP1",
                    number,
                )
        return ValidationReport(f"{self.__class__.__name__}.exchange", not issues, tuple(issues))

    @staticmethod
    def _add(issues: list[ValidationIssue], code: str, message: str, number: int) -> None:
        issues.append(
            ValidationIssue(
                code,
                message,
                ValidationSeverity.HARD,
                ValidationSource.DETERMINISTIC,
                ValidationScope.EXCHANGE,
                exchange_number=number,
            )
        )
