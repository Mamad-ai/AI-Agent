"""Structural validation of a generated dialogue against its plan."""

from src.models.dialogue import Dialogue
from src.models.enums import (
    Speaker,
    ValidationScope,
    ValidationSeverity,
    ValidationSource,
)
from src.models.plan import TurnPlan
from src.models.template import ConversationTemplate
from src.models.validation import ValidationIssue, ValidationReport


class StructureValidator:
    def validate(
        self, dialogue: Dialogue, plan: TurnPlan, template: ConversationTemplate
    ) -> ValidationReport:
        issues: list[ValidationIssue] = []
        if dialogue.template_id != template.template_id or plan.template_id != template.template_id:
            self._add(issues, "TEMPLATE_ID_MISMATCH", "dialogue, plan and template IDs must match")
        if dialogue.language_profile is not template.language_profile:
            self._add(issues, "LANGUAGE_PROFILE_MISMATCH", "dialogue language profile changed")
        if dialogue.plan_version != plan.plan_version:
            self._add(issues, "PLAN_VERSION_MISMATCH", "dialogue and turn plan versions must match")
        if dialogue.total_turns != plan.total_turns:
            self._add(issues, "EXCHANGE_COUNT_MISMATCH", "dialogue must contain exactly planned turns")
        if dialogue.utterance_count != dialogue.total_turns * 2:
            self._add(issues, "UTTERANCE_COUNT_MISMATCH", "each turn must contain two utterances")

        for index, exchange in enumerate(dialogue.exchanges, start=1):
            if exchange.exchange_number != index:
                self._add(issues, "EXCHANGE_NUMBER_MISMATCH", "exchange number is not consecutive", index)
            if exchange.customer.speaker is not Speaker.CUSTOMER:
                self._add(issues, "CUSTOMER_SPEAKER_MISMATCH", "first utterance must be customer", index)
            if exchange.assistant.speaker is not Speaker.ASSISTANT:
                self._add(issues, "ASSISTANT_SPEAKER_MISMATCH", "second utterance must be assistant", index)
            if index <= len(plan.exchanges):
                planned = plan.exchanges[index - 1]
                if exchange.exchange_number != planned.exchange_number:
                    self._add(
                        issues,
                        "PLANNED_EXCHANGE_MISMATCH",
                        "generated exchange does not match planned exchange",
                        index,
                    )
                if exchange.phase is not planned.phase:
                    self._add(issues, "PHASE_MISMATCH", "generated phase changed", index)
                if exchange.phase_variant is not planned.phase_variant:
                    self._add(
                        issues, "PHASE_VARIANT_MISMATCH", "generated phase variant changed", index
                    )
        return ValidationReport(self.__class__.__name__, not issues, tuple(issues))

    @staticmethod
    def _add(issues: list[ValidationIssue], code: str, message: str, number: int | None = None) -> None:
        issues.append(
            ValidationIssue(
                code,
                message,
                ValidationSeverity.HARD,
                ValidationSource.DETERMINISTIC,
                ValidationScope.DIALOGUE,
                exchange_number=number,
            )
        )
