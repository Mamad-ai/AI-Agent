"""Deterministic SAFE/WEAK/ERROR classification."""

from typing import Iterable

from src.agent.semantic_validation_service import SemanticValidationResult
from src.ai.schemas import (
    SemanticDialogueValidationResponse,
    SemanticTurnValidationResponse,
)
from src.models.classification import (
    ClassificationConfig,
    ClassificationReason,
    ClassificationReasonCode,
    ClassificationResult,
)
from src.models.enums import Classification, ValidationSeverity
from src.validators.deterministic_pipeline import DeterministicValidationResult


class ConversationClassifier:
    def classify(
        self,
        deterministic: DeterministicValidationResult,
        semantic: SemanticValidationResult,
        config: ClassificationConfig,
    ) -> ClassificationResult:
        deterministic_hard = sum(
            issue.severity is ValidationSeverity.HARD for issue in deterministic.issues
        )
        semantic_hard = sum(
            issue.severity is ValidationSeverity.HARD for issue in semantic.issues
        )
        quality_count = sum(
            issue.severity is ValidationSeverity.QUALITY for issue in semantic.issues
        )
        uncertain_count = sum(
            issue.severity is ValidationSeverity.UNCERTAIN for issue in semantic.issues
        )
        minimum_confidence = semantic.minimum_confidence
        reasons: list[ClassificationReason] = []

        if deterministic_hard:
            reasons.append(
                self._reason(
                    ClassificationReasonCode.DETERMINISTIC_HARD_FAILURE,
                    "Deterministic validation contains HARD failures.",
                )
            )
        elif not deterministic.passed:
            reasons.append(
                self._reason(
                    ClassificationReasonCode.DETERMINISTIC_VALIDATION_FAILED,
                    "Deterministic validation did not pass.",
                )
            )
        if semantic_hard:
            reasons.append(
                self._reason(
                    ClassificationReasonCode.SEMANTIC_HARD_FAILURE,
                    "Semantic validation contains HARD failures.",
                )
            )
        blocking = self._has_blocking_semantic_failure(semantic)
        if blocking:
            reasons.append(
                self._reason(
                    ClassificationReasonCode.SEMANTIC_BLOCKING_FAILURE,
                    "Semantic response fields contain a blocking failure.",
                )
            )
        if minimum_confidence < config.weak_min_confidence:
            reasons.append(
                self._reason(
                    ClassificationReasonCode.CONFIDENCE_BELOW_WEAK_THRESHOLD,
                    "Minimum semantic confidence is below the acceptable threshold.",
                )
            )

        error = (
            deterministic_hard > 0
            or not deterministic.passed
            or semantic_hard > 0
            or blocking
            or minimum_confidence < config.weak_min_confidence
        )
        if error:
            return self._result(
                Classification.ERROR,
                reasons,
                deterministic,
                semantic,
                deterministic_hard,
                semantic_hard,
                quality_count,
                uncertain_count,
                minimum_confidence,
            )

        low_naturalness = any(
            not response.naturalness
            for response in (*semantic.turn_responses, semantic.dialogue_response)
        )
        if quality_count:
            reasons.append(
                self._reason(
                    ClassificationReasonCode.QUALITY_ISSUES_PRESENT,
                    "Semantic QUALITY issues are present.",
                )
            )
        if uncertain_count:
            reasons.append(
                self._reason(
                    ClassificationReasonCode.UNCERTAIN_ISSUES_PRESENT,
                    "Semantic UNCERTAIN issues are present.",
                )
            )
        if low_naturalness:
            reasons.append(
                self._reason(
                    ClassificationReasonCode.LOW_NATURALNESS,
                    "At least one semantic response reports low naturalness.",
                )
            )
        if minimum_confidence < config.safe_min_confidence:
            reasons.append(
                self._reason(
                    ClassificationReasonCode.CONFIDENCE_BELOW_SAFE_THRESHOLD,
                    "Minimum semantic confidence is below the SAFE threshold.",
                )
            )

        weak = quality_count > 0 or uncertain_count > 0 or low_naturalness or (
            minimum_confidence < config.safe_min_confidence
        )
        if weak:
            return self._result(
                Classification.WEAK,
                reasons,
                deterministic,
                semantic,
                deterministic_hard,
                semantic_hard,
                quality_count,
                uncertain_count,
                minimum_confidence,
            )

        reasons.append(
            self._reason(
                ClassificationReasonCode.CLEAN_HIGH_CONFIDENCE,
                "Validation is clean and confidence meets the SAFE threshold.",
            )
        )
        return self._result(
            Classification.SAFE,
            reasons,
            deterministic,
            semantic,
            deterministic_hard,
            semantic_hard,
            quality_count,
            uncertain_count,
            minimum_confidence,
        )

    def _has_blocking_semantic_failure(self, semantic: SemanticValidationResult) -> bool:
        for response in semantic.turn_responses:
            if self._turn_is_blocking(response):
                return True
        return self._dialogue_is_blocking(semantic.dialogue_response)

    @staticmethod
    def _turn_is_blocking(response: SemanticTurnValidationResponse) -> bool:
        explicit = (
            not response.language_profile_match
            or not response.mlp1_semantic_match
            or not response.mlp2_semantic_match
            or not response.facts_semantically_preserved
            or response.contradiction_found
            or response.unnecessary_question_found
        )
        return explicit or ConversationClassifier._unexplained_failure_is_blocking(response)

    @staticmethod
    def _dialogue_is_blocking(response: SemanticDialogueValidationResponse) -> bool:
        explicit = (
            not response.global_language_profile_match
            or not response.coherence
            or response.contradiction_found
            or not response.state_facts_consistent
            or not response.pd_intention_followed
        )
        return explicit or ConversationClassifier._unexplained_failure_is_blocking(response)

    @staticmethod
    def _unexplained_failure_is_blocking(response: object) -> bool:
        if getattr(response, "overall_pass"):
            return False
        if not getattr(response, "naturalness"):
            return False
        issues: Iterable = getattr(response, "issues")
        return not issues or any(issue.severity is ValidationSeverity.HARD for issue in issues)

    @staticmethod
    def _reason(code: ClassificationReasonCode, message: str) -> ClassificationReason:
        return ClassificationReason(code, message)

    @staticmethod
    def _result(
        classification: Classification,
        reasons: list[ClassificationReason],
        deterministic: DeterministicValidationResult,
        semantic: SemanticValidationResult,
        deterministic_hard: int,
        semantic_hard: int,
        quality_count: int,
        uncertain_count: int,
        minimum_confidence: float,
    ) -> ClassificationResult:
        return ClassificationResult(
            classification=classification,
            reasons=tuple(reasons),
            deterministic_pass=deterministic.passed,
            semantic_pass=semantic.overall_semantic_pass,
            deterministic_hard_failures=deterministic_hard,
            semantic_hard_failures=semantic_hard,
            quality_issue_count=quality_count,
            uncertain_issue_count=uncertain_count,
            minimum_semantic_confidence=minimum_confidence,
            requires_audit=classification is not Classification.SAFE,
        )
