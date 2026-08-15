import json
import unittest

from src.agent.classifier import ConversationClassifier
from src.agent.semantic_validation_service import SemanticValidationResult
from src.ai.schemas import (
    SemanticDialogueValidationResponse,
    SemanticTurnValidationResponse,
)
from src.models.classification import (
    ClassificationConfig,
    ClassificationReasonCode,
)
from src.models.enums import (
    Classification,
    ValidationScope,
    ValidationSeverity,
    ValidationSource,
)
from src.models.validation import ValidationIssue, ValidationReport
from src.validators.deterministic_pipeline import DeterministicValidationResult


CONFIG = ClassificationConfig("test", safe_min_confidence=0.85, weak_min_confidence=0.60)


def semantic_issue(severity: ValidationSeverity) -> ValidationIssue:
    return ValidationIssue(
        code=f"SEMANTIC_{severity.value.upper()}",
        message="semantic test issue",
        severity=severity,
        source=ValidationSource.SEMANTIC_AI,
        scope=ValidationScope.DIALOGUE,
    )


def deterministic_pass() -> DeterministicValidationResult:
    return DeterministicValidationResult((ValidationReport("deterministic", True),))


def deterministic_hard_failure() -> DeterministicValidationResult:
    issue = ValidationIssue(
        "DET_HARD",
        "hard deterministic failure",
        ValidationSeverity.HARD,
        ValidationSource.DETERMINISTIC,
        ValidationScope.DIALOGUE,
    )
    return DeterministicValidationResult((ValidationReport("deterministic", False, (issue,)),))


def turn_response(
    confidence: float = 0.9,
    naturalness: bool = True,
    issues: tuple[ValidationIssue, ...] = (),
    **overrides: bool,
) -> SemanticTurnValidationResponse:
    values = {
        "language_profile_match": True,
        "mlp1_semantic_match": True,
        "mlp2_semantic_match": True,
        "facts_semantically_preserved": True,
        "naturalness": naturalness,
        "contradiction_found": False,
        "unnecessary_question_found": False,
        "overall_pass": True,
    }
    values.update(overrides)
    if any(
        (
            not values["language_profile_match"],
            not values["mlp1_semantic_match"],
            not values["mlp2_semantic_match"],
            not values["facts_semantically_preserved"],
            values["contradiction_found"],
            values["unnecessary_question_found"],
        )
    ):
        values["overall_pass"] = False
    return SemanticTurnValidationResponse(
        confidence=confidence, issues=issues, **values  # type: ignore[arg-type]
    )


def dialogue_response(
    confidence: float = 0.9,
    naturalness: bool = True,
    issues: tuple[ValidationIssue, ...] = (),
    **overrides: bool,
) -> SemanticDialogueValidationResponse:
    values = {
        "global_language_profile_match": True,
        "naturalness": naturalness,
        "coherence": True,
        "contradiction_found": False,
        "state_facts_consistent": True,
        "pd_intention_followed": True,
        "overall_pass": True,
    }
    values.update(overrides)
    if any(
        (
            not values["global_language_profile_match"],
            not values["coherence"],
            values["contradiction_found"],
            not values["state_facts_consistent"],
            not values["pd_intention_followed"],
        )
    ):
        values["overall_pass"] = False
    return SemanticDialogueValidationResponse(
        confidence=confidence, issues=issues, **values  # type: ignore[arg-type]
    )


def semantic_result(
    turn: SemanticTurnValidationResponse | None = None,
    dialogue: SemanticDialogueValidationResponse | None = None,
) -> SemanticValidationResult:
    return SemanticValidationResult(
        (turn or turn_response(),), dialogue or dialogue_response()
    )


def classify(
    semantic: SemanticValidationResult,
    deterministic: DeterministicValidationResult | None = None,
    config: ClassificationConfig = CONFIG,
):
    return ConversationClassifier().classify(
        deterministic or deterministic_pass(), semantic, config
    )


class ConversationClassifierTests(unittest.TestCase):
    def test_clean_high_confidence_is_safe(self) -> None:
        result = classify(semantic_result())
        self.assertIs(result.classification, Classification.SAFE)
        self.assertFalse(result.requires_audit)
        self.assertEqual(
            result.reasons[0].code, ClassificationReasonCode.CLEAN_HIGH_CONFIDENCE
        )

    def test_deterministic_and_semantic_hard_failures_are_error(self) -> None:
        deterministic = classify(semantic_result(), deterministic_hard_failure())
        semantic = classify(
            semantic_result(
                turn_response(issues=(semantic_issue(ValidationSeverity.HARD),))
            )
        )
        self.assertIs(deterministic.classification, Classification.ERROR)
        self.assertIs(semantic.classification, Classification.ERROR)
        self.assertTrue(deterministic.requires_audit)
        self.assertTrue(semantic.requires_audit)

    def test_turn_blocking_failures_are_error(self) -> None:
        cases = (
            {"mlp1_semantic_match": False},
            {"facts_semantically_preserved": False},
            {"contradiction_found": True},
        )
        for override in cases:
            with self.subTest(override=override):
                result = classify(semantic_result(turn_response(**override)))
                self.assertIs(result.classification, Classification.ERROR)
                self.assertIn(
                    ClassificationReasonCode.SEMANTIC_BLOCKING_FAILURE,
                    {reason.code for reason in result.reasons},
                )

    def test_dialogue_blocking_failures_are_error(self) -> None:
        cases = ({"coherence": False}, {"pd_intention_followed": False})
        for override in cases:
            with self.subTest(override=override):
                result = classify(
                    semantic_result(dialogue=dialogue_response(**override))
                )
                self.assertIs(result.classification, Classification.ERROR)

    def test_low_naturalness_is_weak_when_confidence_is_acceptable(self) -> None:
        result = classify(semantic_result(turn_response(naturalness=False)))
        self.assertIs(result.classification, Classification.WEAK)
        self.assertTrue(result.requires_audit)
        self.assertIn(
            ClassificationReasonCode.LOW_NATURALNESS,
            {reason.code for reason in result.reasons},
        )

    def test_quality_and_uncertain_issues_are_weak(self) -> None:
        quality = classify(
            semantic_result(
                turn_response(issues=(semantic_issue(ValidationSeverity.QUALITY),))
            )
        )
        uncertain = classify(
            semantic_result(
                turn_response(issues=(semantic_issue(ValidationSeverity.UNCERTAIN),))
            )
        )
        self.assertIs(quality.classification, Classification.WEAK)
        self.assertIs(uncertain.classification, Classification.WEAK)
        self.assertEqual(quality.quality_issue_count, 1)
        self.assertEqual(uncertain.uncertain_issue_count, 1)

    def test_confidence_threshold_boundaries(self) -> None:
        cases = (
            (0.85, Classification.SAFE),
            (0.849, Classification.WEAK),
            (0.60, Classification.WEAK),
            (0.599, Classification.ERROR),
        )
        for confidence, expected in cases:
            with self.subTest(confidence=confidence):
                result = classify(
                    semantic_result(
                        turn_response(confidence), dialogue_response(confidence)
                    )
                )
                self.assertIs(result.classification, expected)

    def test_invalid_thresholds_are_rejected(self) -> None:
        invalid = (
            (0.9, 0.95),
            (1.1, 0.6),
            (0.8, -0.1),
        )
        for safe, weak in invalid:
            with self.subTest(safe=safe, weak=weak):
                with self.assertRaises(ValueError):
                    ClassificationConfig("bad", safe, weak)

    def test_default_config_loads_and_is_versioned(self) -> None:
        config = ClassificationConfig.load()
        self.assertEqual(config.version, "1")
        self.assertEqual(config.safe_min_confidence, 0.85)
        self.assertEqual(config.weak_min_confidence, 0.60)

    def test_reasons_are_stable_and_result_is_json_serializable(self) -> None:
        result = classify(
            semantic_result(turn_response(confidence=0.7, naturalness=False))
        )
        codes = [reason.code.value for reason in result.reasons]
        self.assertIn("LOW_NATURALNESS", codes)
        self.assertIn("CONFIDENCE_BELOW_SAFE_THRESHOLD", codes)
        json.dumps(result.to_dict(), ensure_ascii=False)

    def test_classifier_does_not_mutate_inputs_or_call_workers(self) -> None:
        deterministic = deterministic_pass()
        semantic = semantic_result()
        deterministic_before = deterministic.to_dict()
        semantic_before = semantic.to_dict()
        classifier = ConversationClassifier()

        result = classifier.classify(deterministic, semantic, CONFIG)

        self.assertIs(result.classification, Classification.SAFE)
        self.assertEqual(deterministic.to_dict(), deterministic_before)
        self.assertEqual(semantic.to_dict(), semantic_before)


if __name__ == "__main__":
    unittest.main()
