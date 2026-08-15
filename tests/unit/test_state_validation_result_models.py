import json
import unittest

from src.models.dialogue import ConversationExchange
from src.models.enums import (
    Classification,
    DeviationType,
    ProcessingStatus,
    ValidationScope,
    ValidationSeverity,
    ValidationSource,
)
from src.models.result import ProcessingResult
from src.models.state import BookingState, ProcessingState
from src.models.validation import ValidationIssue, ValidationReport


class StateTests(unittest.TestCase):
    def test_original_and_current_values_are_separate(self) -> None:
        original = {
            "day": "måndag",
            "time": "10:00",
            "selected_services": ["klippning"],
        }
        booking = BookingState(original_values=original, current_values=original)
        booking.current_values["day"] = "tisdag"
        booking.current_values["selected_services"].append("färgning")
        booking.used_deviations.append(DeviationType.D2)

        self.assertEqual(booking.original_values["day"], "måndag")
        self.assertEqual(booking.current_values["day"], "tisdag")
        self.assertEqual(booking.original_values["selected_services"], ("klippning",))
        self.assertEqual(booking.selected_services, ("klippning", "färgning"))
        self.assertNotIn("selected_services", booking.to_dict())

    def test_processing_state_tracks_complete_turns(self) -> None:
        state = ProcessingState(
            batch_id="batch-1",
            template_id="template-1",
            total_turns=14,
            completed_turns=3,
            booking=BookingState({}, {}),
            source_hash="sha256:template",
            config_version="config-v2",
            prompt_version="prompt-v3",
            model_version="model-v4",
            status=ProcessingStatus.GENERATED,
        )
        self.assertEqual(state.remaining_turns, 11)
        self.assertIsInstance(state.generated_exchanges, list[ConversationExchange].__origin__)
        payload = state.to_dict()
        self.assertEqual(payload["source_hash"], "sha256:template")
        self.assertEqual(payload["config_version"], "config-v2")
        self.assertEqual(payload["prompt_version"], "prompt-v3")
        self.assertEqual(payload["model_version"], "model-v4")
        json.dumps(payload, ensure_ascii=False)


class ValidationAndResultTests(unittest.TestCase):
    def test_failed_report_requires_structured_issue(self) -> None:
        issue = ValidationIssue(
            code="POOL_VALUE_CHANGED",
            message="A template value changed",
            severity=ValidationSeverity.HARD,
            source=ValidationSource.DETERMINISTIC,
            scope=ValidationScope.STATE,
            field_name="day",
            expected="måndag",
            observed="tisdag",
        )
        report = ValidationReport("entity_validator", passed=False, issues=(issue,))
        self.assertTrue(report.has_hard_failures)

    def test_result_is_json_serializable(self) -> None:
        result = ProcessingResult(
            template_id="template-1",
            classification=Classification.SAFE,
            booking_state=BookingState({"day": "måndag"}, {"day": "måndag"}),
            validation_reports=(ValidationReport("validator", passed=True),),
            generation_attempts=1,
        )
        payload = result.to_dict()
        self.assertEqual(payload["classification"], "SAFE")
        json.dumps(payload, ensure_ascii=False)

    def test_passed_report_cannot_contain_issue(self) -> None:
        issue = ValidationIssue(
            code="QUALITY",
            message="Uncertain style",
            severity=ValidationSeverity.UNCERTAIN,
            source=ValidationSource.SEMANTIC_AI,
            scope=ValidationScope.DIALOGUE,
        )
        with self.assertRaises(ValueError):
            ValidationReport("semantic", passed=True, issues=(issue,))


if __name__ == "__main__":
    unittest.main()
