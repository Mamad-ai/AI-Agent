import json
import unittest
from typing import Any

from src.agent.semantic_validation_service import (
    DeterministicValidationRequiredError,
    InvalidSemanticResponseError,
    SemanticDialogueValidatorError,
    SemanticTurnValidatorError,
    SemanticValidationResult,
    SemanticValidationService,
)
from src.ai.language_profiles import LanguageProfileLoader
from src.ai.schemas import (
    SemanticDialogueValidationResponse,
    SemanticTurnValidationResponse,
)
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
from src.models.dialogue import Dialogue
from src.models.validation import ValidationIssue
from src.validators.deterministic_pipeline import DeterministicValidationPipeline
from tests.fakes.fake_conversation_generator import FakeConversationGenerator
from tests.fakes.fake_semantic_validators import (
    FakeSemanticDialogueValidator,
    FakeSemanticTurnValidator,
)
from tests.unit.test_generation_service import (
    MLP_TAXONOMY,
    TAXONOMY,
    make_service,
    make_template_and_plan,
    response_for,
    state_change,
)


def passing_turn(confidence: float = 0.9, issues: tuple[ValidationIssue, ...] = ()) -> SemanticTurnValidationResponse:
    return SemanticTurnValidationResponse(
        True, True, True, True, True, False, False, True, confidence, issues
    )


def failing_turn(**overrides: Any) -> SemanticTurnValidationResponse:
    values = {
        "language_profile_match": True,
        "mlp1_semantic_match": True,
        "mlp2_semantic_match": True,
        "facts_semantically_preserved": True,
        "naturalness": True,
        "contradiction_found": False,
        "unnecessary_question_found": False,
        "overall_pass": False,
        "confidence": 0.7,
    }
    values.update(overrides)
    return SemanticTurnValidationResponse(**values)


def passing_dialogue(
    confidence: float = 0.85, issues: tuple[ValidationIssue, ...] = ()
) -> SemanticDialogueValidationResponse:
    return SemanticDialogueValidationResponse(
        True, True, True, False, True, True, True, confidence, issues
    )


def generated_scenario(
    phases: tuple[Phase, ...],
    variants: tuple[PhaseVariant, ...],
    **kwargs: Any,
) -> tuple[Any, Any, Any]:
    template, plan = make_template_and_plan(phases, variants, **kwargs)
    profile = LanguageProfileLoader().load(template.language_profile)
    generation = make_service(FakeConversationGenerator(response_factory=response_for)).generate(
        template, plan, profile
    )
    return template, plan, generation


def semantic_service(
    turn: FakeSemanticTurnValidator,
    dialogue: FakeSemanticDialogueValidator,
) -> SemanticValidationService:
    return SemanticValidationService(
        turn,
        dialogue,
        DeterministicValidationPipeline(TAXONOMY),
        MLP_TAXONOMY,
    )


class SemanticValidationServiceTests(unittest.TestCase):
    def test_current_deterministic_preflight_runs_before_semantic_workers(self) -> None:
        template, plan, generation = generated_scenario(
            (Phase.P2,), (PhaseVariant.P2_2,)
        )

        class CountingPipeline(DeterministicValidationPipeline):
            def __init__(self) -> None:
                super().__init__(TAXONOMY)
                self.validation_calls = 0

            def validate(self, *args: Any, **kwargs: Any) -> Any:
                self.validation_calls += 1
                return super().validate(*args, **kwargs)

        pipeline = CountingPipeline()
        turn_fake = FakeSemanticTurnValidator(outcomes=(passing_turn(),))
        dialogue_fake = FakeSemanticDialogueValidator(passing_dialogue())
        service = SemanticValidationService(turn_fake, dialogue_fake, pipeline, MLP_TAXONOMY)
        service.validate(
            generation.dialogue,
            plan,
            template,
            generation.final_booking_state,
            generation.deterministic_validation,
            LanguageProfileLoader().load(template.language_profile),
        )
        self.assertEqual(pipeline.validation_calls, 1)
        self.assertEqual(len(turn_fake.received_requests), 1)
        self.assertEqual(len(dialogue_fake.received_requests), 1)

    def test_deterministic_hard_failure_blocks_all_semantic_calls(self) -> None:
        template, plan, generation = generated_scenario(
            (Phase.P2,), (PhaseVariant.P2_2,)
        )
        object.__setattr__(generation.dialogue, "template_id", "wrong")
        deterministic = DeterministicValidationPipeline(TAXONOMY).validate(
            generation.dialogue, plan, template, generation.final_booking_state
        )
        turn_fake = FakeSemanticTurnValidator(response_factory=lambda _: passing_turn())
        dialogue_fake = FakeSemanticDialogueValidator(passing_dialogue())
        with self.assertRaises(DeterministicValidationRequiredError):
            semantic_service(turn_fake, dialogue_fake).validate(
                generation.dialogue,
                plan,
                template,
                generation.final_booking_state,
                deterministic,
                LanguageProfileLoader().load(template.language_profile),
            )
        self.assertEqual(turn_fake.received_requests, [])
        self.assertEqual(dialogue_fake.received_requests, [])

    def test_stale_pass_result_cannot_authorize_changed_dialogue(self) -> None:
        template, plan, generation = generated_scenario(
            (Phase.P2,), (PhaseVariant.P2_2,)
        )
        stale_pass = generation.deterministic_validation
        object.__setattr__(generation.dialogue, "template_id", "changed-after-validation")
        turn_fake = FakeSemanticTurnValidator(response_factory=lambda _: passing_turn())
        dialogue_fake = FakeSemanticDialogueValidator(passing_dialogue())
        with self.assertRaises(DeterministicValidationRequiredError):
            semantic_service(turn_fake, dialogue_fake).validate(
                generation.dialogue,
                plan,
                template,
                generation.final_booking_state,
                stale_pass,
                LanguageProfileLoader().load(template.language_profile),
            )
        self.assertEqual(turn_fake.received_requests, [])
        self.assertEqual(dialogue_fake.received_requests, [])

    def test_stale_pass_result_cannot_authorize_changed_final_state(self) -> None:
        template, plan, generation = generated_scenario(
            (Phase.P2,), (PhaseVariant.P2_2,)
        )
        stale_pass = generation.deterministic_validation
        generation.final_booking_state.current_values["day"] = "fredag"
        turn_fake = FakeSemanticTurnValidator(response_factory=lambda _: passing_turn())
        dialogue_fake = FakeSemanticDialogueValidator(passing_dialogue())
        with self.assertRaises(DeterministicValidationRequiredError):
            semantic_service(turn_fake, dialogue_fake).validate(
                generation.dialogue,
                plan,
                template,
                generation.final_booking_state,
                stale_pass,
                LanguageProfileLoader().load(template.language_profile),
            )
        self.assertEqual(turn_fake.received_requests, [])
        self.assertEqual(dialogue_fake.received_requests, [])

    def test_turn_count_mismatch_never_semantically_validates_a_zip_prefix(self) -> None:
        template, plan, generation = generated_scenario(
            (Phase.P2, Phase.P3, Phase.P5),
            (PhaseVariant.P2_2, PhaseVariant.P3_1, PhaseVariant.P5_1),
        )
        stale_pass = generation.deterministic_validation
        shortened = Dialogue(
            template.template_id,
            template.language_profile,
            generation.dialogue.exchanges[:2],
            plan_version=plan.plan_version,
        )
        turn_fake = FakeSemanticTurnValidator(response_factory=lambda _: passing_turn())
        dialogue_fake = FakeSemanticDialogueValidator(passing_dialogue())
        with self.assertRaises(DeterministicValidationRequiredError):
            semantic_service(turn_fake, dialogue_fake).validate(
                shortened,
                plan,
                template,
                generation.final_booking_state,
                stale_pass,
                LanguageProfileLoader().load(template.language_profile),
            )
        self.assertEqual(turn_fake.received_requests, [])
        self.assertEqual(dialogue_fake.received_requests, [])

    def test_three_turns_run_in_order_and_dialogue_once(self) -> None:
        template, plan, generation = generated_scenario(
            (Phase.P2, Phase.P3, Phase.P5),
            (PhaseVariant.P2_2, PhaseVariant.P3_1, PhaseVariant.P5_1),
        )
        turn_fake = FakeSemanticTurnValidator(response_factory=lambda _: passing_turn())
        dialogue_fake = FakeSemanticDialogueValidator(passing_dialogue())
        profile = LanguageProfileLoader().load(template.language_profile)
        result = semantic_service(turn_fake, dialogue_fake).validate(
            generation.dialogue,
            plan,
            template,
            generation.final_booking_state,
            generation.deterministic_validation,
            profile,
        )
        self.assertEqual(
            [request.generated_exchange.exchange_number for request in turn_fake.received_requests],
            [1, 2, 3],
        )
        self.assertEqual(len(dialogue_fake.received_requests), 1)
        self.assertEqual(turn_fake.received_requests[0].nearby_exchanges, ())
        self.assertEqual(
            turn_fake.received_requests[1].nearby_exchanges[0].exchange_number, 1
        )
        self.assertIs(turn_fake.received_requests[0].language_profile_config, profile)
        self.assertIs(dialogue_fake.received_requests[0].language_profile_config, profile)
        self.assertTrue(result.overall_semantic_pass)

    def test_state_is_replayed_for_d2_d3_p4_2_and_d1(self) -> None:
        cases = (
            (
                Phase.P3,
                PhaseVariant.P3_1,
                DeviationType.D2,
                state_change("day", StateOperation.REPLACE, "tisdag", StateChangeSource.DEVIATION),
                "day",
                "måndag",
                "tisdag",
                (DeviationType.D2,),
            ),
            (
                Phase.P3,
                PhaseVariant.P3_1,
                DeviationType.D3,
                state_change(
                    "selected_services", StateOperation.ADD, "färgning", StateChangeSource.DEVIATION
                ),
                "selected_services",
                ("klippning",),
                ("klippning", "färgning"),
                (DeviationType.D3,),
            ),
            (
                Phase.P4,
                PhaseVariant.P4_2,
                None,
                state_change("day", StateOperation.REPLACE, "tisdag", StateChangeSource.PHASE_VARIANT),
                "day",
                "måndag",
                "tisdag",
                (),
            ),
        )
        for phase, variant, deviation, change, field, before, after, used in cases:
            with self.subTest(variant=variant):
                template, plan, generation = generated_scenario(
                    (phase,),
                    (variant,),
                    deviations=((1, deviation),) if deviation else (),
                    changes={1: (change,)},
                )
                turn_fake = FakeSemanticTurnValidator(response_factory=lambda _: passing_turn())
                service = semantic_service(
                    turn_fake, FakeSemanticDialogueValidator(passing_dialogue())
                )
                service.validate(
                    generation.dialogue,
                    plan,
                    template,
                    generation.final_booking_state,
                    generation.deterministic_validation,
                    LanguageProfileLoader().load(template.language_profile),
                )
                request = turn_fake.received_requests[0]
                self.assertEqual(request.state_before.current_values[field], before)
                self.assertEqual(request.state_after.current_values[field], after)
                self.assertEqual(request.state_after.used_deviations, used)

        template, plan, generation = generated_scenario(
            (Phase.P3,),
            (PhaseVariant.P3_1,),
            deviations=((1, DeviationType.D1),),
        )
        turn_fake = FakeSemanticTurnValidator(response_factory=lambda _: passing_turn())
        semantic_service(turn_fake, FakeSemanticDialogueValidator(passing_dialogue())).validate(
            generation.dialogue,
            plan,
            template,
            generation.final_booking_state,
            generation.deterministic_validation,
            LanguageProfileLoader().load(template.language_profile),
        )
        request = turn_fake.received_requests[0]
        self.assertEqual(request.state_before.current_values, request.state_after.current_values)

    def test_turn_and_dialogue_failures_control_semantic_pass(self) -> None:
        template, plan, generation = generated_scenario(
            (Phase.P2,), (PhaseVariant.P2_2,)
        )
        profile = LanguageProfileLoader().load(template.language_profile)
        turn_failures = (
            failing_turn(contradiction_found=True),
            failing_turn(mlp1_semantic_match=False),
        )
        for response in turn_failures:
            with self.subTest(response=response):
                result = semantic_service(
                    FakeSemanticTurnValidator(outcomes=(response,)),
                    FakeSemanticDialogueValidator(passing_dialogue()),
                ).validate(
                    generation.dialogue,
                    plan,
                    template,
                    generation.final_booking_state,
                    generation.deterministic_validation,
                    profile,
                )
                self.assertFalse(result.overall_semantic_pass)

        incoherent = SemanticDialogueValidationResponse(
            True, True, False, False, True, True, False, 0.7
        )
        result = semantic_service(
            FakeSemanticTurnValidator(outcomes=(passing_turn(),)),
            FakeSemanticDialogueValidator(incoherent),
        ).validate(
            generation.dialogue,
            plan,
            template,
            generation.final_booking_state,
            generation.deterministic_validation,
            profile,
        )
        self.assertFalse(result.overall_semantic_pass)

    def test_result_preserves_issue_severities_and_minimum_confidence(self) -> None:
        def issue(severity: ValidationSeverity) -> ValidationIssue:
            return ValidationIssue(
                severity.value,
                "semantic issue",
                severity,
                ValidationSource.SEMANTIC_AI,
                ValidationScope.DIALOGUE,
            )

        result = SemanticValidationResult(
            (
                passing_turn(0.8, (issue(ValidationSeverity.QUALITY),)),
                SemanticTurnValidationResponse(
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                    False,
                    False,
                    0.4,
                    (issue(ValidationSeverity.HARD),),
                ),
            ),
            passing_dialogue(0.6, (issue(ValidationSeverity.UNCERTAIN),)),
        )
        self.assertTrue(result.has_quality_issues)
        self.assertTrue(result.has_uncertain_issues)
        self.assertTrue(result.has_hard_failures)
        self.assertEqual(result.minimum_confidence, 0.4)
        self.assertFalse(result.overall_semantic_pass)
        json.dumps(result.to_dict(), ensure_ascii=False)

    def test_worker_exceptions_and_invalid_types_are_clear(self) -> None:
        template, plan, generation = generated_scenario(
            (Phase.P2,), (PhaseVariant.P2_2,)
        )
        args = (
            generation.dialogue,
            plan,
            template,
            generation.final_booking_state,
            generation.deterministic_validation,
            LanguageProfileLoader().load(template.language_profile),
        )
        with self.assertRaises(SemanticTurnValidatorError):
            semantic_service(
                FakeSemanticTurnValidator(outcomes=(RuntimeError("turn failed"),)),
                FakeSemanticDialogueValidator(passing_dialogue()),
            ).validate(*args)
        with self.assertRaises(SemanticDialogueValidatorError):
            semantic_service(
                FakeSemanticTurnValidator(outcomes=(passing_turn(),)),
                FakeSemanticDialogueValidator(RuntimeError("dialogue failed")),
            ).validate(*args)

        invalid_turn = FakeSemanticTurnValidator(response_factory=lambda _: passing_turn())
        invalid_turn._factory = lambda _: object()  # type: ignore[assignment]
        with self.assertRaises(InvalidSemanticResponseError):
            semantic_service(
                invalid_turn, FakeSemanticDialogueValidator(passing_dialogue())
            ).validate(*args)

        invalid_dialogue = FakeSemanticDialogueValidator(passing_dialogue())
        invalid_dialogue._outcome = object()  # type: ignore[assignment]
        with self.assertRaises(InvalidSemanticResponseError):
            semantic_service(
                FakeSemanticTurnValidator(outcomes=(passing_turn(),)), invalid_dialogue
            ).validate(*args)

    def test_service_does_not_mutate_inputs(self) -> None:
        template, plan, generation = generated_scenario(
            (Phase.P2,), (PhaseVariant.P2_2,)
        )
        snapshots = (
            template.to_dict(),
            plan.to_dict(),
            generation.dialogue.to_dict(),
            generation.final_booking_state.to_dict(),
        )
        semantic_service(
            FakeSemanticTurnValidator(outcomes=(passing_turn(),)),
            FakeSemanticDialogueValidator(passing_dialogue()),
        ).validate(
            generation.dialogue,
            plan,
            template,
            generation.final_booking_state,
            generation.deterministic_validation,
            LanguageProfileLoader().load(template.language_profile),
        )
        self.assertEqual(template.to_dict(), snapshots[0])
        self.assertEqual(plan.to_dict(), snapshots[1])
        self.assertEqual(generation.dialogue.to_dict(), snapshots[2])
        self.assertEqual(generation.final_booking_state.to_dict(), snapshots[3])


if __name__ == "__main__":
    unittest.main()
