import json
import unittest
from dataclasses import FrozenInstanceError
from typing import Any

from src.agent.application_service import ApplicationService
from src.agent.turn_plan_builder import TurnPlanBuilder
from src.ai.schemas import (
    GenerationResponse,
    SemanticDialogueValidationResponse,
    SemanticTurnValidationResponse,
)
from src.models.classification import ClassificationConfig
from src.models.enums import (
    Classification,
    DeviationType,
    Phase,
    PhaseVariant,
    StateChangeSource,
    StateOperation,
    ValidationScope,
    ValidationSeverity,
    ValidationSource,
)
from src.models.input_bundle import AgentInputBundle, MLPTaxonomy, PoolData
from src.models.validation import ValidationIssue
from tests.fakes.fake_conversation_generator import FakeConversationGenerator
from tests.fakes.fake_semantic_validators import (
    FakeSemanticDialogueValidator,
    FakeSemanticTurnValidator,
)
from tests.unit.test_generation_service import (
    make_template_and_plan,
    response_for,
    state_change,
)


CONFIG = ClassificationConfig("test", 0.85, 0.60)


def passing_turn(
    confidence: float = 0.9, issues: tuple[ValidationIssue, ...] = ()
) -> SemanticTurnValidationResponse:
    return SemanticTurnValidationResponse(
        True, True, True, True, True, False, False, True, confidence, issues
    )


def passing_dialogue(confidence: float = 0.9) -> SemanticDialogueValidationResponse:
    return SemanticDialogueValidationResponse(
        True, True, True, False, True, True, True, confidence
    )


def make_bundle(turns: int = 3) -> AgentInputBundle:
    phases = (Phase.P2, Phase.P3, Phase.P5)[:turns]
    variants = (PhaseVariant.P2_2, PhaseVariant.P3_1, PhaseVariant.P5_1)[:turns]
    template, _ = make_template_and_plan(phases, variants)
    return AgentInputBundle(
        template=template,
        pool=PoolData(
            services=("klippning", "färgning", "skägg"),
            days=("måndag", "tisdag"),
            time_expressions=("10:00", "14:00"),
            stylists=("Alex",),
            booking_actions=("boka", "omboka"),
            authentication_values={"phone_formats": ["SE"]},
            version="pool-v1",
        ),
        mlp_taxonomy=MLPTaxonomy(
            {
                "Booking": ("boka", "omboka", "avboka"),
                "FAQ": (),
                "Tjänster": ("lägga till", "ta bort", "byta ut"),
                "SmallTalk": (),
                "Auth": ("skapa", "ändra"),
            }, version="mlp-v1"
        ),
        input_versions={"templates": "templates-v1"},
    )


def make_application(
    generator: FakeConversationGenerator,
    turn_validator: FakeSemanticTurnValidator | None = None,
    dialogue_validator: FakeSemanticDialogueValidator | None = None,
) -> ApplicationService:
    return ApplicationService(
        generator,
        turn_validator
        or FakeSemanticTurnValidator(response_factory=lambda _: passing_turn()),
        dialogue_validator or FakeSemanticDialogueValidator(passing_dialogue()),
        classification_config=CONFIG,
    )


class InputModelTests(unittest.TestCase):
    def test_bundle_pool_and_taxonomy_are_immutable_and_serializable(self) -> None:
        bundle = make_bundle()
        with self.assertRaises(FrozenInstanceError):
            bundle.pool = PoolData()  # type: ignore[misc]
        with self.assertRaises(TypeError):
            bundle.input_versions["templates"] = "changed"  # type: ignore[index]
        with self.assertRaises(TypeError):
            bundle.mlp_taxonomy.relations["Booking"] = frozenset()  # type: ignore[index]
        self.assertIsInstance(bundle.mlp_taxonomy.relations["Booking"], frozenset)
        json.dumps(bundle.pool.to_dict(), ensure_ascii=False)
        json.dumps(bundle.mlp_taxonomy.to_dict(), ensure_ascii=False)
        json.dumps(bundle.to_dict(), ensure_ascii=False)

    def test_pool_values_do_not_overwrite_completed_template_values(self) -> None:
        bundle = make_bundle()
        template_before = bundle.template.to_dict()
        pool_before = bundle.pool.to_dict()
        result = make_application(
            FakeConversationGenerator(response_factory=response_for)
        ).run(bundle)
        self.assertIs(result.classification, Classification.SAFE)
        self.assertEqual(bundle.template.to_dict(), template_before)
        self.assertEqual(bundle.pool.to_dict(), pool_before)
        self.assertEqual(
            result.final_booking_state.current_values["selected_services"],
            ["klippning"],
        )

    def test_taxonomy_is_the_deterministic_pipeline_source(self) -> None:
        bundle = make_bundle(1)
        incompatible = AgentInputBundle(
            bundle.template,
            bundle.pool,
            MLPTaxonomy({"Booking": ("different_action",)}, "wrong-taxonomy"),
        )
        turn_fake = FakeSemanticTurnValidator(response_factory=lambda _: passing_turn())
        dialogue_fake = FakeSemanticDialogueValidator(passing_dialogue())
        result = make_application(
            FakeConversationGenerator(response_factory=response_for),
            turn_fake,
            dialogue_fake,
        ).run(incompatible)
        self.assertIs(result.classification, Classification.ERROR)
        self.assertEqual(turn_fake.received_requests, [])
        self.assertEqual(dialogue_fake.received_requests, [])


class ApplicationServiceTests(unittest.TestCase):
    def test_end_to_end_without_template_mlp_facit_can_be_safe(self) -> None:
        template, _ = make_template_and_plan(
            (Phase.P3,),
            (PhaseVariant.P3_1,),
            deviations=((1, DeviationType.D3),),
            changes={
                1: (
                    state_change(
                        "selected_services",
                        StateOperation.ADD,
                        "skägg",
                        StateChangeSource.DEVIATION,
                    ),
                )
            },
            include_semantic_hints=False,
        )
        bundle = AgentInputBundle(template, make_bundle(1).pool, make_bundle(1).mlp_taxonomy)

        def add_service(request: Any) -> GenerationResponse:
            self.assertEqual(request.mlp_taxonomy.to_dict(), bundle.mlp_taxonomy.to_dict())
            self.assertIsNone(TurnPlanBuilder().build(template).exchanges[0].semantic_action)
            return GenerationResponse(
                1,
                "Kan vi lägga till skägg också?",
                "Absolut, jag lägger till skägg.",
                "Tjänster",
                "lägga till",
            )

        generator = FakeConversationGenerator(response_factory=add_service)
        turns = FakeSemanticTurnValidator(response_factory=lambda _: passing_turn())
        result = make_application(generator, turns).run(bundle)

        self.assertIs(result.classification, Classification.SAFE)
        self.assertEqual(turns.received_requests[0].generated_exchange.customer_mlp1, "Tjänster")
        self.assertEqual(turns.received_requests[0].generated_exchange.customer_mlp2, "lägga till")
        self.assertEqual(
            turns.received_requests[0].mlp_taxonomy.to_dict(), bundle.mlp_taxonomy.to_dict()
        )
        self.assertEqual(template.mlp_expectations, ())

    def test_semantically_wrong_generated_label_without_hidden_facit_is_error(self) -> None:
        template, _ = make_template_and_plan(
            (Phase.P2,),
            (PhaseVariant.P2_2,),
            include_semantic_hints=False,
        )
        base = make_bundle(1)
        bundle = AgentInputBundle(template, base.pool, base.mlp_taxonomy)
        response = GenerationResponse(
            1,
            "Kan vi lägga till skägg också?",
            "Absolut.",
            "Booking",
            "boka",
        )
        semantic_mismatch = SemanticTurnValidationResponse(
            True, False, False, True, True, False, False, False, 0.95
        )
        result = make_application(
            FakeConversationGenerator(outcomes=(response,)),
            FakeSemanticTurnValidator(outcomes=(semantic_mismatch,)),
        ).run(bundle)
        self.assertIs(result.classification, Classification.ERROR)

    def test_clean_three_turn_run_is_safe_with_exact_worker_counts(self) -> None:
        bundle = make_bundle(3)
        generator = FakeConversationGenerator(response_factory=response_for)
        turns = FakeSemanticTurnValidator(response_factory=lambda _: passing_turn())
        dialogue = FakeSemanticDialogueValidator(passing_dialogue())
        template_before = bundle.template.to_dict()
        taxonomy_before = bundle.mlp_taxonomy.to_dict()

        result = make_application(generator, turns, dialogue).run(bundle)

        self.assertIs(result.classification, Classification.SAFE)
        self.assertEqual(result.status, Classification.SAFE)
        self.assertEqual(result.output_label, "safe")
        self.assertEqual(len(generator.received_requests), 3)
        self.assertEqual(len(turns.received_requests), 3)
        self.assertEqual(len(dialogue.received_requests), 1)
        self.assertEqual(bundle.template.to_dict(), template_before)
        self.assertEqual(bundle.mlp_taxonomy.to_dict(), taxonomy_before)
        json.dumps(result.to_dict(), ensure_ascii=False)

    def test_quality_semantic_result_is_weak_and_maps_to_tvek(self) -> None:
        bundle = make_bundle(1)
        quality = ValidationIssue(
            "QUALITY",
            "minor naturalness issue",
            ValidationSeverity.QUALITY,
            ValidationSource.SEMANTIC_AI,
            ValidationScope.EXCHANGE,
        )
        turns = FakeSemanticTurnValidator(outcomes=(passing_turn(issues=(quality,)),))
        result = make_application(
            FakeConversationGenerator(response_factory=response_for), turns
        ).run(bundle)
        self.assertIs(result.classification, Classification.WEAK)
        self.assertEqual(result.output_label, "tvek")

    def test_blocking_semantic_result_is_error_and_maps_to_fel(self) -> None:
        bundle = make_bundle(1)
        blocking = SemanticTurnValidationResponse(
            True, False, True, True, True, False, False, False, 0.9
        )
        turns = FakeSemanticTurnValidator(outcomes=(blocking,))
        result = make_application(
            FakeConversationGenerator(response_factory=response_for), turns
        ).run(bundle)
        self.assertIs(result.classification, Classification.ERROR)
        self.assertEqual(result.output_label, "fel")

    def test_deterministic_hard_failure_stops_semantic_workers(self) -> None:
        bundle = make_bundle(1)
        object.__setattr__(bundle.template, "available_services", ("färgning",))
        turns = FakeSemanticTurnValidator(response_factory=lambda _: passing_turn())
        dialogue = FakeSemanticDialogueValidator(passing_dialogue())
        result = make_application(
            FakeConversationGenerator(response_factory=response_for), turns, dialogue
        ).run(bundle)
        self.assertIs(result.classification, Classification.ERROR)
        self.assertEqual(result.errors[0].code, "DETERMINISTIC_VALIDATION_FAILURE")
        self.assertEqual(turns.received_requests, [])
        self.assertEqual(dialogue.received_requests, [])

    def test_generation_worker_exception_is_structured_error(self) -> None:
        bundle = make_bundle(1)
        result = make_application(
            FakeConversationGenerator(outcomes=(RuntimeError("generator down"),))
        ).run(bundle)
        self.assertIs(result.classification, Classification.ERROR)
        self.assertEqual(result.errors[0].stage, "generation")
        self.assertEqual(result.output_label, "fel")

    def test_invalid_generation_response_is_structured_error(self) -> None:
        bundle = make_bundle(1)
        invalid = GenerationResponse(2, "Kund", "AI", "Booking", "boka")
        result = make_application(FakeConversationGenerator(outcomes=(invalid,))).run(bundle)
        self.assertIs(result.classification, Classification.ERROR)
        self.assertEqual(result.errors[0].code, "InvalidGenerationResponseError")

    def test_semantic_worker_exception_is_structured_error(self) -> None:
        bundle = make_bundle(1)
        turns = FakeSemanticTurnValidator(outcomes=(RuntimeError("validator down"),))
        result = make_application(
            FakeConversationGenerator(response_factory=response_for), turns
        ).run(bundle)
        self.assertIs(result.classification, Classification.ERROR)
        self.assertEqual(result.errors[0].stage, "semantic_validation")
        self.assertIsNotNone(result.dialogue)
        self.assertIsNotNone(result.deterministic_validation)


if __name__ == "__main__":
    unittest.main()
