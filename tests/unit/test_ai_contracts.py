import json
import unittest
from dataclasses import FrozenInstanceError, fields, replace

from src.ai.language_profiles import LanguageProfileLoader
from src.ai.prompt_builder import PromptContextBuilder
from src.ai.schemas import (
    BookingStateSnapshot,
    GenerationRequest,
    GenerationResponse,
    SemanticDialogueValidationResponse,
    SemanticDialogueValidationRequest,
    SemanticTurnValidationRequest,
    SemanticTurnValidationResponse,
)
from src.agent.state_machine import StateMachine
from src.models.dialogue import ConversationExchange, Dialogue, Utterance
from src.models.enums import (
    DeviationType,
    LanguageProfile,
    Phase,
    PhaseVariant,
    Speaker,
    StateChangeSource,
    StateOperation,
    ValidationScope,
    ValidationSeverity,
    ValidationSource,
)
from src.models.input_bundle import MLPTaxonomy, PoolData
from src.models.plan import PlannedExchange, PlannedSemanticAction, PlannedStateChange, TurnPlan
from src.models.state import BookingState
from src.models.validation import ValidationIssue


def make_planned_exchange() -> PlannedExchange:
    return PlannedExchange(
        exchange_number=2,
        phase=Phase.P4,
        phase_variant=PhaseVariant.P4_2,
        semantic_action=PlannedSemanticAction("Booking", "boka"),
        required_customer_facts={"day": "tisdag"},
        required_assistant_facts={"name": "Ada"},
        allowed_customer_facts={"day": "tisdag", "time": "14:00"},
        allowed_assistant_facts={"name": "Ada"},
        deviation=None,
        state_changes=(
            PlannedStateChange(
                "day",
                StateOperation.REPLACE,
                "tisdag",
                StateChangeSource.PHASE_VARIANT,
            ),
        ),
    )


def make_request(profile: LanguageProfile = LanguageProfile.VARDAGLIG_SVENSKA) -> GenerationRequest:
    planned = make_planned_exchange()
    state = BookingState(
        original_values={"day": "måndag", "name": "Ada"},
        current_values={"day": "måndag", "name": "Ada"},
    )
    state_after = BookingState(
        original_values=state.original_values,
        current_values=state.current_values,
    )
    StateMachine().apply(planned, state_after)
    previous = ConversationExchange(
        1,
        Utterance(Speaker.CUSTOMER, "Hej"),
        Utterance(Speaker.ASSISTANT, "Hej!"),
        phase=Phase.P3,
        phase_variant=PhaseVariant.P3_1,
    )
    return GenerationRequest(
        template_id="template-1",
        plan_version="1",
        exchange_number=planned.exchange_number,
        phase=planned.phase,
        phase_variant=planned.phase_variant,
        language_profile=profile,
        mlp_taxonomy=MLPTaxonomy(
            {
                "Booking": ("boka", "omboka", "avboka"),
                "FAQ": (),
                "Tjänster": ("lägga till", "ta bort", "byta ut"),
                "SmallTalk": (),
                "Auth": ("skapa", "ändra"),
            }
        ),
        pool_data=PoolData(services=("klippning", "fade")),
        required_customer_facts=planned.required_customer_facts,
        required_assistant_facts=planned.required_assistant_facts,
        allowed_customer_facts=planned.allowed_customer_facts,
        allowed_assistant_facts=planned.allowed_assistant_facts,
        deviation=planned.deviation,
        planned_state_changes=planned.state_changes,
        booking_state_before=BookingStateSnapshot.from_booking_state(state),
        booking_state_after_expected=BookingStateSnapshot.from_booking_state(state_after),
        previous_approved_exchanges=(previous,),
        remaining_turns_after_current=3,
        applicable_rules=("P4.2 requires one correction",),
    )


def make_response(exchange_number: int = 2) -> GenerationResponse:
    return GenerationResponse(
        exchange_number=exchange_number,
        customer_text="Jag menade tisdag.",
        assistant_text="Tack, jag ändrar dagen.",
        customer_mlp1="Booking",
        customer_mlp2="boka",
        customer_facts={"day": "tisdag"},
        assistant_facts={"name": "Ada"},
        summary_facts={"day": "tisdag", "name": "Ada"},
        model_metadata={"model": "fake-test-worker"},
    )


class GenerationContractTests(unittest.TestCase):
    def test_generation_request_is_immutable_and_deeply_frozen(self) -> None:
        request = make_request()
        with self.assertRaises(FrozenInstanceError):
            request.exchange_number = 3  # type: ignore[misc]
        with self.assertRaises(TypeError):
            request.allowed_customer_facts["day"] = "fredag"  # type: ignore[index]

    def test_request_contains_locked_plan_information(self) -> None:
        request = make_request()
        self.assertIs(request.phase, Phase.P4)
        self.assertIs(request.phase_variant, PhaseVariant.P4_2)
        request_fields = {item.name for item in fields(GenerationRequest)}
        self.assertNotIn("expected_mlp1", request_fields)
        self.assertNotIn("expected_mlp2", request_fields)
        self.assertEqual(request.mlp_taxonomy.relations["Booking"], frozenset({"boka", "omboka", "avboka"}))
        self.assertEqual(request.pool_data.services, ("klippning", "fade"))
        self.assertEqual(request.required_customer_facts["day"], "tisdag")
        self.assertEqual(request.allowed_customer_facts["time"], "14:00")

    def test_request_contains_immutable_before_and_expected_after_state(self) -> None:
        request = make_request()
        self.assertEqual(request.booking_state_before.current_values["day"], "måndag")
        self.assertEqual(request.booking_state_after_expected.current_values["day"], "tisdag")
        with self.assertRaises(TypeError):
            request.booking_state_before.current_values["day"] = "fredag"  # type: ignore[index]
        with self.assertRaises(TypeError):
            request.booking_state_after_expected.current_values["day"] = "fredag"  # type: ignore[index]

    def test_booking_snapshot_contains_immutable_required_field_progress(self) -> None:
        state = BookingState(
            original_values={"phone": "0701234567", "name": "Ada"},
            current_values={"phone": "0701234567", "name": "Ada"},
            expressed_required_fields={"phone"},
            missing_required_fields={"name"},
        )
        snapshot = BookingStateSnapshot.from_booking_state(state)
        self.assertEqual(snapshot.expressed_required_fields, frozenset({"phone"}))
        self.assertEqual(snapshot.missing_required_fields, frozenset({"name"}))
        with self.assertRaises(AttributeError):
            snapshot.expressed_required_fields.add("name")  # type: ignore[attr-defined]
        payload = snapshot.to_dict()
        self.assertEqual(payload["expressed_required_fields"], ["phone"])
        self.assertEqual(payload["missing_required_fields"], ["name"])

    def test_remaining_turns_means_only_turns_after_current_exchange(self) -> None:
        request = make_request()
        self.assertEqual(request.exchange_number, 2)
        self.assertEqual(request.remaining_turns_after_current, 3)
        self.assertNotIn("remaining_turns", request.to_dict())

    def test_response_converts_to_exactly_one_planned_exchange(self) -> None:
        planned = make_planned_exchange()
        exchange = make_response().to_exchange(planned)
        self.assertIsInstance(exchange, ConversationExchange)
        self.assertEqual(exchange.exchange_number, 2)
        self.assertIs(exchange.phase, Phase.P4)
        self.assertIs(exchange.phase_variant, PhaseVariant.P4_2)

    def test_response_supports_each_mlp1_shape_in_the_taxonomy(self) -> None:
        labels = (
            ("Booking", "boka"),
            ("Tjänster", "lägga till"),
            ("Auth", "skapa"),
            ("FAQ", None),
            ("SmallTalk", None),
        )
        for mlp1, mlp2 in labels:
            with self.subTest(mlp1=mlp1, mlp2=mlp2):
                response = replace(make_response(), customer_mlp1=mlp1, customer_mlp2=mlp2)
                self.assertEqual(response.customer_mlp1, mlp1)
                self.assertEqual(response.customer_mlp2, mlp2)
                json.dumps(response.to_dict(), ensure_ascii=False)

    def test_response_cannot_supply_plan_structure_or_state_changes(self) -> None:
        names = {item.name for item in fields(GenerationResponse)}
        self.assertNotIn("phase", names)
        self.assertNotIn("phase_variant", names)
        self.assertNotIn("deviation", names)
        self.assertNotIn("state_changes", names)

    def test_wrong_response_exchange_number_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "exchange_number"):
            make_response(exchange_number=3).to_exchange(make_planned_exchange())

    def test_summary_facts_bind_for_p4_but_are_rejected_for_p2_and_p3(self) -> None:
        response = make_response()
        self.assertTrue(response.to_exchange(make_planned_exchange()).summary_facts)
        non_summary_plans = (
            replace(
                make_planned_exchange(),
                phase=Phase.P2,
                phase_variant=PhaseVariant.P2_2,
                state_changes=(),
            ),
            replace(
                make_planned_exchange(),
                phase=Phase.P3,
                phase_variant=PhaseVariant.P3_1,
                state_changes=(),
            ),
        )
        for planned in non_summary_plans:
            with self.subTest(phase=planned.phase):
                with self.assertRaisesRegex(ValueError, "only allowed for P4"):
                    response.to_exchange(planned)

    def test_generation_models_are_json_serializable(self) -> None:
        json.dumps(make_request().to_dict(), ensure_ascii=False)
        json.dumps(make_response().to_dict(), ensure_ascii=False)


class LanguageProfileAndPromptTests(unittest.TestCase):
    def test_all_four_language_profiles_load(self) -> None:
        loader = LanguageProfileLoader()
        loaded = {profile: loader.load(profile) for profile in LanguageProfile}
        self.assertEqual(set(loaded), set(LanguageProfile))
        self.assertTrue(all(config.profile is profile for profile, config in loaded.items()))

    def test_language_profile_contains_only_style_context(self) -> None:
        style = LanguageProfileLoader().load(LanguageProfile.ORTEN_SVENSKA).to_dict()
        forbidden = {
            "intent",
            "mlp1",
            "mlp2",
            "services",
            "day",
            "time",
            "stylist",
            "name",
            "phone",
            "phase",
            "phase_variant",
            "deviation",
            "state_changes",
            "total_turns",
        }
        self.assertTrue(forbidden.isdisjoint(style))

    def test_prompt_builder_preserves_facts_and_is_deterministic(self) -> None:
        request = make_request()
        profile = LanguageProfileLoader().load(request.language_profile)
        builder = PromptContextBuilder()
        first = builder.build(request, profile).to_dict()
        second = builder.build(request, profile).to_dict()
        self.assertEqual(first, second)
        generation = first["generation"]
        self.assertEqual(
            generation["required_customer_facts"], request.to_dict()["required_customer_facts"]
        )
        self.assertEqual(
            generation["allowed_customer_facts"], request.to_dict()["allowed_customer_facts"]
        )
        self.assertTrue(first["constraints"]["facts_not_in_allowed_are_forbidden"])
        self.assertEqual(first["generation"]["mlp_taxonomy"], request.mlp_taxonomy.to_dict())


class SemanticContractTests(unittest.TestCase):
    def test_semantic_requests_include_actual_style_context(self) -> None:
        request = make_request()
        profile = LanguageProfileLoader().load(request.language_profile)
        planned = make_planned_exchange()
        exchange = make_response().to_exchange(planned)
        snapshot = request.booking_state_before
        turn_request = SemanticTurnValidationRequest(
            planned,
            exchange,
            request.language_profile,
            profile,
            request.mlp_taxonomy,
            snapshot,
            request.booking_state_after_expected,
        )
        dialogue_exchange = replace(exchange, exchange_number=1)
        dialogue_plan_exchange = replace(planned, exchange_number=1)
        dialogue_request = SemanticDialogueValidationRequest(
            Dialogue(
                request.template_id,
                request.language_profile,
                (dialogue_exchange,),
                plan_version=request.plan_version,
            ),
            TurnPlan(request.template_id, 1, (dialogue_plan_exchange,)),
            request.booking_state_after_expected,
            request.language_profile,
            profile,
            request.mlp_taxonomy,
        )
        self.assertEqual(turn_request.language_profile_config.formality, profile.formality)
        self.assertEqual(dialogue_request.language_profile_config.address_style, profile.address_style)
        json.dumps(turn_request.to_dict(), ensure_ascii=False)
        json.dumps(dialogue_request.to_dict(), ensure_ascii=False)

    def test_semantic_issue_must_use_semantic_ai_source(self) -> None:
        semantic_issue = ValidationIssue(
            code="UNNATURAL",
            message="The exchange sounds unnatural",
            severity=ValidationSeverity.QUALITY,
            source=ValidationSource.SEMANTIC_AI,
            scope=ValidationScope.EXCHANGE,
            exchange_number=2,
        )
        response = SemanticTurnValidationResponse(
            language_profile_match=True,
            mlp1_semantic_match=True,
            mlp2_semantic_match=True,
            facts_semantically_preserved=True,
            naturalness=False,
            contradiction_found=False,
            unnecessary_question_found=False,
            overall_pass=False,
            confidence=0.8,
            issues=(semantic_issue,),
        )
        self.assertIs(response.issues[0].source, ValidationSource.SEMANTIC_AI)

        deterministic_issue = ValidationIssue(
            code="WRONG_SOURCE",
            message="Wrong source",
            severity=ValidationSeverity.HARD,
            source=ValidationSource.DETERMINISTIC,
            scope=ValidationScope.EXCHANGE,
        )
        with self.assertRaisesRegex(ValueError, "SEMANTIC_AI"):
            SemanticTurnValidationResponse(
                True, True, True, True, True, False, False, False, 0.5, (deterministic_issue,)
            )

    def test_semantic_responses_are_json_serializable(self) -> None:
        turn = SemanticTurnValidationResponse(
            True, True, True, True, True, False, False, True, 0.95
        )
        dialogue = SemanticDialogueValidationResponse(
            True, True, True, False, True, True, True, 0.9
        )
        json.dumps(turn.to_dict(), ensure_ascii=False)
        json.dumps(dialogue.to_dict(), ensure_ascii=False)

    def test_logically_contradictory_turn_responses_are_rejected(self) -> None:
        invalid_overrides = (
            {"contradiction_found": True},
            {"mlp1_semantic_match": False},
            {"facts_semantically_preserved": False},
        )
        base = {
            "language_profile_match": True,
            "mlp1_semantic_match": True,
            "mlp2_semantic_match": True,
            "facts_semantically_preserved": True,
            "naturalness": True,
            "contradiction_found": False,
            "unnecessary_question_found": False,
            "overall_pass": True,
            "confidence": 0.9,
        }
        for override in invalid_overrides:
            with self.subTest(override=override):
                with self.assertRaisesRegex(ValueError, "contradicts"):
                    SemanticTurnValidationResponse(**{**base, **override})

    def test_logically_contradictory_dialogue_responses_are_rejected(self) -> None:
        invalid_overrides = (
            {"coherence": False},
            {"pd_intention_followed": False},
        )
        base = {
            "global_language_profile_match": True,
            "naturalness": True,
            "coherence": True,
            "contradiction_found": False,
            "state_facts_consistent": True,
            "pd_intention_followed": True,
            "overall_pass": True,
            "confidence": 0.9,
        }
        for override in invalid_overrides:
            with self.subTest(override=override):
                with self.assertRaisesRegex(ValueError, "contradicts"):
                    SemanticDialogueValidationResponse(**{**base, **override})

    def test_naturalness_false_alone_does_not_forbid_overall_pass(self) -> None:
        turn = SemanticTurnValidationResponse(
            True, True, True, True, False, False, False, True, 0.8
        )
        dialogue = SemanticDialogueValidationResponse(
            True, False, True, False, True, True, True, 0.8
        )
        self.assertTrue(turn.overall_pass)
        self.assertTrue(dialogue.overall_pass)


if __name__ == "__main__":
    unittest.main()
