import unittest
from typing import Any, Sequence

from src.agent.generation_request_builder import GenerationRequestBuilder
from src.agent.generation_service import (
    ConversationGenerationService,
    GenerationStateTransitionError,
    GeneratorExecutionError,
    InvalidGenerationResponseError,
)
from src.agent.state_machine import StateMachine
from src.agent.turn_plan_builder import TurnPlanBuilder
from src.ai.language_profiles import LanguageProfileLoader
from src.ai.schemas import GenerationRequest, GenerationResponse
from src.models.input_bundle import MLPTaxonomy, PoolData
from src.models.dialogue import ConversationExchange, Utterance
from src.models.enums import (
    DeviationType,
    LanguageProfile,
    Phase,
    PhaseVariant,
    Speaker,
    StateChangeSource,
    StateOperation,
)
from src.models.plan import PlannedStateChange, TurnPlan
from src.models.state import BookingState
from src.models.template import (
    ConversationTemplate,
    DeviationAssignment,
    MLPExpectation,
    TemplateExchangeRequirements,
    TemplateStateChange,
    thaw_value,
)
from src.validators.deterministic_pipeline import DeterministicValidationPipeline
from src.validators.entity_validator import EntityValidator
from tests.fakes.fake_conversation_generator import FakeConversationGenerator


TAXONOMY = {
    "Booking": frozenset({"boka", "omboka", "avboka"}),
    "FAQ": frozenset(),
    "Tjänster": frozenset({"lägga till", "ta bort", "byta ut"}),
    "SmallTalk": frozenset(),
    "Auth": frozenset({"skapa", "ändra"}),
}
MLP_TAXONOMY = MLPTaxonomy({key: tuple(values) for key, values in TAXONOMY.items()})
POOL_DATA = PoolData(services=("klippning", "färgning", "skägg", "tvätt"))


def state_change(
    field_name: str,
    operation: StateOperation,
    value: Any,
    source: StateChangeSource,
) -> TemplateStateChange:
    return TemplateStateChange(field_name, operation, value, source)


def make_template_and_plan(
    phases: Sequence[Phase],
    variants: Sequence[PhaseVariant],
    deviations: Sequence[tuple[int, DeviationType]] = (),
    changes: dict[int, Sequence[TemplateStateChange]] | None = None,
    original: dict[str, Any] | None = None,
    customer_facts: dict[int, dict[str, Any]] | None = None,
    assistant_facts: dict[int, dict[str, Any]] | None = None,
    required_fields: Sequence[str] = (),
    include_semantic_hints: bool = True,
) -> tuple[ConversationTemplate, TurnPlan]:
    changes = changes or {}
    customer_facts = customer_facts or {}
    assistant_facts = assistant_facts or {}
    original = original or {
        "day": "måndag",
        "time": "10:00",
        "selected_services": ["klippning"],
    }
    template = ConversationTemplate(
        template_id="generation-template",
        total_turns=len(phases),
        language_profile=LanguageProfile.VARDAGLIG_SVENSKA,
        original_values=original,
        phase_by_exchange=tuple(phases),
        phase_variant_by_exchange=tuple(variants),
        available_services=("klippning", "färgning", "skägg", "tvätt"),
        required_fields=tuple(required_fields),
        deviations=tuple(
            DeviationAssignment(number, phases[number - 1], deviation)
            for number, deviation in deviations
        ),
        mlp_expectations=(
            tuple(
                MLPExpectation(number, "Booking", "boka")
                for number in range(1, len(phases) + 1)
            )
            if include_semantic_hints
            else ()
        ),
        exchange_requirements=tuple(
            TemplateExchangeRequirements(
                number,
                required_customer_facts=customer_facts.get(number, {}),
                required_assistant_facts=assistant_facts.get(number, {}),
                state_changes=tuple(changes.get(number, ())),
            )
            for number in range(1, len(phases) + 1)
        ),
    )
    return template, TurnPlanBuilder().build(template)


def response_for(request: GenerationRequest) -> GenerationResponse:
    summary = (
        request.booking_state_after_expected.current_values if request.phase is Phase.P4 else {}
    )
    return GenerationResponse(
        exchange_number=request.exchange_number,
        customer_text=f"Kund {request.exchange_number}",
        assistant_text=f"AI {request.exchange_number}",
        customer_mlp1="Booking",
        customer_mlp2="boka",
        customer_facts=request.required_customer_facts,
        assistant_facts=request.required_assistant_facts,
        summary_facts=summary,
    )


def make_service(fake: FakeConversationGenerator) -> ConversationGenerationService:
    return ConversationGenerationService(
        fake, DeterministicValidationPipeline(TAXONOMY), POOL_DATA, MLP_TAXONOMY
    )


class GenerationRequestBuilderTests(unittest.TestCase):
    def test_builder_calculates_before_after_and_remaining_without_mutation(self) -> None:
        template, plan = make_template_and_plan(
            (Phase.P3,),
            (PhaseVariant.P3_1,),
            deviations=((1, DeviationType.D2),),
            changes={
                1: (
                    state_change(
                        "day", StateOperation.REPLACE, "tisdag", StateChangeSource.DEVIATION
                    ),
                )
            },
        )
        state = BookingState(template.original_values, thaw_value(template.original_values))
        before = state.to_dict()
        request = GenerationRequestBuilder().build(
            template,
            plan,
            plan.exchanges[0],
            state,
            (),
            LanguageProfileLoader().load(template.language_profile),
            POOL_DATA,
            MLP_TAXONOMY,
        )

        self.assertEqual(request.booking_state_before.current_values["day"], "måndag")
        self.assertEqual(request.booking_state_after_expected.current_values["day"], "tisdag")
        self.assertEqual(request.remaining_turns_after_current, 0)
        self.assertEqual(request.previous_approved_exchanges, ())
        self.assertEqual(state.to_dict(), before)


class ConversationGenerationServiceTests(unittest.TestCase):
    def test_three_turns_make_three_calls_three_exchanges_and_six_utterances(self) -> None:
        template, plan = make_template_and_plan(
            (Phase.P2, Phase.P3, Phase.P5),
            (PhaseVariant.P2_2, PhaseVariant.P3_1, PhaseVariant.P5_1),
            deviations=((2, DeviationType.D2),),
            changes={
                2: (
                    state_change(
                        "day", StateOperation.REPLACE, "tisdag", StateChangeSource.DEVIATION
                    ),
                )
            },
        )
        fake = FakeConversationGenerator(response_factory=response_for)
        template_before = template.to_dict()
        plan_before = plan.to_dict()
        result = make_service(fake).generate(
            template, plan, LanguageProfileLoader().load(template.language_profile)
        )

        self.assertEqual(len(fake.received_requests), 3)
        self.assertEqual(result.dialogue.total_turns, 3)
        self.assertEqual(result.dialogue.utterance_count, 6)
        self.assertEqual(result.final_booking_state.current_values["day"], "tisdag")
        self.assertEqual(result.final_booking_state.used_deviations, [DeviationType.D2])
        self.assertTrue(result.passed)
        self.assertEqual(len(result.deterministic_validation.reports), 5)
        self.assertEqual(template.to_dict(), template_before)
        self.assertEqual(plan.to_dict(), plan_before)

    def test_previous_exchanges_only_contain_already_generated_turns(self) -> None:
        template, plan = make_template_and_plan(
            (Phase.P2, Phase.P3, Phase.P5),
            (PhaseVariant.P2_2, PhaseVariant.P3_1, PhaseVariant.P5_1),
        )
        fake = FakeConversationGenerator(response_factory=response_for)
        make_service(fake).generate(
            template, plan, LanguageProfileLoader().load(template.language_profile)
        )
        self.assertEqual(fake.received_requests[0].previous_approved_exchanges, ())
        self.assertEqual(
            tuple(item.exchange_number for item in fake.received_requests[1].previous_approved_exchanges),
            (1,),
        )
        self.assertEqual(
            tuple(item.exchange_number for item in fake.received_requests[2].previous_approved_exchanges),
            (1, 2),
        )
        self.assertEqual(
            tuple(item.remaining_turns_after_current for item in fake.received_requests),
            (2, 1, 0),
        )

    def test_python_applies_d3_and_p4_2_state(self) -> None:
        cases = (
            (
                Phase.P3,
                PhaseVariant.P3_1,
                DeviationType.D3,
                state_change(
                    "selected_services",
                    StateOperation.ADD,
                    "färgning",
                    StateChangeSource.DEVIATION,
                ),
            ),
            (
                Phase.P4,
                PhaseVariant.P4_2,
                None,
                state_change(
                    "day", StateOperation.REPLACE, "tisdag", StateChangeSource.PHASE_VARIANT
                ),
            ),
        )
        for phase, variant, deviation, change in cases:
            with self.subTest(variant=variant):
                deviations = ((1, deviation),) if deviation else ()
                template, plan = make_template_and_plan(
                    (phase,), (variant,), deviations=deviations, changes={1: (change,)}
                )
                fake = FakeConversationGenerator(response_factory=response_for)
                result = make_service(fake).generate(
                    template, plan, LanguageProfileLoader().load(template.language_profile)
                )
                if deviation is DeviationType.D3:
                    self.assertEqual(
                        result.final_booking_state.selected_services,
                        ("klippning", "färgning"),
                    )
                    self.assertEqual(result.final_booking_state.used_deviations, [DeviationType.D3])
                else:
                    self.assertEqual(result.final_booking_state.current_values["day"], "tisdag")
                    self.assertEqual(result.final_booking_state.used_deviations, [])

    def test_wrong_exchange_number_and_non_p4_summary_are_rejected(self) -> None:
        template, plan = make_template_and_plan((Phase.P2,), (PhaseVariant.P2_2,))
        wrong_number = FakeConversationGenerator(outcomes=(response_for_request_number(2),))
        with self.assertRaises(InvalidGenerationResponseError):
            make_service(wrong_number).generate(
                template, plan, LanguageProfileLoader().load(template.language_profile)
            )

        summary_response = GenerationResponse(
            1, "Kund", "AI", "booking", "update", summary_facts={"day": "måndag"}
        )
        with self.assertRaises(InvalidGenerationResponseError):
            make_service(FakeConversationGenerator(outcomes=(summary_response,))).generate(
                template, plan, LanguageProfileLoader().load(template.language_profile)
            )

    def test_generator_exception_is_wrapped_clearly(self) -> None:
        template, plan = make_template_and_plan((Phase.P2,), (PhaseVariant.P2_2,))
        fake = FakeConversationGenerator(outcomes=(RuntimeError("worker unavailable"),))
        with self.assertRaises(GeneratorExecutionError) as caught:
            make_service(fake).generate(
                template, plan, LanguageProfileLoader().load(template.language_profile)
            )
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)

    def test_state_transition_error_is_wrapped_clearly(self) -> None:
        template, plan = make_template_and_plan((Phase.P3,), (PhaseVariant.P3_1,))
        invalid_change = PlannedStateChange(
            "selected_services", StateOperation.ADD, "klippning", StateChangeSource.DEVIATION
        )
        object.__setattr__(plan.exchanges[0], "deviation", DeviationType.D3)
        object.__setattr__(plan.exchanges[0], "state_changes", (invalid_change,))
        fake = FakeConversationGenerator(response_factory=response_for)
        with self.assertRaises(GenerationStateTransitionError) as caught:
            make_service(fake).generate(
                template, plan, LanguageProfileLoader().load(template.language_profile)
            )
        self.assertEqual(len(fake.received_requests), 0)
        self.assertIsNotNone(caught.exception.__cause__)

    def test_hard_local_deterministic_failure_stops_without_repair(self) -> None:
        template, plan = make_template_and_plan((Phase.P2,), (PhaseVariant.P2_2,))

        def unplanned_fact(request: GenerationRequest) -> GenerationResponse:
            response = response_for(request)
            object.__setattr__(response, "customer_facts", {"day": "fredag"})
            return response

        fake = FakeConversationGenerator(response_factory=unplanned_fact)
        with self.assertRaises(InvalidGenerationResponseError) as caught:
            make_service(fake).generate(
                template, plan, LanguageProfileLoader().load(template.language_profile)
            )
        self.assertIsNotNone(caught.exception.validation_result)
        self.assertTrue(caught.exception.validation_result.has_hard_failures)

    def test_generator_cannot_create_extra_exchange(self) -> None:
        template, plan = make_template_and_plan(
            (Phase.P2, Phase.P3, Phase.P5),
            (PhaseVariant.P2_2, PhaseVariant.P3_1, PhaseVariant.P5_1),
        )
        outcomes = tuple(response_for_request_number(number) for number in (1, 2, 3, 4))
        fake = FakeConversationGenerator(outcomes=outcomes)
        result = make_service(fake).generate(
            template, plan, LanguageProfileLoader().load(template.language_profile)
        )
        self.assertEqual(result.dialogue.total_turns, plan.total_turns)
        self.assertEqual(len(fake.received_requests), plan.total_turns)
        self.assertEqual(fake.unused_outcome_count, 1)

    def test_required_fields_are_expressed_and_removed_from_missing_progressively(self) -> None:
        original = {
            "phone": "0701234567",
            "name": "Ada",
            "selected_services": ["klippning"],
        }
        template, plan = make_template_and_plan(
            (Phase.P3, Phase.P3),
            (PhaseVariant.P3_1, PhaseVariant.P3_2),
            original=original,
            customer_facts={1: {"phone": "0701234567"}},
            assistant_facts={2: {"name": "Ada"}},
            required_fields=("phone", "name"),
        )
        fake = FakeConversationGenerator(response_factory=response_for)
        result = make_service(fake).generate(
            template, plan, LanguageProfileLoader().load(template.language_profile)
        )
        self.assertEqual(result.final_booking_state.expressed_required_fields, {"phone", "name"})
        self.assertEqual(result.final_booking_state.missing_required_fields, set())
        self.assertEqual(
            fake.received_requests[1].booking_state_before.current_values["phone"],
            "0701234567",
        )
        self.assertEqual(len(fake.received_requests[1].previous_approved_exchanges), 1)
        first_request = fake.received_requests[0]
        second_request = fake.received_requests[1]
        self.assertEqual(first_request.booking_state_before.expressed_required_fields, frozenset())
        self.assertEqual(
            first_request.booking_state_after_expected.expressed_required_fields,
            frozenset(),
        )
        self.assertEqual(
            first_request.booking_state_after_expected.missing_required_fields,
            frozenset({"phone", "name"}),
        )
        self.assertEqual(
            second_request.booking_state_before.expressed_required_fields,
            frozenset({"phone"}),
        )
        self.assertEqual(
            second_request.booking_state_before.missing_required_fields,
            frozenset({"name"}),
        )
        self.assertEqual(
            second_request.booking_state_after_expected.expressed_required_fields,
            frozenset({"phone"}),
        )
        self.assertEqual(
            second_request.booking_state_after_expected.missing_required_fields,
            frozenset({"name"}),
        )

    def test_never_expressed_required_field_remains_missing(self) -> None:
        original = {"phone": "0701234567", "selected_services": ["klippning"]}
        template, plan = make_template_and_plan(
            (Phase.P3,),
            (PhaseVariant.P3_1,),
            original=original,
            required_fields=("phone",),
        )
        result = make_service(FakeConversationGenerator(response_factory=response_for)).generate(
            template, plan, LanguageProfileLoader().load(template.language_profile)
        )
        self.assertEqual(result.final_booking_state.expressed_required_fields, set())
        self.assertEqual(result.final_booking_state.missing_required_fields, {"phone"})
        self.assertFalse(result.passed)

    def test_wrong_or_unplanned_fact_is_not_counted_as_expressed(self) -> None:
        template, plan = make_template_and_plan(
            (Phase.P3,),
            (PhaseVariant.P3_1,),
            original={"phone": "0701234567", "selected_services": ["klippning"]},
            customer_facts={1: {"phone": "0701234567"}},
            required_fields=("phone",),
        )
        planned = plan.exchanges[0]
        wrong = response_for_request_number(1).to_exchange(planned)
        object.__setattr__(wrong, "customer_facts", {"phone": "fel"})
        unplanned = response_for_request_number(1).to_exchange(planned)
        object.__setattr__(unplanned, "customer_facts", {"other": "0701234567"})
        validator = EntityValidator()
        self.assertEqual(
            validator.correctly_expressed_fields(wrong, planned, template.required_fields), set()
        )
        self.assertEqual(
            validator.correctly_expressed_fields(unplanned, planned, template.required_fields), set()
        )

    def test_wrong_mlp_in_exchange_two_stops_before_exchange_three(self) -> None:
        template, plan = make_template_and_plan(
            (Phase.P2, Phase.P3, Phase.P5),
            (PhaseVariant.P2_2, PhaseVariant.P3_1, PhaseVariant.P5_1),
        )

        def wrong_second_mlp(request: GenerationRequest) -> GenerationResponse:
            response = response_for(request)
            if request.exchange_number == 2:
                object.__setattr__(response, "customer_mlp1", "wrong")
            return response

        fake = FakeConversationGenerator(response_factory=wrong_second_mlp)
        with self.assertRaises(InvalidGenerationResponseError):
            make_service(fake).generate(
                template, plan, LanguageProfileLoader().load(template.language_profile)
            )
        self.assertEqual(len(fake.received_requests), 2)

    def test_unplanned_fact_in_exchange_two_stops_before_exchange_three(self) -> None:
        template, plan = make_template_and_plan(
            (Phase.P2, Phase.P3, Phase.P5),
            (PhaseVariant.P2_2, PhaseVariant.P3_1, PhaseVariant.P5_1),
        )

        def unplanned_second_fact(request: GenerationRequest) -> GenerationResponse:
            response = response_for(request)
            if request.exchange_number == 2:
                object.__setattr__(response, "assistant_facts", {"day": "fredag"})
            return response

        fake = FakeConversationGenerator(response_factory=unplanned_second_fact)
        with self.assertRaises(InvalidGenerationResponseError):
            make_service(fake).generate(
                template, plan, LanguageProfileLoader().load(template.language_profile)
            )
        self.assertEqual(len(fake.received_requests), 2)

    def test_wrong_p4_summary_stops_before_next_call_and_before_real_state_apply(self) -> None:
        template, plan = make_template_and_plan(
            (Phase.P4, Phase.P5),
            (PhaseVariant.P4_2, PhaseVariant.P5_1),
            changes={
                1: (
                    state_change(
                        "day", StateOperation.REPLACE, "tisdag", StateChangeSource.PHASE_VARIANT
                    ),
                )
            },
        )

        class CountingStateMachine(StateMachine):
            def __init__(self) -> None:
                self.apply_count = 0

            def apply(self, exchange: Any, state: BookingState) -> None:
                self.apply_count += 1
                super().apply(exchange, state)

        def stale_summary(request: GenerationRequest) -> GenerationResponse:
            if request.exchange_number == 1:
                return GenerationResponse(
                    1,
                    "Kund 1",
                    "AI 1",
                    "Booking",
                    "boka",
                    summary_facts={
                        "day": "måndag",
                        "time": "10:00",
                        "selected_services": ["klippning"],
                    },
                )
            return response_for(request)

        machine = CountingStateMachine()
        fake = FakeConversationGenerator(response_factory=stale_summary)
        service = ConversationGenerationService(
            fake,
            DeterministicValidationPipeline(TAXONOMY),
            POOL_DATA,
            MLP_TAXONOMY,
            state_machine=machine,
        )
        with self.assertRaises(InvalidGenerationResponseError) as caught:
            service.generate(
                template, plan, LanguageProfileLoader().load(template.language_profile)
            )
        self.assertEqual(len(fake.received_requests), 1)
        self.assertEqual(machine.apply_count, 1)
        self.assertIn(
            "P4_SUMMARY_STATE_MISMATCH",
            {issue.code for issue in caught.exception.validation_result.issues},
        )


def response_for_request_number(number: int) -> GenerationResponse:
    return GenerationResponse(
        number,
        f"Kund {number}",
        f"AI {number}",
        "Booking",
        "boka",
    )


if __name__ == "__main__":
    unittest.main()
