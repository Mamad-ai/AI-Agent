import unittest
from typing import Any, Optional, Sequence

from src.agent.state_machine import StateMachine, StateTransitionError
from src.agent.turn_plan_builder import TurnPlanBuilder
from src.models.enums import (
    DeviationType,
    LanguageProfile,
    Phase,
    PhaseVariant,
    StateChangeSource,
    StateOperation,
)
from src.models.plan import PlannedExchange, PlannedSemanticAction, PlannedStateChange
from src.models.state import BookingState
from src.models.template import (
    ConversationTemplate,
    DeviationAssignment,
    MLPExpectation,
    TemplateExchangeRequirements,
    TemplateSemanticAction,
    TemplateStateChange,
)
from src.validators.template_validator import TemplateValidator


def template_change(
    field_name: str,
    operation: StateOperation,
    value: Any,
    source: StateChangeSource,
) -> TemplateStateChange:
    return TemplateStateChange(field_name, operation, value, source)


def make_template(
    phase: Phase = Phase.P4,
    variant: PhaseVariant = PhaseVariant.P4_1,
    changes: Sequence[TemplateStateChange] = (),
    deviation: Optional[DeviationType] = None,
    original_values: Optional[dict[str, Any]] = None,
) -> ConversationTemplate:
    deviations = (
        (DeviationAssignment(1, phase, deviation),) if deviation is not None else ()
    )
    return ConversationTemplate(
        template_id="template-1",
        total_turns=1,
        language_profile=LanguageProfile.VARDAGLIG_SVENSKA,
        original_values=original_values
        if original_values is not None
        else {"day": "måndag", "time": "10:00", "selected_services": ["klippning"]},
        phase_by_exchange=(phase,),
        phase_variant_by_exchange=(variant,),
        deviations=deviations,
        mlp_expectations=(MLPExpectation(1, "Booking", "boka"),),
        exchange_requirements=(
            TemplateExchangeRequirements(
                exchange_number=1,
                required_customer_facts={},
                required_assistant_facts={},
                state_changes=tuple(changes),
            ),
        ),
    )


def codes(template: ConversationTemplate) -> set[str]:
    return {issue.code for issue in TemplateValidator().validate(template).issues}


def make_sequence_template(
    phases: Sequence[Phase],
    variants: Sequence[PhaseVariant],
    deviations: Sequence[tuple[int, DeviationType]] = (),
    changes_by_exchange: Optional[dict[int, Sequence[TemplateStateChange]]] = None,
    selected_services: Sequence[str] = ("klippning",),
) -> ConversationTemplate:
    changes_by_exchange = changes_by_exchange or {}
    return ConversationTemplate(
        template_id="sequence-template",
        total_turns=len(phases),
        language_profile=LanguageProfile.VARDAGLIG_SVENSKA,
        original_values={"selected_services": list(selected_services)},
        phase_by_exchange=tuple(phases),
        phase_variant_by_exchange=tuple(variants),
        deviations=tuple(
            DeviationAssignment(number, phases[number - 1], deviation)
            for number, deviation in deviations
        ),
        mlp_expectations=tuple(
            MLPExpectation(number, "Booking", "boka")
            for number in range(1, len(phases) + 1)
        ),
        exchange_requirements=tuple(
            TemplateExchangeRequirements(
                exchange_number=number,
                state_changes=tuple(changes_by_exchange.get(number, ())),
            )
            for number in range(1, len(phases) + 1)
        ),
    )


def planned_change(
    field_name: str,
    operation: StateOperation,
    value: Any,
    source: StateChangeSource,
) -> PlannedStateChange:
    return PlannedStateChange(field_name, operation, value, source)


def planned_exchange(
    changes: Sequence[PlannedStateChange],
    deviation: Optional[DeviationType] = None,
    variant: PhaseVariant = PhaseVariant.P4_1,
) -> PlannedExchange:
    return PlannedExchange(
        exchange_number=1,
        phase=Phase.P4,
        phase_variant=variant,
        semantic_action=PlannedSemanticAction("Booking", "boka"),
        required_customer_facts={},
        required_assistant_facts={},
        deviation=deviation,
        state_changes=tuple(changes),
    )


class TemplateValidatorTests(unittest.TestCase):
    def test_turn_plan_builds_without_optional_mlp_expectations(self) -> None:
        template = make_template()
        object.__setattr__(template, "mlp_expectations", ())
        report = TemplateValidator().validate(template)
        self.assertTrue(report.passed)
        plan = TurnPlanBuilder().build(template)
        self.assertIsNone(plan.exchanges[0].semantic_action)

    def test_explicit_domain_semantics_become_optional_planned_action(self) -> None:
        template = make_template()
        object.__setattr__(
            template,
            "planned_semantic_actions",
            (TemplateSemanticAction(1, "service_change", "add"),),
        )
        plan = TurnPlanBuilder().build(template)
        self.assertEqual(plan.exchanges[0].semantic_action.category, "service_change")
        self.assertEqual(plan.exchanges[0].semantic_action.action, "add")

    def test_legacy_mlp_hint_is_not_converted_to_planned_semantic_action(self) -> None:
        plan = TurnPlanBuilder().build(make_template())
        self.assertIsNone(plan.exchanges[0].semantic_action)

    def test_valid_p4_2_with_one_phase_variant_replace(self) -> None:
        template = make_template(
            variant=PhaseVariant.P4_2,
            changes=(
                template_change(
                    "day", StateOperation.REPLACE, "tisdag", StateChangeSource.PHASE_VARIANT
                ),
            ),
        )
        self.assertTrue(TemplateValidator().validate(template).passed)

    def test_valid_p4_3_with_multiple_phase_variant_replaces(self) -> None:
        template = make_template(
            variant=PhaseVariant.P4_3,
            changes=(
                template_change(
                    "day", StateOperation.REPLACE, "tisdag", StateChangeSource.PHASE_VARIANT
                ),
                template_change(
                    "time", StateOperation.REPLACE, "14:00", StateChangeSource.PHASE_VARIANT
                ),
            ),
        )
        self.assertTrue(TemplateValidator().validate(template).passed)

    def test_valid_d2_d3_and_d5(self) -> None:
        cases = (
            (
                DeviationType.D2,
                (template_change("day", StateOperation.REPLACE, "tisdag", StateChangeSource.DEVIATION),),
            ),
            (
                DeviationType.D3,
                (
                    template_change(
                        "selected_services",
                        StateOperation.ADD,
                        "färgning",
                        StateChangeSource.DEVIATION,
                    ),
                ),
            ),
            (
                DeviationType.D5,
                (
                    template_change(
                        "day", StateOperation.REPLACE, "tisdag", StateChangeSource.DEVIATION
                    ),
                    template_change(
                        "time", StateOperation.REPLACE, "14:00", StateChangeSource.DEVIATION
                    ),
                ),
            ),
        )
        for deviation, changes in cases:
            with self.subTest(deviation=deviation):
                self.assertTrue(
                    TemplateValidator().validate(make_template(changes=changes, deviation=deviation)).passed
                )

    def test_d1_is_valid_in_every_phase_without_state_changes(self) -> None:
        variants = {
            Phase.P1: PhaseVariant.P1_1,
            Phase.P2: PhaseVariant.P2_1,
            Phase.P3: PhaseVariant.P3_1,
            Phase.P4: PhaseVariant.P4_1,
            Phase.P5: PhaseVariant.P5_1,
        }
        for phase, variant in variants.items():
            with self.subTest(phase=phase):
                report = TemplateValidator().validate(
                    make_template(phase=phase, variant=variant, deviation=DeviationType.D1)
                )
                self.assertTrue(report.passed)

    def test_d1_with_state_change_fails(self) -> None:
        template = make_template(
            deviation=DeviationType.D1,
            changes=(
                template_change(
                    "day", StateOperation.REPLACE, "tisdag", StateChangeSource.DEVIATION
                ),
            ),
        )
        self.assertIn("D1_CHANGES_STATE", codes(template))

    def test_phase_variant_mismatch_fails(self) -> None:
        template = make_template(phase=Phase.P3, variant=PhaseVariant.P4_2)
        self.assertIn("PHASE_VARIANT_MISMATCH", codes(template))

    def test_invalid_p4_change_counts_fail(self) -> None:
        p4_2 = make_template(
            variant=PhaseVariant.P4_2,
            changes=(
                template_change("day", StateOperation.REPLACE, "tisdag", StateChangeSource.PHASE_VARIANT),
                template_change("time", StateOperation.REPLACE, "14:00", StateChangeSource.PHASE_VARIANT),
            ),
        )
        p4_3 = make_template(
            variant=PhaseVariant.P4_3,
            changes=(
                template_change("day", StateOperation.REPLACE, "tisdag", StateChangeSource.PHASE_VARIANT),
            ),
        )
        self.assertIn("INVALID_P4_2_CORRECTION", codes(p4_2))
        self.assertIn("INVALID_P4_3_CORRECTIONS", codes(p4_3))

    def test_invalid_deviation_operations_and_counts_fail(self) -> None:
        d2 = make_template(
            deviation=DeviationType.D2,
            changes=(
                template_change("selected_services", StateOperation.ADD, "färgning", StateChangeSource.DEVIATION),
            ),
        )
        d3 = make_template(
            deviation=DeviationType.D3,
            changes=(
                template_change("day", StateOperation.REPLACE, "tisdag", StateChangeSource.DEVIATION),
            ),
        )
        d5 = make_template(
            deviation=DeviationType.D5,
            changes=(
                template_change("day", StateOperation.REPLACE, "tisdag", StateChangeSource.DEVIATION),
            ),
        )
        self.assertIn("INVALID_D2_CHANGE", codes(d2))
        self.assertIn("INVALID_D3_CHANGE", codes(d3))
        self.assertIn("INVALID_D5_CHANGES", codes(d5))

    def test_d3_cannot_duplicate_selected_service(self) -> None:
        template = make_template(
            deviation=DeviationType.D3,
            changes=(
                template_change(
                    "selected_services",
                    StateOperation.ADD,
                    "klippning",
                    StateChangeSource.DEVIATION,
                ),
            ),
        )
        report = TemplateValidator().validate(template)
        self.assertIn("DUPLICATE_D3_SERVICE", {issue.code for issue in report.issues})
        self.assertTrue(all(issue.severity.value == "hard" for issue in report.issues))

    def test_p3_cannot_introduce_booking_fact_without_stateful_deviation(self) -> None:
        template = make_template(phase=Phase.P3, variant=PhaseVariant.P3_1)
        requirements = TemplateExchangeRequirements(
            exchange_number=1,
            required_customer_facts={"day": "tisdag"},
        )
        object.__setattr__(template, "exchange_requirements", (requirements,))
        self.assertIn("P3_INTRODUCES_BOOKING_FACT", codes(template))

    def test_malformed_required_fields_are_reported_without_crashing(self) -> None:
        template = make_template()
        object.__setattr__(template, "required_fields", ("day", ["not-a-field-name"]))
        self.assertIn("INVALID_REQUIRED_FIELD", codes(template))

    def test_stateful_deviations_are_rejected_in_p1_and_p2(self) -> None:
        cases = (
            (Phase.P1, PhaseVariant.P1_1, DeviationType.D2, "day", StateOperation.REPLACE),
            (
                Phase.P2,
                PhaseVariant.P2_1,
                DeviationType.D3,
                "selected_services",
                StateOperation.ADD,
            ),
            (Phase.P2, PhaseVariant.P2_1, DeviationType.D5, "day", StateOperation.REPLACE),
        )
        for phase, variant, deviation, field_name, operation in cases:
            changes = [
                template_change(field_name, operation, "nytt", StateChangeSource.DEVIATION)
            ]
            if deviation is DeviationType.D5:
                changes.append(
                    template_change(
                        "time", StateOperation.REPLACE, "14:00", StateChangeSource.DEVIATION
                    )
                )
            with self.subTest(deviation=deviation, phase=phase):
                template = make_template(
                    phase=phase,
                    variant=variant,
                    deviation=deviation,
                    changes=changes,
                )
                self.assertIn("INVALID_DEVIATION_PLACEMENT", codes(template))

    def test_two_deviations_and_repeated_deviation_types_are_allowed(self) -> None:
        d1_d1 = make_sequence_template(
            (Phase.P1, Phase.P2),
            (PhaseVariant.P1_2, PhaseVariant.P2_2),
            deviations=((1, DeviationType.D1), (2, DeviationType.D1)),
        )
        d2_d2 = make_sequence_template(
            (Phase.P3, Phase.P4),
            (PhaseVariant.P3_1, PhaseVariant.P4_1),
            deviations=((1, DeviationType.D2), (2, DeviationType.D2)),
            changes_by_exchange={
                1: (
                    template_change(
                        "day", StateOperation.REPLACE, "tisdag", StateChangeSource.DEVIATION
                    ),
                ),
                2: (
                    template_change(
                        "time", StateOperation.REPLACE, "14:00", StateChangeSource.DEVIATION
                    ),
                ),
            },
        )
        self.assertTrue(TemplateValidator().validate(d1_d1).passed)
        self.assertTrue(TemplateValidator().validate(d2_d2).passed)

    def test_three_deviations_fail(self) -> None:
        template = make_sequence_template(
            (Phase.P1, Phase.P2, Phase.P3),
            (PhaseVariant.P1_1, PhaseVariant.P2_1, PhaseVariant.P3_1),
            deviations=(
                (1, DeviationType.D1),
                (2, DeviationType.D1),
                (3, DeviationType.D1),
            ),
        )
        self.assertIn("TOO_MANY_DEVIATIONS", codes(template))

    def test_repeated_d3_uses_sequential_service_state(self) -> None:
        def d3(service: str) -> tuple[TemplateStateChange, ...]:
            return (
                template_change(
                    "selected_services", StateOperation.ADD, service, StateChangeSource.DEVIATION
                ),
            )

        valid = make_sequence_template(
            (Phase.P3, Phase.P4),
            (PhaseVariant.P3_1, PhaseVariant.P4_1),
            deviations=((1, DeviationType.D3), (2, DeviationType.D3)),
            changes_by_exchange={1: d3("skägg"), 2: d3("färgning")},
        )
        duplicate = make_sequence_template(
            (Phase.P3, Phase.P4),
            (PhaseVariant.P3_1, PhaseVariant.P4_1),
            deviations=((1, DeviationType.D3), (2, DeviationType.D3)),
            changes_by_exchange={1: d3("skägg"), 2: d3("skägg")},
        )
        fifth = make_sequence_template(
            (Phase.P3, Phase.P4),
            (PhaseVariant.P3_1, PhaseVariant.P4_1),
            deviations=((1, DeviationType.D3), (2, DeviationType.D3)),
            changes_by_exchange={1: d3("d"), 2: d3("e")},
            selected_services=("a", "b", "c"),
        )

        self.assertTrue(TemplateValidator().validate(valid).passed)
        self.assertIn("DUPLICATE_D3_SERVICE", codes(duplicate))
        self.assertIn("TOO_MANY_SERVICES", codes(fifth))

    def test_d5_requires_distinct_fields(self) -> None:
        distinct = make_template(
            phase=Phase.P3,
            variant=PhaseVariant.P3_1,
            deviation=DeviationType.D5,
            changes=(
                template_change("day", StateOperation.REPLACE, "tisdag", StateChangeSource.DEVIATION),
                template_change("time", StateOperation.REPLACE, "14:00", StateChangeSource.DEVIATION),
            ),
        )
        repeated = make_template(
            phase=Phase.P3,
            variant=PhaseVariant.P3_1,
            deviation=DeviationType.D5,
            changes=(
                template_change("day", StateOperation.REPLACE, "tisdag", StateChangeSource.DEVIATION),
                template_change("day", StateOperation.REPLACE, "onsdag", StateChangeSource.DEVIATION),
            ),
        )
        self.assertTrue(TemplateValidator().validate(distinct).passed)
        self.assertIn("INVALID_D5_CHANGES", codes(repeated))

    def test_p4_3_requires_distinct_fields(self) -> None:
        distinct = make_template(
            variant=PhaseVariant.P4_3,
            changes=(
                template_change("day", StateOperation.REPLACE, "tisdag", StateChangeSource.PHASE_VARIANT),
                template_change("time", StateOperation.REPLACE, "14:00", StateChangeSource.PHASE_VARIANT),
            ),
        )
        repeated = make_template(
            variant=PhaseVariant.P4_3,
            changes=(
                template_change("day", StateOperation.REPLACE, "tisdag", StateChangeSource.PHASE_VARIANT),
                template_change("day", StateOperation.REPLACE, "onsdag", StateChangeSource.PHASE_VARIANT),
            ),
        )
        self.assertTrue(TemplateValidator().validate(distinct).passed)
        self.assertIn("INVALID_P4_3_CORRECTIONS", codes(repeated))

    def test_explicitly_forbidden_p1_to_p2_links_fail(self) -> None:
        forbidden = (
            (PhaseVariant.P1_2, PhaseVariant.P2_1),
            (PhaseVariant.P1_4, PhaseVariant.P2_1),
            (PhaseVariant.P1_4, PhaseVariant.P2_4),
        )
        for p1_variant, p2_variant in forbidden:
            with self.subTest(p1=p1_variant, p2=p2_variant):
                template = make_sequence_template(
                    (Phase.P1, Phase.P2),
                    (p1_variant, p2_variant),
                )
                self.assertIn("INVALID_P1_P2_LINK", codes(template))

    def test_complete_p1_to_p2_transition_table(self) -> None:
        expected = {
            PhaseVariant.P1_1: {
                PhaseVariant.P2_1: False,
                PhaseVariant.P2_2: False,
                PhaseVariant.P2_3: False,
                PhaseVariant.P2_4: False,
            },
            PhaseVariant.P1_2: {
                PhaseVariant.P2_1: False,
                PhaseVariant.P2_2: True,
                PhaseVariant.P2_3: True,
                PhaseVariant.P2_4: True,
            },
            PhaseVariant.P1_4: {
                PhaseVariant.P2_1: False,
                PhaseVariant.P2_2: True,
                PhaseVariant.P2_3: True,
                PhaseVariant.P2_4: False,
            },
        }
        for p1_variant, transitions in expected.items():
            for p2_variant, should_pass in transitions.items():
                with self.subTest(p1=p1_variant, p2=p2_variant):
                    template = make_sequence_template(
                        (Phase.P1, Phase.P2),
                        (p1_variant, p2_variant),
                    )
                    report = TemplateValidator().validate(template)
                    self.assertEqual(report.passed, should_pass)

    def test_p1_3_remains_allowed_for_all_p2_variants(self) -> None:
        for p2_variant in (
            PhaseVariant.P2_1,
            PhaseVariant.P2_2,
            PhaseVariant.P2_3,
            PhaseVariant.P2_4,
        ):
            with self.subTest(p2=p2_variant):
                template = make_sequence_template(
                    (Phase.P1, Phase.P2),
                    (PhaseVariant.P1_3, p2_variant),
                )
                self.assertTrue(TemplateValidator().validate(template).passed)


class TurnPlanBuilderTests(unittest.TestCase):
    def test_builder_copies_template_without_selecting_or_changing_values(self) -> None:
        template = make_template(
            variant=PhaseVariant.P4_2,
            changes=(
                template_change("day", StateOperation.REPLACE, "tisdag", StateChangeSource.PHASE_VARIANT),
            ),
        )
        first = TurnPlanBuilder().build(template)
        second = TurnPlanBuilder().build(template)

        self.assertEqual(first, second)
        self.assertEqual(first.to_dict(), second.to_dict())
        exchange = first.exchanges[0]
        self.assertIs(exchange.phase, Phase.P4)
        self.assertIs(exchange.phase_variant, PhaseVariant.P4_2)
        self.assertIsNone(exchange.deviation)
        self.assertIs(exchange.state_changes[0].source, StateChangeSource.PHASE_VARIANT)
        self.assertEqual(exchange.state_changes[0].value, "tisdag")


class StateMachineTests(unittest.TestCase):
    def test_replace_preserves_original_and_phase_change_is_not_deviation(self) -> None:
        state = BookingState(
            original_values={"day": "måndag", "selected_services": ["klippning"]},
            current_values={"day": "måndag", "selected_services": ["klippning"]},
        )
        exchange = planned_exchange(
            (
                planned_change(
                    "day", StateOperation.REPLACE, "tisdag", StateChangeSource.PHASE_VARIANT
                ),
            ),
            variant=PhaseVariant.P4_2,
        )
        StateMachine().apply(exchange, state)

        self.assertEqual(state.original_values["day"], "måndag")
        self.assertEqual(state.current_values["day"], "tisdag")
        self.assertEqual(state.used_deviations, [])

    def test_d2_d3_and_d5_are_registered_as_deviations(self) -> None:
        state = BookingState(
            original_values={"day": "måndag", "time": "10:00", "selected_services": []},
            current_values={"day": "måndag", "time": "10:00", "selected_services": []},
        )
        exchanges = (
            planned_exchange(
                (planned_change("day", StateOperation.REPLACE, "tisdag", StateChangeSource.DEVIATION),),
                DeviationType.D2,
            ),
            planned_exchange(
                (
                    planned_change(
                        "selected_services", StateOperation.ADD, "klippning", StateChangeSource.DEVIATION
                    ),
                ),
                DeviationType.D3,
            ),
            planned_exchange(
                (
                    planned_change("day", StateOperation.REPLACE, "onsdag", StateChangeSource.DEVIATION),
                    planned_change("time", StateOperation.REPLACE, "14:00", StateChangeSource.DEVIATION),
                ),
                DeviationType.D5,
            ),
        )
        machine = StateMachine()
        for exchange in exchanges:
            machine.apply(exchange, state)
        self.assertEqual(state.used_deviations, [DeviationType.D2, DeviationType.D3, DeviationType.D5])

    def test_duplicate_and_fifth_services_fail_atomically(self) -> None:
        machine = StateMachine()
        duplicate_state = BookingState(
            original_values={"selected_services": ["klippning"]},
            current_values={"selected_services": ["klippning"]},
        )
        duplicate = planned_exchange(
            (
                planned_change(
                    "selected_services", StateOperation.ADD, "klippning", StateChangeSource.DEVIATION
                ),
            ),
            DeviationType.D3,
        )
        with self.assertRaisesRegex(StateTransitionError, "already selected"):
            machine.apply(duplicate, duplicate_state)
        self.assertEqual(duplicate_state.selected_services, ("klippning",))
        self.assertEqual(duplicate_state.used_deviations, [])

        full_state = BookingState(
            original_values={"selected_services": ["a", "b", "c", "d"]},
            current_values={"selected_services": ["a", "b", "c", "d"]},
        )
        fifth = planned_exchange(
            (
                planned_change(
                    "selected_services", StateOperation.ADD, "e", StateChangeSource.DEVIATION
                ),
            ),
            DeviationType.D3,
        )
        with self.assertRaisesRegex(StateTransitionError, "at most four"):
            machine.apply(fifth, full_state)
        self.assertEqual(full_state.selected_services, ("a", "b", "c", "d"))

    def test_fourth_service_is_allowed(self) -> None:
        state = BookingState(
            original_values={"selected_services": ["a", "b", "c"]},
            current_values={"selected_services": ["a", "b", "c"]},
        )
        fourth = planned_exchange(
            (
                planned_change(
                    "selected_services", StateOperation.ADD, "d", StateChangeSource.DEVIATION
                ),
            ),
            DeviationType.D3,
        )
        StateMachine().apply(fourth, state)
        self.assertEqual(state.selected_services, ("a", "b", "c", "d"))


if __name__ == "__main__":
    unittest.main()
