import unittest
import json
from dataclasses import replace
from typing import Any, Optional, Sequence

from src.agent.state_machine import StateMachine
from src.agent.turn_plan_builder import TurnPlanBuilder
from src.models.dialogue import ConversationExchange, Dialogue, Utterance
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
from src.validators.mlp_structure_validator import MLPStructureValidator
from src.validators.state_validator import StateValidator
from src.validators.structure_validator import StructureValidator
from src.validators.summary_validator import SummaryValidator


TAXONOMY = {
    "Booking": frozenset({"boka", "omboka", "avboka"}),
    "FAQ": frozenset(),
    "Tjänster": frozenset({"lägga till", "ta bort", "byta ut"}),
    "SmallTalk": frozenset(),
    "Auth": frozenset({"skapa", "ändra"}),
}


def make_scenario(
    *,
    phase: Phase = Phase.P2,
    variant: PhaseVariant = PhaseVariant.P2_2,
    deviation: Optional[DeviationType] = None,
    changes: Sequence[TemplateStateChange] = (),
    original_values: Optional[dict[str, Any]] = None,
    required_customer_facts: Optional[dict[str, Any]] = None,
    required_assistant_facts: Optional[dict[str, Any]] = None,
    allowed_customer_facts: Optional[dict[str, Any]] = None,
    allowed_assistant_facts: Optional[dict[str, Any]] = None,
    required_fields: Sequence[str] = (),
) -> tuple[ConversationTemplate, TurnPlan, Dialogue, BookingState]:
    original = original_values or {
        "day": "måndag",
        "time": "10:00",
        "name": "Ada",
        "phone": "0701234567",
        "selected_services": ["klippning"],
    }
    customer_facts = required_customer_facts or {}
    assistant_facts = required_assistant_facts or {}
    template = ConversationTemplate(
        template_id="template-1",
        total_turns=1,
        language_profile=LanguageProfile.VARDAGLIG_SVENSKA,
        original_values=original,
        phase_by_exchange=(phase,),
        phase_variant_by_exchange=(variant,),
        available_services=("klippning", "färgning", "skägg", "tvätt"),
        required_fields=tuple(required_fields),
        deviations=(DeviationAssignment(1, phase, deviation),) if deviation else (),
        mlp_expectations=(MLPExpectation(1, "Booking", "boka"),),
        exchange_requirements=(
            TemplateExchangeRequirements(
                1,
                required_customer_facts=customer_facts,
                required_assistant_facts=assistant_facts,
                allowed_customer_facts=allowed_customer_facts or customer_facts,
                allowed_assistant_facts=allowed_assistant_facts or assistant_facts,
                state_changes=tuple(changes),
            ),
        ),
    )
    plan = TurnPlanBuilder().build(template)
    state = BookingState(template.original_values, thaw_value(template.original_values))
    StateMachine().apply(plan.exchanges[0], state)
    exchange = ConversationExchange(
        1,
        Utterance(Speaker.CUSTOMER, "Kundtext"),
        Utterance(Speaker.ASSISTANT, "AI-text"),
        phase=phase,
        phase_variant=variant,
        customer_mlp1="Booking",
        customer_mlp2="boka",
        customer_facts=customer_facts,
        assistant_facts=assistant_facts,
        summary_facts=state.current_values if phase is Phase.P4 else {},
    )
    dialogue = Dialogue(template.template_id, template.language_profile, (exchange,))
    return template, plan, dialogue, state


def change(
    field_name: str,
    operation: StateOperation,
    value: Any,
    source: StateChangeSource,
) -> TemplateStateChange:
    return TemplateStateChange(field_name, operation, value, source)


class StructureValidatorTests(unittest.TestCase):
    def test_exact_exchange_count_passes(self) -> None:
        template, plan, dialogue, _ = make_scenario()
        self.assertTrue(StructureValidator().validate(dialogue, plan, template).passed)
        self.assertEqual(dialogue.total_turns, 1)
        self.assertEqual(dialogue.utterance_count, 2)

    def test_missing_and_extra_exchanges_are_hard_failures(self) -> None:
        template, plan, dialogue, _ = make_scenario()
        missing = Dialogue(template.template_id, template.language_profile, ())
        extra_exchange = replace(dialogue.exchanges[0], exchange_number=2)
        extra = Dialogue(
            template.template_id,
            template.language_profile,
            (dialogue.exchanges[0], extra_exchange),
        )
        for candidate in (missing, extra):
            with self.subTest(turns=candidate.total_turns):
                report = StructureValidator().validate(candidate, plan, template)
                self.assertFalse(report.passed)
                self.assertTrue(report.has_hard_failures)

    def test_wrong_speaker_order_fails(self) -> None:
        template, plan, dialogue, _ = make_scenario()
        object.__setattr__(dialogue.exchanges[0].customer, "speaker", Speaker.ASSISTANT)
        report = StructureValidator().validate(dialogue, plan, template)
        self.assertFalse(report.passed)

    def test_wrong_template_id_and_language_profile_fail(self) -> None:
        template, plan, dialogue, _ = make_scenario()
        wrong_id = Dialogue("wrong", dialogue.language_profile, dialogue.exchanges)
        wrong_profile = Dialogue(
            dialogue.template_id, LanguageProfile.FORMELL_SVENSKA, dialogue.exchanges
        )
        self.assertFalse(StructureValidator().validate(wrong_id, plan, template).passed)
        self.assertFalse(StructureValidator().validate(wrong_profile, plan, template).passed)

    def test_matching_plan_version_passes_and_mismatch_fails(self) -> None:
        template, plan, dialogue, _ = make_scenario()
        self.assertEqual(dialogue.plan_version, plan.plan_version)
        self.assertTrue(StructureValidator().validate(dialogue, plan, template).passed)
        object.__setattr__(dialogue, "plan_version", "old-plan")
        report = StructureValidator().validate(dialogue, plan, template)
        self.assertIn("PLAN_VERSION_MISMATCH", {issue.code for issue in report.issues})
        self.assertTrue(report.has_hard_failures)


class StateValidatorTests(unittest.TestCase):
    def test_correct_d2_d3_and_d5_states_pass(self) -> None:
        cases = (
            (
                PhaseVariant.P3_1,
                DeviationType.D2,
                (change("day", StateOperation.REPLACE, "tisdag", StateChangeSource.DEVIATION),),
            ),
            (
                PhaseVariant.P3_1,
                DeviationType.D3,
                (
                    change(
                        "selected_services",
                        StateOperation.ADD,
                        "färgning",
                        StateChangeSource.DEVIATION,
                    ),
                ),
            ),
            (
                PhaseVariant.P3_1,
                DeviationType.D5,
                (
                    change("day", StateOperation.REPLACE, "tisdag", StateChangeSource.DEVIATION),
                    change("time", StateOperation.REPLACE, "14:00", StateChangeSource.DEVIATION),
                ),
            ),
        )
        for variant, deviation, changes in cases:
            with self.subTest(deviation=deviation):
                template, plan, _, state = make_scenario(
                    phase=Phase.P3, variant=variant, deviation=deviation, changes=changes
                )
                self.assertTrue(StateValidator().validate(plan, template, state).passed)

    def test_d1_state_change_fails(self) -> None:
        template, plan, _, state = make_scenario(
            phase=Phase.P3, variant=PhaseVariant.P3_1, deviation=DeviationType.D1
        )
        invalid = PlannedStateChange(
            "day", StateOperation.REPLACE, "tisdag", StateChangeSource.DEVIATION
        )
        object.__setattr__(plan.exchanges[0], "state_changes", (invalid,))
        report = StateValidator().validate(plan, template, state)
        self.assertIn("D1_CHANGES_STATE", {issue.code for issue in report.issues})

    def test_p4_correction_registered_as_deviation_fails(self) -> None:
        template, plan, _, state = make_scenario(
            phase=Phase.P4,
            variant=PhaseVariant.P4_2,
            changes=(
                change("day", StateOperation.REPLACE, "tisdag", StateChangeSource.PHASE_VARIANT),
            ),
        )
        state.used_deviations.append(DeviationType.D2)
        self.assertFalse(StateValidator().validate(plan, template, state).passed)

    def test_changed_original_values_fail(self) -> None:
        template, plan, _, state = make_scenario()
        state.original_values = {"day": "tisdag"}
        self.assertFalse(StateValidator().validate(plan, template, state).passed)

    def test_duplicate_d3_and_five_services_fail(self) -> None:
        template, plan, _, state = make_scenario(
            phase=Phase.P3,
            variant=PhaseVariant.P3_1,
            deviation=DeviationType.D3,
            changes=(
                change(
                    "selected_services",
                    StateOperation.ADD,
                    "färgning",
                    StateChangeSource.DEVIATION,
                ),
            ),
        )
        duplicate = PlannedStateChange(
            "selected_services", StateOperation.ADD, "klippning", StateChangeSource.DEVIATION
        )
        object.__setattr__(plan.exchanges[0], "state_changes", (duplicate,))
        self.assertFalse(StateValidator().validate(plan, template, state).passed)

        _, full_plan, _, full_state = make_scenario()
        full_state.current_values["selected_services"] = ["a", "b", "c", "d", "e"]
        self.assertFalse(StateValidator().validate(full_plan, template, full_state).passed)


class EntityAndMLPValidatorTests(unittest.TestCase):
    def test_required_structured_value_correct_and_missing_or_changed(self) -> None:
        template, plan, dialogue, state = make_scenario(
            required_customer_facts={"day": "måndag"}, required_fields=("day",)
        )
        validator = EntityValidator()
        self.assertTrue(validator.validate(dialogue, plan, template, state).passed)

        object.__setattr__(dialogue.exchanges[0], "customer_facts", {})
        self.assertFalse(validator.validate(dialogue, plan, template, state).passed)
        object.__setattr__(dialogue.exchanges[0], "customer_facts", {"day": "tisdag"})
        self.assertIn(
            "STRUCTURED_FACT_CHANGED",
            {issue.code for issue in validator.validate(dialogue, plan, template, state).issues},
        )

    def test_missing_required_current_value_fails(self) -> None:
        template, plan, dialogue, state = make_scenario(required_fields=("phone",))
        del state.current_values["phone"]
        self.assertFalse(EntityValidator().validate(dialogue, plan, template, state).passed)

    def test_allowed_extra_customer_fact_with_planned_value_passes(self) -> None:
        template, plan, dialogue, state = make_scenario(
            required_customer_facts={"selected_services": ["klippning"]},
            allowed_customer_facts={
                "selected_services": ["klippning"],
                "day": "måndag",
            },
        )
        object.__setattr__(
            dialogue.exchanges[0],
            "customer_facts",
            {"selected_services": ["klippning"], "day": "måndag"},
        )
        self.assertTrue(EntityValidator().validate(dialogue, plan, template, state).passed)

    def test_unplanned_customer_and_assistant_facts_are_hard_failures(self) -> None:
        template, plan, dialogue, state = make_scenario()
        object.__setattr__(dialogue.exchanges[0], "customer_facts", {"day": "fredag"})
        object.__setattr__(dialogue.exchanges[0], "assistant_facts", {"time": "14:00"})
        report = EntityValidator().validate(dialogue, plan, template, state)
        unplanned = [issue for issue in report.issues if issue.code == "UNPLANNED_STRUCTURED_FACT"]
        self.assertEqual(len(unplanned), 2)
        self.assertTrue(report.has_hard_failures)

    def test_allowed_fact_with_wrong_value_fails(self) -> None:
        template, plan, dialogue, state = make_scenario(
            allowed_customer_facts={"day": "måndag"}
        )
        object.__setattr__(dialogue.exchanges[0], "customer_facts", {"day": "fredag"})
        report = EntityValidator().validate(dialogue, plan, template, state)
        self.assertIn("ALLOWED_FACT_VALUE_CHANGED", {issue.code for issue in report.issues})

    def test_structurally_empty_required_values_fail(self) -> None:
        cases = (
            ("phone", None),
            ("name", ""),
            ("selected_services", []),
        )
        for field_name, empty_value in cases:
            with self.subTest(field=field_name):
                values = {
                    "phone": "0701234567",
                    "name": "Ada",
                    "selected_services": ["klippning"],
                }
                values[field_name] = empty_value
                template, plan, dialogue, state = make_scenario(
                    original_values=values,
                    required_fields=(field_name,),
                )
                report = EntityValidator().validate(dialogue, plan, template, state)
                self.assertIn("MISSING_REQUIRED_VALUE", {issue.code for issue in report.issues})

    def test_fact_models_remain_json_serializable(self) -> None:
        template, plan, dialogue, _ = make_scenario(
            required_customer_facts={"day": "måndag"},
            allowed_customer_facts={"day": "måndag", "time": "10:00"},
            allowed_assistant_facts={"name": "Ada"},
        )
        json.dumps(template.to_dict(), ensure_ascii=False)
        json.dumps(plan.to_dict(), ensure_ascii=False)
        json.dumps(dialogue.to_dict(), ensure_ascii=False)

    def test_valid_invalid_and_missing_mlp_labels(self) -> None:
        template, plan, dialogue, _ = make_scenario()
        validator = MLPStructureValidator(TAXONOMY)
        self.assertTrue(validator.validate(dialogue, plan).passed)
        invalid = MLPStructureValidator({"Booking": frozenset({"avboka"})})
        self.assertFalse(invalid.validate(dialogue, plan).passed)
        object.__setattr__(dialogue.exchanges[0], "customer_mlp2", None)
        self.assertFalse(validator.validate(dialogue, plan).passed)

    def test_all_taxonomy_shapes_are_validated_without_plan_label_facit(self) -> None:
        template, plan, dialogue, _ = make_scenario()
        validator = MLPStructureValidator(TAXONOMY)
        valid = (
            ("Booking", "boka"),
            ("Booking", "omboka"),
            ("Booking", "avboka"),
            ("FAQ", None),
            ("Tjänster", "lägga till"),
            ("Tjänster", "ta bort"),
            ("Tjänster", "byta ut"),
            ("SmallTalk", None),
            ("Auth", "skapa"),
            ("Auth", "ändra"),
        )
        for mlp1, mlp2 in valid:
            with self.subTest(mlp1=mlp1, mlp2=mlp2):
                object.__setattr__(dialogue.exchanges[0], "customer_mlp1", mlp1)
                object.__setattr__(dialogue.exchanges[0], "customer_mlp2", mlp2)
                self.assertTrue(validator.validate(dialogue, plan).passed)

        invalid = (
            ("FAQ", "boka"),
            ("SmallTalk", "ändra"),
            ("Booking", None),
            ("Auth", "boka"),
            ("Okänd", None),
        )
        for mlp1, mlp2 in invalid:
            with self.subTest(mlp1=mlp1, mlp2=mlp2):
                object.__setattr__(dialogue.exchanges[0], "customer_mlp1", mlp1)
                object.__setattr__(dialogue.exchanges[0], "customer_mlp2", mlp2)
                self.assertFalse(validator.validate(dialogue, plan).passed)


class SummaryAndPipelineTests(unittest.TestCase):
    def test_p4_summary_uses_current_not_original_state(self) -> None:
        template, plan, dialogue, _ = make_scenario(
            phase=Phase.P4,
            variant=PhaseVariant.P4_2,
            changes=(
                change("day", StateOperation.REPLACE, "tisdag", StateChangeSource.PHASE_VARIANT),
            ),
        )
        validator = SummaryValidator()
        self.assertTrue(validator.validate(dialogue, plan, template).passed)
        old_summary = thaw_value(template.original_values)
        object.__setattr__(dialogue.exchanges[0], "summary_facts", old_summary)
        self.assertFalse(validator.validate(dialogue, plan, template).passed)

    def test_pipeline_collects_reports_and_hard_failure_controls_result(self) -> None:
        template, plan, dialogue, state = make_scenario()
        pipeline = DeterministicValidationPipeline(TAXONOMY)
        passed = pipeline.validate(dialogue, plan, template, state)
        self.assertTrue(passed.passed)
        self.assertEqual(len(passed.reports), 5)

        object.__setattr__(dialogue, "template_id", "wrong")
        failed = pipeline.validate(dialogue, plan, template, state)
        self.assertFalse(failed.passed)
        self.assertTrue(failed.has_hard_failures)
        self.assertGreaterEqual(len(failed.issues), 1)

    def test_validators_do_not_mutate_dialogue_or_template(self) -> None:
        template, plan, dialogue, state = make_scenario()
        template_before = template.to_dict()
        dialogue_before = dialogue.to_dict()
        DeterministicValidationPipeline(TAXONOMY).validate(dialogue, plan, template, state)
        self.assertEqual(template.to_dict(), template_before)
        self.assertEqual(dialogue.to_dict(), dialogue_before)


if __name__ == "__main__":
    unittest.main()
