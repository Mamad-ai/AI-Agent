import json
import unittest
from dataclasses import FrozenInstanceError

from src.models.enums import DeviationType, LanguageProfile, Phase, PhaseVariant
from src.models.template import (
    ConversationTemplate,
    DeviationAssignment,
    MLPExpectation,
    TemplateExchangeRequirements,
    TemplateSemanticAction,
)
from src.validators.template_validator import TemplateValidator


class ConversationTemplateTests(unittest.TestCase):
    def test_template_without_mlp_expectations_is_valid(self) -> None:
        template = ConversationTemplate(
            template_id="without-mlp-facit",
            total_turns=1,
            language_profile=LanguageProfile.VARDAGLIG_SVENSKA,
            original_values={"selected_services": ["klippning"]},
            phase_by_exchange=(Phase.P2,),
            phase_variant_by_exchange=(PhaseVariant.P2_2,),
            exchange_requirements=(TemplateExchangeRequirements(1),),
        )
        self.assertTrue(TemplateValidator().validate(template).passed)

    def test_template_is_immutable_and_deep_freezes_source_values(self) -> None:
        source_values = {"day": "måndag", "contact": {"name": "Ada"}}
        template = ConversationTemplate(
            template_id="template-1",
            total_turns=1,
            language_profile=LanguageProfile.VARDAGLIG_SVENSKA,
            original_values=source_values,
            phase_by_exchange=(Phase.P1,),
            phase_variant_by_exchange=(PhaseVariant.P1_1,),
        )

        source_values["day"] = "tisdag"
        source_values["contact"]["name"] = "Eve"

        self.assertEqual(template.original_values["day"], "måndag")
        self.assertEqual(template.original_values["contact"]["name"], "Ada")
        with self.assertRaises(TypeError):
            template.original_values["day"] = "onsdag"  # type: ignore[index]
        with self.assertRaises(FrozenInstanceError):
            template.total_turns = 2  # type: ignore[misc]

    def test_d1_can_be_explicitly_assigned_to_every_phase(self) -> None:
        phases = tuple(Phase)
        deviations = tuple(
            DeviationAssignment(index, phase, DeviationType.D1)
            for index, phase in enumerate(phases, start=1)
        )
        template = ConversationTemplate(
            template_id="d1-all-phases",
            total_turns=5,
            language_profile=LanguageProfile.FORMELL_SVENSKA,
            original_values={},
            phase_by_exchange=phases,
            phase_variant_by_exchange=(
                PhaseVariant.P1_1,
                PhaseVariant.P2_1,
                PhaseVariant.P3_1,
                PhaseVariant.P4_1,
                PhaseVariant.P5_1,
            ),
            deviations=deviations,
        )

        self.assertEqual(len(template.deviations), 5)
        self.assertTrue(all(not item.deviation_type.changes_booking_state for item in deviations))

    def test_mlp_expectation_belongs_to_customer_side_of_exchange(self) -> None:
        expected = MLPExpectation(1, "Booking", "boka")
        self.assertEqual(expected.exchange_number, 1)
        self.assertEqual(expected.mlp1, "Booking")

    def test_mlp_expectation_supports_semantic_action_without_subaction(self) -> None:
        expected = MLPExpectation(1, "FAQ", None)
        self.assertIsNone(expected.mlp2)
        self.assertIsNone(expected.to_dict()["mlp2"])

    def test_explicit_template_semantic_action_is_separate_from_mlp_hint(self) -> None:
        action = TemplateSemanticAction(1, "authentication", "create")
        self.assertEqual(
            action.to_dict(),
            {"exchange_number": 1, "domain": "authentication", "action": "create"},
        )

    def test_template_to_dict_is_json_serializable(self) -> None:
        template = ConversationTemplate(
            template_id="serializable",
            total_turns=1,
            language_profile=LanguageProfile.ORTEN_SVENSKA,
            original_values={"selected_services": ["klippning"]},
            phase_by_exchange=(Phase.P2,),
            phase_variant_by_exchange=(PhaseVariant.P2_2,),
            available_services=("klippning", "färgning"),
        )
        payload = template.to_dict()
        encoded = json.dumps(payload, ensure_ascii=False)
        self.assertIn("klippning", encoded)
        self.assertNotIn("selected_services", payload)
        self.assertEqual(payload["original_values"]["selected_services"], ["klippning"])
        self.assertEqual(payload["phase_by_exchange"], ["P2"])
        self.assertEqual(payload["phase_variant_by_exchange"], ["P2.2"])

    def test_invalid_phase_count_can_be_represented_for_structured_validation(self) -> None:
        template = ConversationTemplate(
            template_id="bad",
            total_turns=2,
            language_profile=LanguageProfile.BRYTANDE_SVENSKA,
            original_values={},
            phase_by_exchange=(Phase.P1,),
            phase_variant_by_exchange=(PhaseVariant.P1_1,),
        )
        self.assertEqual(template.total_turns, 2)

    def test_phase_variant_must_belong_to_main_phase(self) -> None:
        self.assertTrue(PhaseVariant.P4_2.belongs_to(Phase.P4))
        self.assertFalse(PhaseVariant.P4_2.belongs_to(Phase.P3))
        template = ConversationTemplate(
            template_id="wrong-variant",
            total_turns=1,
            language_profile=LanguageProfile.FORMELL_SVENSKA,
            original_values={},
            phase_by_exchange=(Phase.P3,),
            phase_variant_by_exchange=(PhaseVariant.P4_2,),
        )
        self.assertIs(template.phase_variant_by_exchange[0], PhaseVariant.P4_2)


if __name__ == "__main__":
    unittest.main()
