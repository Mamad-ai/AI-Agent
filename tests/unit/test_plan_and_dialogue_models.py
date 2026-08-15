import json
import unittest

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
from src.models.plan import PlannedExchange, PlannedSemanticAction, PlannedStateChange, TurnPlan


def planned_exchange(number: int) -> PlannedExchange:
    return PlannedExchange(
        exchange_number=number,
        phase=Phase.P2,
        phase_variant=PhaseVariant.P2_1,
        semantic_action=PlannedSemanticAction("Booking", "boka"),
        required_customer_facts={},
        required_assistant_facts={},
    )


class TurnPlanTests(unittest.TestCase):
    def test_planned_exchange_allows_no_semantic_action(self) -> None:
        exchange = PlannedExchange(
            exchange_number=1,
            phase=Phase.P2,
            phase_variant=PhaseVariant.P2_2,
            required_customer_facts={},
            required_assistant_facts={},
        )
        self.assertIsNone(exchange.semantic_action)
        self.assertIsNone(exchange.to_dict()["semantic_action"])

    def test_turn_plan_counts_complete_exchanges(self) -> None:
        exchanges = tuple(planned_exchange(index) for index in range(1, 15))
        plan = TurnPlan("template-14", 14, exchanges)
        self.assertEqual(plan.total_turns, 14)
        self.assertEqual(len(plan.exchanges), 14)

    def test_d1_cannot_plan_a_state_change(self) -> None:
        change = PlannedStateChange(
            "day", StateOperation.REPLACE, "tisdag", StateChangeSource.DEVIATION
        )
        with self.assertRaisesRegex(ValueError, "D1"):
            PlannedExchange(
                exchange_number=1,
                phase=Phase.P3,
                phase_variant=PhaseVariant.P3_1,
                semantic_action=PlannedSemanticAction("FAQ"),
                required_customer_facts={},
                required_assistant_facts={},
                deviation=DeviationType.D1,
                state_changes=(change,),
            )

    def test_deviation_sourced_change_requires_explicit_deviation(self) -> None:
        with self.assertRaisesRegex(ValueError, "require an explicit deviation"):
            PlannedExchange(
                exchange_number=1,
                phase=Phase.P3,
                phase_variant=PhaseVariant.P3_1,
                semantic_action=PlannedSemanticAction("Auth", "skapa"),
                required_customer_facts={"name": "Ada"},
                required_assistant_facts={},
                state_changes=(
                    PlannedStateChange(
                        "day",
                        StateOperation.REPLACE,
                        "tisdag",
                        StateChangeSource.DEVIATION,
                    ),
                ),
            )

    def test_d2_and_d3_preserve_operation_for_later_strict_validation(self) -> None:
        d2 = PlannedExchange(
            exchange_number=1,
            phase=Phase.P4,
            phase_variant=PhaseVariant.P4_1,
            semantic_action=PlannedSemanticAction("Booking", "omboka"),
            required_customer_facts={},
            required_assistant_facts={},
            deviation=DeviationType.D2,
            state_changes=(
                PlannedStateChange(
                    "day", StateOperation.REPLACE, "tisdag", StateChangeSource.DEVIATION
                ),
            ),
        )
        d3 = PlannedExchange(
            exchange_number=2,
            phase=Phase.P4,
            phase_variant=PhaseVariant.P4_1,
            semantic_action=PlannedSemanticAction("Tjänster", "lägga till"),
            required_customer_facts={},
            required_assistant_facts={},
            deviation=DeviationType.D3,
            state_changes=(
                PlannedStateChange(
                    "selected_services",
                    StateOperation.ADD,
                    "färgning",
                    StateChangeSource.DEVIATION,
                ),
            ),
        )

        self.assertIs(d2.deviation, DeviationType.D2)
        self.assertIs(d2.state_changes[0].operation, StateOperation.REPLACE)
        self.assertEqual(d2.to_dict()["state_changes"][0]["operation"], "replace")
        self.assertEqual(d2.to_dict()["state_changes"][0]["source"], "deviation")
        self.assertIs(d3.deviation, DeviationType.D3)
        self.assertIs(d3.state_changes[0].operation, StateOperation.ADD)
        self.assertEqual(d3.to_dict()["state_changes"][0]["operation"], "add")

    def test_d5_preserves_multiple_replacements_for_later_validation(self) -> None:
        exchange = PlannedExchange(
            exchange_number=1,
            phase=Phase.P3,
            phase_variant=PhaseVariant.P3_1,
            semantic_action=PlannedSemanticAction("Booking", "omboka"),
            required_customer_facts={},
            required_assistant_facts={},
            deviation=DeviationType.D5,
            state_changes=(
                PlannedStateChange(
                    "day", StateOperation.REPLACE, "tisdag", StateChangeSource.DEVIATION
                ),
                PlannedStateChange(
                    "time", StateOperation.REPLACE, "14:00", StateChangeSource.DEVIATION
                ),
            ),
        )

        self.assertIs(exchange.deviation, DeviationType.D5)
        self.assertEqual(len(exchange.state_changes), 2)
        self.assertTrue(
            all(change.operation is StateOperation.REPLACE for change in exchange.state_changes)
        )

    def test_p4_2_can_replace_state_without_deviation(self) -> None:
        exchange = PlannedExchange(
            exchange_number=1,
            phase=Phase.P4,
            phase_variant=PhaseVariant.P4_2,
            semantic_action=PlannedSemanticAction("Booking", "omboka"),
            required_customer_facts={},
            required_assistant_facts={},
            state_changes=(
                PlannedStateChange(
                    "day",
                    StateOperation.REPLACE,
                    "tisdag",
                    StateChangeSource.PHASE_VARIANT,
                ),
            ),
        )

        self.assertIsNone(exchange.deviation)
        self.assertIs(exchange.state_changes[0].source, StateChangeSource.PHASE_VARIANT)

    def test_p4_3_can_replace_multiple_fields_without_deviation(self) -> None:
        exchange = PlannedExchange(
            exchange_number=1,
            phase=Phase.P4,
            phase_variant=PhaseVariant.P4_3,
            semantic_action=PlannedSemanticAction("Booking", "omboka"),
            required_customer_facts={},
            required_assistant_facts={},
            state_changes=(
                PlannedStateChange(
                    "day",
                    StateOperation.REPLACE,
                    "tisdag",
                    StateChangeSource.PHASE_VARIANT,
                ),
                PlannedStateChange(
                    "time",
                    StateOperation.REPLACE,
                    "14:00",
                    StateChangeSource.PHASE_VARIANT,
                ),
            ),
        )

        self.assertIsNone(exchange.deviation)
        self.assertEqual(len(exchange.state_changes), 2)
        self.assertTrue(
            all(change.source is StateChangeSource.PHASE_VARIANT for change in exchange.state_changes)
        )
        payload = exchange.to_dict()
        self.assertEqual(payload["phase"], "P4")
        self.assertEqual(payload["phase_variant"], "P4.3")

    def test_planned_exchange_rejects_variant_from_another_phase(self) -> None:
        with self.assertRaisesRegex(ValueError, "P4.2 does not belong to P3"):
            PlannedExchange(
                exchange_number=1,
                phase=Phase.P3,
                phase_variant=PhaseVariant.P4_2,
                semantic_action=PlannedSemanticAction("Auth", "skapa"),
                required_customer_facts={},
                required_assistant_facts={},
            )


class DialogueTests(unittest.TestCase):
    def test_fourteen_turns_are_twenty_eight_utterances(self) -> None:
        exchanges = tuple(
            ConversationExchange(
                index,
                Utterance(Speaker.CUSTOMER, f"Kund {index}"),
                Utterance(Speaker.ASSISTANT, f"AI {index}"),
            )
            for index in range(1, 15)
        )
        dialogue = Dialogue(
            template_id="template-14",
            language_profile=LanguageProfile.VARDAGLIG_SVENSKA,
            exchanges=exchanges,
        )
        self.assertEqual(dialogue.total_turns, 14)
        self.assertEqual(dialogue.utterance_count, 28)
        self.assertEqual(len(dialogue.to_dict()["exchanges"]), 14)

    def test_exchange_enforces_customer_then_assistant_roles(self) -> None:
        with self.assertRaisesRegex(ValueError, "customer speaker"):
            ConversationExchange(
                1,
                Utterance(Speaker.ASSISTANT, "Fel roll"),
                Utterance(Speaker.ASSISTANT, "Svar"),
            )

    def test_dialogue_is_json_serializable(self) -> None:
        dialogue = Dialogue(
            template_id="json",
            language_profile=LanguageProfile.FORMELL_SVENSKA,
            exchanges=(
                ConversationExchange(
                    1,
                    Utterance(Speaker.CUSTOMER, "Hej"),
                    Utterance(Speaker.ASSISTANT, "Välkommen"),
                ),
            ),
        )
        json.dumps(dialogue.to_dict(), ensure_ascii=False)


if __name__ == "__main__":
    unittest.main()
