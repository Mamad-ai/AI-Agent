"""Deterministic builders for semantic validation requests."""

from typing import Optional

from src.ai.language_profiles import LanguageProfileConfig
from src.ai.schemas import (
    BookingStateSnapshot,
    SemanticDialogueValidationRequest,
    SemanticTurnValidationRequest,
)
from src.models.dialogue import ConversationExchange, Dialogue
from src.models.input_bundle import MLPTaxonomy
from src.models.plan import PlannedExchange, TurnPlan
from src.models.state import BookingState


class SemanticTurnRequestBuilder:
    def build(
        self,
        planned: PlannedExchange,
        generated: ConversationExchange,
        profile: LanguageProfileConfig,
        mlp_taxonomy: MLPTaxonomy,
        state_before: BookingState,
        state_after: BookingState,
        previous_exchange: Optional[ConversationExchange] = None,
    ) -> SemanticTurnValidationRequest:
        if generated.exchange_number != planned.exchange_number:
            raise ValueError("generated and planned exchange numbers must match")
        nearby = (previous_exchange,) if previous_exchange is not None else ()
        return SemanticTurnValidationRequest(
            planned_exchange=planned,
            generated_exchange=generated,
            language_profile=profile.profile,
            language_profile_config=profile,
            mlp_taxonomy=mlp_taxonomy,
            state_before=BookingStateSnapshot.from_booking_state(state_before),
            state_after=BookingStateSnapshot.from_booking_state(state_after),
            nearby_exchanges=nearby,
        )


class SemanticDialogueRequestBuilder:
    def build(
        self,
        dialogue: Dialogue,
        plan: TurnPlan,
        final_state: BookingState,
        profile: LanguageProfileConfig,
        mlp_taxonomy: MLPTaxonomy,
    ) -> SemanticDialogueValidationRequest:
        if dialogue.template_id != plan.template_id:
            raise ValueError("dialogue and plan IDs must match")
        if dialogue.language_profile is not profile.profile:
            raise ValueError("language profile config does not match dialogue")
        return SemanticDialogueValidationRequest(
            dialogue=dialogue,
            turn_plan=plan,
            final_booking_state=BookingStateSnapshot.from_booking_state(final_state),
            language_profile=profile.profile,
            language_profile_config=profile,
            mlp_taxonomy=mlp_taxonomy,
        )
