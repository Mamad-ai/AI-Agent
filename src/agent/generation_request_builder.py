"""Deterministic construction of one generation request."""

from typing import Sequence

from src.ai.language_profiles import LanguageProfileConfig
from src.ai.schemas import BookingStateSnapshot, GenerationRequest
from src.agent.state_machine import StateMachine
from src.models.dialogue import ConversationExchange
from src.models.input_bundle import MLPTaxonomy, PoolData
from src.models.plan import PlannedExchange, TurnPlan
from src.models.state import BookingState
from src.models.template import ConversationTemplate, thaw_value


class GenerationRequestBuilder:
    def __init__(self, state_machine: StateMachine | None = None) -> None:
        self._state_machine = state_machine or StateMachine()

    def build(
        self,
        template: ConversationTemplate,
        plan: TurnPlan,
        exchange: PlannedExchange,
        state_before: BookingState,
        previous_approved_exchanges: Sequence[ConversationExchange],
        language_profile_config: LanguageProfileConfig,
        pool_data: PoolData,
        mlp_taxonomy: MLPTaxonomy,
        applicable_rules: Sequence[str] = (),
    ) -> GenerationRequest:
        if plan.template_id != template.template_id:
            raise ValueError("turn plan and template IDs must match")
        if exchange.exchange_number > plan.total_turns or (
            plan.exchanges[exchange.exchange_number - 1] != exchange
        ):
            raise ValueError("exchange must be the matching Python-owned TurnPlan exchange")
        if language_profile_config.profile is not template.language_profile:
            raise ValueError("language profile config does not match template")
        expected_previous = exchange.exchange_number - 1
        previous_numbers = tuple(item.exchange_number for item in previous_approved_exchanges)
        if previous_numbers != tuple(range(1, expected_previous + 1)):
            raise ValueError("previous exchanges must contain only all earlier approved exchanges")

        state_after = self._copy_state(state_before)
        self._state_machine.apply(exchange, state_after)
        return GenerationRequest(
            template_id=template.template_id,
            plan_version=plan.plan_version,
            exchange_number=exchange.exchange_number,
            phase=exchange.phase,
            phase_variant=exchange.phase_variant,
            language_profile=template.language_profile,
            mlp_taxonomy=mlp_taxonomy,
            pool_data=pool_data,
            required_customer_facts=exchange.required_customer_facts,
            required_assistant_facts=exchange.required_assistant_facts,
            allowed_customer_facts=exchange.allowed_customer_facts,
            allowed_assistant_facts=exchange.allowed_assistant_facts,
            deviation=exchange.deviation,
            planned_state_changes=exchange.state_changes,
            booking_state_before=BookingStateSnapshot.from_booking_state(state_before),
            booking_state_after_expected=BookingStateSnapshot.from_booking_state(state_after),
            previous_approved_exchanges=tuple(previous_approved_exchanges),
            remaining_turns_after_current=plan.total_turns - exchange.exchange_number,
            applicable_rules=tuple(applicable_rules),
        )

    @staticmethod
    def _copy_state(state: BookingState) -> BookingState:
        return BookingState(
            original_values=state.original_values,
            current_values=thaw_value(state.current_values),
            available_services=state.available_services,
            expressed_required_fields=set(state.expressed_required_fields),
            missing_required_fields=set(state.missing_required_fields),
            used_deviations=list(state.used_deviations),
        )
