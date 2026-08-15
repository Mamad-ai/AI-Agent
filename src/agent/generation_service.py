"""Small sequential generation service using a provider-neutral worker."""

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

from src.ai.interfaces import ConversationGenerator
from src.ai.language_profiles import LanguageProfileConfig
from src.ai.schemas import GenerationResponse
from src.agent.generation_request_builder import GenerationRequestBuilder
from src.agent.state_machine import StateMachine, StateTransitionError
from src.models.dialogue import ConversationExchange, Dialogue
from src.models.input_bundle import MLPTaxonomy, PoolData
from src.models.plan import TurnPlan
from src.models.state import BookingState
from src.models.template import ConversationTemplate, thaw_value
from src.validators.deterministic_pipeline import (
    DeterministicValidationPipeline,
    DeterministicValidationResult,
)


class GenerationServiceError(RuntimeError):
    """Base error for the bounded generation service."""


class GeneratorExecutionError(GenerationServiceError):
    """The generator worker raised an exception."""


class InvalidGenerationResponseError(GenerationServiceError):
    """A worker response violated the structured response contract."""

    def __init__(
        self,
        message: str,
        validation_result: DeterministicValidationResult | None = None,
    ) -> None:
        super().__init__(message)
        self.validation_result = validation_result


class GenerationStateTransitionError(GenerationServiceError):
    """A Python-owned planned state transition failed."""


@dataclass(frozen=True)
class ConversationGenerationResult:
    dialogue: Dialogue
    final_booking_state: BookingState
    deterministic_validation: DeterministicValidationResult

    @property
    def passed(self) -> bool:
        return self.deterministic_validation.passed


class ConversationGenerationService:
    """Generate sequentially; approved means locally deterministic-accepted, not SAFE."""

    def __init__(
        self,
        generator: ConversationGenerator,
        validation_pipeline: DeterministicValidationPipeline,
        pool_data: PoolData,
        mlp_taxonomy: MLPTaxonomy,
        request_builder: GenerationRequestBuilder | None = None,
        state_machine: StateMachine | None = None,
    ) -> None:
        self._generator = generator
        self._pipeline = validation_pipeline
        self._pool_data = pool_data
        self._mlp_taxonomy = mlp_taxonomy
        self._state_machine = state_machine or StateMachine()
        self._request_builder = request_builder or GenerationRequestBuilder(self._state_machine)

    def generate(
        self,
        template: ConversationTemplate,
        plan: TurnPlan,
        language_profile_config: LanguageProfileConfig,
        rules_by_exchange: Optional[Mapping[int, Sequence[str]]] = None,
    ) -> ConversationGenerationResult:
        rules_by_exchange = rules_by_exchange or {}
        state = BookingState(
            original_values=template.original_values,
            current_values=thaw_value(template.original_values),
            available_services=template.available_services,
            missing_required_fields=set(template.required_fields),
        )
        approved: list[ConversationExchange] = []

        for planned in plan.exchanges:
            try:
                request = self._request_builder.build(
                    template,
                    plan,
                    planned,
                    state,
                    approved,
                    language_profile_config,
                    self._pool_data,
                    self._mlp_taxonomy,
                    rules_by_exchange.get(planned.exchange_number, ()),
                )
            except StateTransitionError as error:
                raise GenerationStateTransitionError(
                    f"failed to calculate expected state for exchange {planned.exchange_number}"
                ) from error

            try:
                response = self._generator.generate(request)
            except Exception as error:
                raise GeneratorExecutionError(
                    f"generator failed for exchange {planned.exchange_number}"
                ) from error

            if not isinstance(response, GenerationResponse):
                raise InvalidGenerationResponseError(
                    f"generator returned an invalid response type for exchange {planned.exchange_number}"
                )
            try:
                generated = response.to_exchange(planned)
            except (TypeError, ValueError) as error:
                raise InvalidGenerationResponseError(
                    f"invalid response for exchange {planned.exchange_number}"
                ) from error

            local_validation = self._pipeline.validate_exchange(
                generated,
                planned,
                request.booking_state_after_expected.current_values,
            )
            if not local_validation.passed:
                raise InvalidGenerationResponseError(
                    f"exchange {planned.exchange_number} failed local deterministic validation",
                    local_validation,
                )

            try:
                self._state_machine.apply(planned, state)
            except StateTransitionError as error:
                raise GenerationStateTransitionError(
                    f"failed to apply state for exchange {planned.exchange_number}"
                ) from error
            expressed = self._pipeline.correctly_expressed_required_fields(
                generated, planned, template.required_fields
            )
            state.expressed_required_fields.update(expressed)
            state.missing_required_fields.difference_update(expressed)
            approved.append(generated)

        dialogue = Dialogue(
            template_id=template.template_id,
            language_profile=template.language_profile,
            exchanges=tuple(approved),
            plan_version=plan.plan_version,
        )
        validation = self._pipeline.validate(dialogue, plan, template, state)
        return ConversationGenerationResult(dialogue, state, validation)
