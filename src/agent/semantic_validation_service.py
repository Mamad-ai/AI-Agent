"""Semantic validation service using provider-neutral validator workers."""

from dataclasses import dataclass
from typing import Any, Tuple

from src.ai.interfaces import SemanticDialogueValidator, SemanticTurnValidator
from src.ai.language_profiles import LanguageProfileConfig
from src.ai.schemas import (
    SemanticDialogueValidationResponse,
    SemanticTurnValidationResponse,
)
from src.agent.semantic_request_builder import (
    SemanticDialogueRequestBuilder,
    SemanticTurnRequestBuilder,
)
from src.agent.state_machine import StateMachine, StateTransitionError
from src.models.dialogue import Dialogue
from src.models.enums import ValidationSeverity
from src.models.input_bundle import MLPTaxonomy
from src.models.plan import TurnPlan
from src.models.state import BookingState
from src.models.template import ConversationTemplate, thaw_value
from src.models.validation import ValidationIssue
from src.validators.deterministic_pipeline import (
    DeterministicValidationResult,
    DeterministicValidationPipeline,
)


class SemanticValidationError(RuntimeError):
    """Base error for semantic validation execution."""


class DeterministicValidationRequiredError(SemanticValidationError):
    """Semantic validation was attempted despite deterministic HARD failures."""


class SemanticTurnValidatorError(SemanticValidationError):
    """A semantic turn worker failed."""


class SemanticDialogueValidatorError(SemanticValidationError):
    """A semantic dialogue worker failed."""


class InvalidSemanticResponseError(SemanticValidationError):
    """A semantic worker returned an object outside its response contract."""


@dataclass(frozen=True)
class SemanticValidationResult:
    turn_responses: Tuple[SemanticTurnValidationResponse, ...]
    dialogue_response: SemanticDialogueValidationResponse

    def __post_init__(self) -> None:
        object.__setattr__(self, "turn_responses", tuple(self.turn_responses))

    @property
    def issues(self) -> Tuple[ValidationIssue, ...]:
        return tuple(
            issue
            for response in (*self.turn_responses, self.dialogue_response)
            for issue in response.issues
        )

    @property
    def has_hard_failures(self) -> bool:
        return any(issue.severity is ValidationSeverity.HARD for issue in self.issues)

    @property
    def has_quality_issues(self) -> bool:
        return any(issue.severity is ValidationSeverity.QUALITY for issue in self.issues)

    @property
    def has_uncertain_issues(self) -> bool:
        return any(issue.severity is ValidationSeverity.UNCERTAIN for issue in self.issues)

    @property
    def minimum_confidence(self) -> float:
        return min(
            *(response.confidence for response in self.turn_responses),
            self.dialogue_response.confidence,
        )

    @property
    def overall_semantic_pass(self) -> bool:
        return (
            all(response.overall_pass for response in self.turn_responses)
            and self.dialogue_response.overall_pass
            and not self.has_hard_failures
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_responses": [item.to_dict() for item in self.turn_responses],
            "dialogue_response": self.dialogue_response.to_dict(),
            "issues": [item.to_dict() for item in self.issues],
            "has_hard_failures": self.has_hard_failures,
            "has_quality_issues": self.has_quality_issues,
            "has_uncertain_issues": self.has_uncertain_issues,
            "minimum_confidence": self.minimum_confidence,
            "overall_semantic_pass": self.overall_semantic_pass,
        }


class SemanticValidationService:
    """Run semantic workers only after the exact current inputs pass preflight.

    The gate is always a fresh deterministic validation of the current Dialogue,
    TurnPlan, ConversationTemplate and final BookingState. A previously supplied
    result cannot authorize semantic validation or override a current HARD failure.
    """

    def __init__(
        self,
        turn_validator: SemanticTurnValidator,
        dialogue_validator: SemanticDialogueValidator,
        deterministic_pipeline: DeterministicValidationPipeline,
        mlp_taxonomy: MLPTaxonomy,
        state_machine: StateMachine | None = None,
        turn_request_builder: SemanticTurnRequestBuilder | None = None,
        dialogue_request_builder: SemanticDialogueRequestBuilder | None = None,
    ) -> None:
        self._turn_validator = turn_validator
        self._dialogue_validator = dialogue_validator
        self._pipeline = deterministic_pipeline
        self._mlp_taxonomy = mlp_taxonomy
        self._state_machine = state_machine or StateMachine()
        self._turn_builder = turn_request_builder or SemanticTurnRequestBuilder()
        self._dialogue_builder = dialogue_request_builder or SemanticDialogueRequestBuilder()

    def validate(
        self,
        dialogue: Dialogue,
        plan: TurnPlan,
        template: ConversationTemplate,
        final_state: BookingState,
        deterministic_result: DeterministicValidationResult,
        profile: LanguageProfileConfig,
    ) -> SemanticValidationResult:
        current_preflight = self._pipeline.validate(dialogue, plan, template, final_state)
        if current_preflight.has_hard_failures:
            raise DeterministicValidationRequiredError(
                "the exact current inputs must pass deterministic validation without HARD failures"
            )
        if dialogue.total_turns != plan.total_turns:
            raise DeterministicValidationRequiredError(
                "semantic validation requires identical dialogue and plan turn counts"
            )

        # Kept in the API for compatibility/result aggregation only. It is never
        # used as authority for the semantic gate; current_preflight is authoritative.
        _ = deterministic_result

        replay_state = BookingState(
            original_values=template.original_values,
            current_values=thaw_value(template.original_values),
            available_services=template.available_services,
            missing_required_fields=set(template.required_fields),
        )
        turn_responses: list[SemanticTurnValidationResponse] = []
        previous = None
        for planned, generated in zip(plan.exchanges, dialogue.exchanges):
            state_before = self._copy_state(replay_state)
            try:
                self._state_machine.apply(planned, replay_state)
            except StateTransitionError as error:
                raise SemanticValidationError(
                    f"failed to replay state for exchange {planned.exchange_number}"
                ) from error
            expressed = self._pipeline.correctly_expressed_required_fields(
                generated, planned, template.required_fields
            )
            replay_state.expressed_required_fields.update(expressed)
            replay_state.missing_required_fields.difference_update(expressed)
            request = self._turn_builder.build(
                planned,
                generated,
                profile,
                self._mlp_taxonomy,
                state_before,
                replay_state,
                previous,
            )
            try:
                response = self._turn_validator.validate_turn(request)
            except Exception as error:
                raise SemanticTurnValidatorError(
                    f"semantic turn validator failed for exchange {planned.exchange_number}"
                ) from error
            if not isinstance(response, SemanticTurnValidationResponse):
                raise InvalidSemanticResponseError(
                    f"invalid semantic turn response for exchange {planned.exchange_number}"
                )
            turn_responses.append(response)
            previous = generated

        dialogue_request = self._dialogue_builder.build(
            dialogue, plan, final_state, profile, self._mlp_taxonomy
        )
        try:
            dialogue_response = self._dialogue_validator.validate_dialogue(dialogue_request)
        except Exception as error:
            raise SemanticDialogueValidatorError("semantic dialogue validator failed") from error
        if not isinstance(dialogue_response, SemanticDialogueValidationResponse):
            raise InvalidSemanticResponseError("invalid semantic dialogue response")
        return SemanticValidationResult(tuple(turn_responses), dialogue_response)

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
