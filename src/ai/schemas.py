"""Immutable structured requests and responses for future AI workers."""

from dataclasses import dataclass, field
from typing import Any, FrozenSet, Mapping, Optional, Tuple

from src.models.dialogue import ConversationExchange, Dialogue, Utterance
from src.models.enums import (
    DeviationType,
    LanguageProfile,
    Phase,
    PhaseVariant,
    Speaker,
    ValidationSource,
)
from src.models.plan import PlannedExchange, PlannedStateChange, TurnPlan
from src.models.input_bundle import MLPTaxonomy, PoolData
from src.models.state import BookingState
from src.models.template import freeze_value, thaw_value
from src.models.validation import ValidationIssue

from .language_profiles import LanguageProfileConfig


@dataclass(frozen=True)
class BookingStateSnapshot:
    """Immutable state projection safe to include in an AI request."""

    original_values: Mapping[str, Any]
    current_values: Mapping[str, Any]
    used_deviations: Tuple[DeviationType, ...] = ()
    expressed_required_fields: FrozenSet[str] = frozenset()
    missing_required_fields: FrozenSet[str] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "original_values", freeze_value(self.original_values))
        object.__setattr__(self, "current_values", freeze_value(self.current_values))
        object.__setattr__(self, "used_deviations", tuple(self.used_deviations))
        object.__setattr__(
            self, "expressed_required_fields", frozenset(self.expressed_required_fields)
        )
        object.__setattr__(self, "missing_required_fields", frozenset(self.missing_required_fields))

    @classmethod
    def from_booking_state(cls, state: BookingState) -> "BookingStateSnapshot":
        return cls(
            state.original_values,
            state.current_values,
            tuple(state.used_deviations),
            frozenset(state.expressed_required_fields),
            frozenset(state.missing_required_fields),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_values": thaw_value(self.original_values),
            "current_values": thaw_value(self.current_values),
            "used_deviations": [item.value for item in self.used_deviations],
            "expressed_required_fields": sorted(self.expressed_required_fields),
            "missing_required_fields": sorted(self.missing_required_fields),
        }


@dataclass(frozen=True)
class GenerationRequest:
    """Locked input for one exchange.

    ``previous_approved_exchanges`` contains only earlier exchanges that passed
    the local deterministic fact and MLP checks. It does not mean SAFE or
    semantically approved. ``booking_state_after_expected`` applies only the
    Python-owned planned booking transition. Its required-field progress stays
    equal to ``booking_state_before`` until generated facts have actually passed
    local validation.
    """

    template_id: str
    plan_version: str
    exchange_number: int
    phase: Phase
    phase_variant: PhaseVariant
    language_profile: LanguageProfile
    mlp_taxonomy: MLPTaxonomy
    pool_data: PoolData
    required_customer_facts: Mapping[str, Any]
    required_assistant_facts: Mapping[str, Any]
    allowed_customer_facts: Mapping[str, Any]
    allowed_assistant_facts: Mapping[str, Any]
    deviation: Optional[DeviationType]
    planned_state_changes: Tuple[PlannedStateChange, ...]
    booking_state_before: BookingStateSnapshot
    booking_state_after_expected: BookingStateSnapshot
    previous_approved_exchanges: Tuple[ConversationExchange, ...]
    remaining_turns_after_current: int
    applicable_rules: Tuple[str, ...]

    def __post_init__(self) -> None:
        if self.exchange_number < 1:
            raise ValueError("exchange_number must be at least 1")
        if self.remaining_turns_after_current < 0:
            raise ValueError("remaining_turns_after_current cannot be negative")
        if not self.phase_variant.belongs_to(self.phase):
            raise ValueError("phase_variant does not belong to phase")
        object.__setattr__(self, "required_customer_facts", freeze_value(self.required_customer_facts))
        object.__setattr__(self, "required_assistant_facts", freeze_value(self.required_assistant_facts))
        object.__setattr__(self, "allowed_customer_facts", freeze_value(self.allowed_customer_facts))
        object.__setattr__(self, "allowed_assistant_facts", freeze_value(self.allowed_assistant_facts))
        object.__setattr__(self, "planned_state_changes", tuple(self.planned_state_changes))
        object.__setattr__(
            self, "previous_approved_exchanges", tuple(self.previous_approved_exchanges)
        )
        object.__setattr__(self, "applicable_rules", tuple(self.applicable_rules))

    def to_dict(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "plan_version": self.plan_version,
            "exchange_number": self.exchange_number,
            "phase": self.phase.value,
            "phase_variant": self.phase_variant.value,
            "language_profile": self.language_profile.value,
            "mlp_taxonomy": self.mlp_taxonomy.to_dict(),
            "pool_data": self.pool_data.to_dict(),
            "required_customer_facts": thaw_value(self.required_customer_facts),
            "required_assistant_facts": thaw_value(self.required_assistant_facts),
            "allowed_customer_facts": thaw_value(self.allowed_customer_facts),
            "allowed_assistant_facts": thaw_value(self.allowed_assistant_facts),
            "deviation": self.deviation.value if self.deviation else None,
            "planned_state_changes": [item.to_dict() for item in self.planned_state_changes],
            "booking_state_before": self.booking_state_before.to_dict(),
            "booking_state_after_expected": self.booking_state_after_expected.to_dict(),
            "previous_approved_exchanges": [
                item.to_dict() for item in self.previous_approved_exchanges
            ],
            "remaining_turns_after_current": self.remaining_turns_after_current,
            "applicable_rules": list(self.applicable_rules),
        }


@dataclass(frozen=True)
class GenerationResponse:
    exchange_number: int
    customer_text: str
    assistant_text: str
    customer_mlp1: str
    customer_mlp2: Optional[str]
    customer_facts: Mapping[str, Any] = field(default_factory=dict)
    assistant_facts: Mapping[str, Any] = field(default_factory=dict)
    summary_facts: Mapping[str, Any] = field(default_factory=dict)
    model_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.exchange_number < 1:
            raise ValueError("exchange_number must be at least 1")
        if not self.customer_text.strip() or not self.assistant_text.strip():
            raise ValueError("customer_text and assistant_text must be non-empty")
        object.__setattr__(self, "customer_facts", freeze_value(self.customer_facts))
        object.__setattr__(self, "assistant_facts", freeze_value(self.assistant_facts))
        object.__setattr__(self, "summary_facts", freeze_value(self.summary_facts))
        object.__setattr__(self, "model_metadata", freeze_value(self.model_metadata))

    def to_exchange(self, planned: PlannedExchange) -> ConversationExchange:
        """Bind model-authored text/facts to Python-owned plan structure."""

        if self.exchange_number != planned.exchange_number:
            raise ValueError("generation response exchange_number does not match the plan")
        if planned.phase is not Phase.P4 and self.summary_facts:
            raise ValueError("summary_facts are only allowed for P4 exchanges")
        return ConversationExchange(
            exchange_number=planned.exchange_number,
            customer=Utterance(Speaker.CUSTOMER, self.customer_text),
            assistant=Utterance(Speaker.ASSISTANT, self.assistant_text),
            phase=planned.phase,
            phase_variant=planned.phase_variant,
            customer_mlp1=self.customer_mlp1,
            customer_mlp2=self.customer_mlp2,
            customer_facts=self.customer_facts,
            assistant_facts=self.assistant_facts,
            summary_facts=self.summary_facts,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "exchange_number": self.exchange_number,
            "customer_text": self.customer_text,
            "assistant_text": self.assistant_text,
            "customer_mlp1": self.customer_mlp1,
            "customer_mlp2": self.customer_mlp2,
            "customer_facts": thaw_value(self.customer_facts),
            "assistant_facts": thaw_value(self.assistant_facts),
            "summary_facts": thaw_value(self.summary_facts),
            "model_metadata": thaw_value(self.model_metadata),
        }


def _validate_semantic_response(confidence: float, issues: Tuple[ValidationIssue, ...]) -> None:
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if any(issue.source is not ValidationSource.SEMANTIC_AI for issue in issues):
        raise ValueError("semantic response issues must use ValidationSource.SEMANTIC_AI")


@dataclass(frozen=True)
class SemanticTurnValidationRequest:
    planned_exchange: PlannedExchange
    generated_exchange: ConversationExchange
    language_profile: LanguageProfile
    language_profile_config: LanguageProfileConfig
    mlp_taxonomy: MLPTaxonomy
    state_before: BookingStateSnapshot
    state_after: BookingStateSnapshot
    nearby_exchanges: Tuple[ConversationExchange, ...] = ()

    def __post_init__(self) -> None:
        if self.language_profile_config.profile is not self.language_profile:
            raise ValueError("language profile config does not match language_profile")
        object.__setattr__(self, "nearby_exchanges", tuple(self.nearby_exchanges))

    def to_dict(self) -> dict[str, Any]:
        return {
            "planned_exchange": self.planned_exchange.to_dict(),
            "generated_exchange": self.generated_exchange.to_dict(),
            "language_profile": self.language_profile.value,
            "language_profile_config": self.language_profile_config.to_dict(),
            "mlp_taxonomy": self.mlp_taxonomy.to_dict(),
            "state_before": self.state_before.to_dict(),
            "state_after": self.state_after.to_dict(),
            "nearby_exchanges": [item.to_dict() for item in self.nearby_exchanges],
        }


@dataclass(frozen=True)
class SemanticTurnValidationResponse:
    language_profile_match: bool
    mlp1_semantic_match: bool
    mlp2_semantic_match: bool
    facts_semantically_preserved: bool
    naturalness: bool
    contradiction_found: bool
    unnecessary_question_found: bool
    overall_pass: bool
    confidence: float
    issues: Tuple[ValidationIssue, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "issues", tuple(self.issues))
        _validate_semantic_response(self.confidence, self.issues)
        incompatible_with_pass = (
            not self.language_profile_match
            or not self.mlp1_semantic_match
            or not self.mlp2_semantic_match
            or not self.facts_semantically_preserved
            or self.contradiction_found
            or self.unnecessary_question_found
        )
        if self.overall_pass and incompatible_with_pass:
            raise ValueError("overall_pass contradicts semantic turn assessments")

    def to_dict(self) -> dict[str, Any]:
        return {
            "language_profile_match": self.language_profile_match,
            "mlp1_semantic_match": self.mlp1_semantic_match,
            "mlp2_semantic_match": self.mlp2_semantic_match,
            "facts_semantically_preserved": self.facts_semantically_preserved,
            "naturalness": self.naturalness,
            "contradiction_found": self.contradiction_found,
            "unnecessary_question_found": self.unnecessary_question_found,
            "overall_pass": self.overall_pass,
            "confidence": self.confidence,
            "issues": [item.to_dict() for item in self.issues],
        }


@dataclass(frozen=True)
class SemanticDialogueValidationRequest:
    dialogue: Dialogue
    turn_plan: TurnPlan
    final_booking_state: BookingStateSnapshot
    language_profile: LanguageProfile
    language_profile_config: LanguageProfileConfig
    mlp_taxonomy: MLPTaxonomy

    def __post_init__(self) -> None:
        if self.language_profile_config.profile is not self.language_profile:
            raise ValueError("language profile config does not match language_profile")

    def to_dict(self) -> dict[str, Any]:
        return {
            "dialogue": self.dialogue.to_dict(),
            "turn_plan": self.turn_plan.to_dict(),
            "final_booking_state": self.final_booking_state.to_dict(),
            "language_profile": self.language_profile.value,
            "language_profile_config": self.language_profile_config.to_dict(),
            "mlp_taxonomy": self.mlp_taxonomy.to_dict(),
        }


@dataclass(frozen=True)
class SemanticDialogueValidationResponse:
    global_language_profile_match: bool
    naturalness: bool
    coherence: bool
    contradiction_found: bool
    state_facts_consistent: bool
    pd_intention_followed: bool
    overall_pass: bool
    confidence: float
    issues: Tuple[ValidationIssue, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "issues", tuple(self.issues))
        _validate_semantic_response(self.confidence, self.issues)
        incompatible_with_pass = (
            not self.global_language_profile_match
            or not self.coherence
            or self.contradiction_found
            or not self.state_facts_consistent
            or not self.pd_intention_followed
        )
        if self.overall_pass and incompatible_with_pass:
            raise ValueError("overall_pass contradicts semantic dialogue assessments")

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_language_profile_match": self.global_language_profile_match,
            "naturalness": self.naturalness,
            "coherence": self.coherence,
            "contradiction_found": self.contradiction_found,
            "state_facts_consistent": self.state_facts_consistent,
            "pd_intention_followed": self.pd_intention_followed,
            "overall_pass": self.overall_pass,
            "confidence": self.confidence,
            "issues": [item.to_dict() for item in self.issues],
        }
