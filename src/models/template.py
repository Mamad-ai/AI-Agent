"""Immutable normalized input template models.

The completed template is the source of truth. These models preserve its values;
they do not select from pools or fill in missing values.
"""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Optional, Tuple

from .enums import (
    DeviationType,
    LanguageProfile,
    Phase,
    PhaseVariant,
    StateChangeSource,
    StateOperation,
)


def freeze_value(value: Any) -> Any:
    """Recursively convert JSON-like containers to immutable containers."""

    if isinstance(value, Mapping):
        return MappingProxyType({str(key): freeze_value(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze_value(item) for item in value)
    if isinstance(value, set):
        return frozenset(freeze_value(item) for item in value)
    return value


def thaw_value(value: Any) -> Any:
    """Convert immutable JSON-like containers to JSON-serializable values."""

    if isinstance(value, Mapping):
        return {str(key): thaw_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, frozenset, set)):
        return [thaw_value(item) for item in value]
    return value


@dataclass(frozen=True)
class MLPExpectation:
    """Optional template-owned semantic-action hint, never generator label facit."""

    exchange_number: int
    mlp1: str
    mlp2: Optional[str]

    def __post_init__(self) -> None:
        if self.exchange_number < 1:
            raise ValueError("exchange_number must be at least 1")
        if not self.mlp1 or self.mlp2 == "":
            raise ValueError("mlp1 must be non-empty and mlp2 must be non-empty or None")

    def to_dict(self) -> dict[str, Any]:
        return {
            "exchange_number": self.exchange_number,
            "mlp1": self.mlp1,
            "mlp2": self.mlp2,
        }


@dataclass(frozen=True)
class TemplateSemanticAction:
    """Optional domain intent supplied by the template, independent of MLP labels."""

    exchange_number: int
    domain: str
    action: Optional[str] = None

    def __post_init__(self) -> None:
        if self.exchange_number < 1:
            raise ValueError("exchange_number must be at least 1")
        if not self.domain or self.action == "":
            raise ValueError("domain is required; action must be non-empty or None")

    def to_dict(self) -> dict[str, Any]:
        return {
            "exchange_number": self.exchange_number,
            "domain": self.domain,
            "action": self.action,
        }


@dataclass(frozen=True)
class DeviationAssignment:
    """A deviation explicitly assigned to an exchange by the template."""

    exchange_number: int
    phase: Phase
    deviation_type: DeviationType

    def __post_init__(self) -> None:
        if self.exchange_number < 1:
            raise ValueError("exchange_number must be at least 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "exchange_number": self.exchange_number,
            "phase": self.phase.value,
            "deviation_type": self.deviation_type.value,
        }


@dataclass(frozen=True)
class TemplateStateChange:
    """A state change explicitly supplied by the completed template."""

    field_name: str
    operation: StateOperation
    value: Any
    source: StateChangeSource

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", freeze_value(self.value))

    def to_dict(self) -> dict[str, Any]:
        return {
            "field_name": self.field_name,
            "operation": self.operation.value,
            "value": thaw_value(self.value),
            "source": self.source.value,
        }


@dataclass(frozen=True)
class TemplateExchangeRequirements:
    """Source-of-truth content assigned to one complete exchange."""

    exchange_number: int
    required_customer_facts: Mapping[str, Any] = field(default_factory=dict)
    required_assistant_facts: Mapping[str, Any] = field(default_factory=dict)
    allowed_customer_facts: Mapping[str, Any] = field(default_factory=dict)
    allowed_assistant_facts: Mapping[str, Any] = field(default_factory=dict)
    state_changes: Tuple[TemplateStateChange, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "required_customer_facts", freeze_value(self.required_customer_facts))
        object.__setattr__(self, "required_assistant_facts", freeze_value(self.required_assistant_facts))
        object.__setattr__(
            self,
            "allowed_customer_facts",
            freeze_value(self.allowed_customer_facts or self.required_customer_facts),
        )
        object.__setattr__(
            self,
            "allowed_assistant_facts",
            freeze_value(self.allowed_assistant_facts or self.required_assistant_facts),
        )
        object.__setattr__(self, "state_changes", tuple(self.state_changes))

    def to_dict(self) -> dict[str, Any]:
        return {
            "exchange_number": self.exchange_number,
            "required_customer_facts": thaw_value(self.required_customer_facts),
            "required_assistant_facts": thaw_value(self.required_assistant_facts),
            "allowed_customer_facts": thaw_value(self.allowed_customer_facts),
            "allowed_assistant_facts": thaw_value(self.allowed_assistant_facts),
            "state_changes": [change.to_dict() for change in self.state_changes],
        }


@dataclass(frozen=True)
class ConversationTemplate:
    """Normalized and practically immutable completed template.

    ``total_turns`` always means complete customer-to-assistant exchanges.
    Thus 14 turns require 14 customer and 14 assistant utterances.
    """

    template_id: str
    total_turns: int
    language_profile: LanguageProfile
    original_values: Mapping[str, Any]
    phase_by_exchange: Tuple[Phase, ...]
    phase_variant_by_exchange: Tuple[PhaseVariant, ...]
    available_services: Tuple[str, ...] = ()
    required_fields: Tuple[str, ...] = ()
    deviations: Tuple[DeviationAssignment, ...] = ()
    mlp_expectations: Tuple[MLPExpectation, ...] = ()
    planned_semantic_actions: Tuple[TemplateSemanticAction, ...] = ()
    exchange_requirements: Tuple[TemplateExchangeRequirements, ...] = ()
    schema_version: str = "1"
    source_hash: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.template_id:
            raise ValueError("template_id must be non-empty")
        object.__setattr__(self, "original_values", freeze_value(self.original_values))
        object.__setattr__(self, "metadata", freeze_value(self.metadata))
        object.__setattr__(self, "phase_by_exchange", tuple(self.phase_by_exchange))
        object.__setattr__(
            self, "phase_variant_by_exchange", tuple(self.phase_variant_by_exchange)
        )
        object.__setattr__(self, "available_services", tuple(self.available_services))
        object.__setattr__(self, "required_fields", tuple(self.required_fields))
        object.__setattr__(self, "deviations", tuple(self.deviations))
        object.__setattr__(self, "mlp_expectations", tuple(self.mlp_expectations))
        object.__setattr__(self, "planned_semantic_actions", tuple(self.planned_semantic_actions))
        object.__setattr__(self, "exchange_requirements", tuple(self.exchange_requirements))

    def to_dict(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "total_turns": self.total_turns,
            "language_profile": self.language_profile.value,
            "original_values": thaw_value(self.original_values),
            "phase_by_exchange": [phase.value for phase in self.phase_by_exchange],
            "phase_variant_by_exchange": [
                variant.value for variant in self.phase_variant_by_exchange
            ],
            "available_services": list(self.available_services),
            "required_fields": list(self.required_fields),
            "deviations": [item.to_dict() for item in self.deviations],
            "mlp_expectations": [item.to_dict() for item in self.mlp_expectations],
            "planned_semantic_actions": [
                item.to_dict() for item in self.planned_semantic_actions
            ],
            "exchange_requirements": [item.to_dict() for item in self.exchange_requirements],
            "schema_version": self.schema_version,
            "source_hash": self.source_hash,
            "metadata": thaw_value(self.metadata),
        }
