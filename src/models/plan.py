"""Deterministic plan models for complete customer-to-assistant exchanges."""

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple

from .enums import (
    DeviationType,
    Phase,
    PhaseVariant,
    StateChangeSource,
    StateOperation,
)
from .template import freeze_value, thaw_value


@dataclass(frozen=True)
class PlannedStateChange:
    """A template-authorized state change expected in an exchange."""

    field_name: str
    operation: StateOperation
    value: Any
    source: StateChangeSource

    def __post_init__(self) -> None:
        if not self.field_name:
            raise ValueError("field_name must be non-empty")
        object.__setattr__(self, "value", freeze_value(self.value))

    def to_dict(self) -> dict[str, Any]:
        return {
            "field_name": self.field_name,
            "operation": self.operation.value,
            "value": thaw_value(self.value),
            "source": self.source.value,
        }


@dataclass(frozen=True)
class PlannedSemanticAction:
    """Python-owned intended action used by semantic validation, not generation labeling."""

    category: str
    action: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.category or self.action == "":
            raise ValueError("semantic action category is required; action is non-empty or None")

    def to_dict(self) -> dict[str, Optional[str]]:
        return {"category": self.category, "action": self.action}


@dataclass(frozen=True)
class PlannedExchange:
    """Plan for one complete turn: exactly one customer and one AI utterance."""

    exchange_number: int
    phase: Phase
    phase_variant: PhaseVariant
    required_customer_facts: Mapping[str, Any]
    required_assistant_facts: Mapping[str, Any]
    semantic_action: Optional[PlannedSemanticAction] = None
    allowed_customer_facts: Mapping[str, Any] = field(default_factory=dict)
    allowed_assistant_facts: Mapping[str, Any] = field(default_factory=dict)
    deviation: Optional[DeviationType] = None
    state_changes: Tuple[PlannedStateChange, ...] = ()

    def __post_init__(self) -> None:
        if self.exchange_number < 1:
            raise ValueError("exchange_number must be at least 1")
        if not self.phase_variant.belongs_to(self.phase):
            raise ValueError(
                f"phase variant {self.phase_variant.value} does not belong to {self.phase.value}"
            )
        if self.deviation is DeviationType.D1 and self.state_changes:
            raise ValueError("D1 must not contain booking state changes")
        for change in self.state_changes:
            if change.source is StateChangeSource.DEVIATION and self.deviation is None:
                raise ValueError("deviation-sourced state changes require an explicit deviation")
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
            "phase": self.phase.value,
            "phase_variant": self.phase_variant.value,
            "semantic_action": self.semantic_action.to_dict() if self.semantic_action else None,
            "required_customer_facts": thaw_value(self.required_customer_facts),
            "required_assistant_facts": thaw_value(self.required_assistant_facts),
            "allowed_customer_facts": thaw_value(self.allowed_customer_facts),
            "allowed_assistant_facts": thaw_value(self.allowed_assistant_facts),
            "deviation": self.deviation.value if self.deviation else None,
            "state_changes": [change.to_dict() for change in self.state_changes],
        }


@dataclass(frozen=True)
class TurnPlan:
    """Ordered plan whose size is measured in complete exchanges."""

    template_id: str
    total_turns: int
    exchanges: Tuple[PlannedExchange, ...]
    plan_version: str = "1"

    def __post_init__(self) -> None:
        exchanges = tuple(self.exchanges)
        if self.total_turns < 1:
            raise ValueError("total_turns must be at least 1")
        if len(exchanges) != self.total_turns:
            raise ValueError("exchanges must contain exactly total_turns complete turns")
        if tuple(item.exchange_number for item in exchanges) != tuple(range(1, self.total_turns + 1)):
            raise ValueError("exchange numbers must be consecutive and start at 1")
        object.__setattr__(self, "exchanges", exchanges)

    def to_dict(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "total_turns": self.total_turns,
            "plan_version": self.plan_version,
            "exchanges": [exchange.to_dict() for exchange in self.exchanges],
        }
