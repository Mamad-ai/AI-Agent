"""Machine-readable booking and processing state models."""

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple

from .dialogue import ConversationExchange
from .enums import DeviationType, Phase, ProcessingStatus
from .plan import TurnPlan
from .template import freeze_value, thaw_value
from .validation import ValidationReport


SELECTED_SERVICES_FIELD = "selected_services"


@dataclass
class BookingState:
    """Original template values and independently mutable current values.

    Booking facts, including selected services, have exactly one current source
    of truth: ``current_values``. ``available_services`` is validation metadata
    only and must never be used to choose a service.
    """

    original_values: Mapping[str, Any]
    current_values: dict[str, Any]
    available_services: Tuple[str, ...] = ()
    expressed_required_fields: set[str] = field(default_factory=set)
    missing_required_fields: set[str] = field(default_factory=set)
    used_deviations: list[DeviationType] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.original_values = freeze_value(self.original_values)
        self.current_values = dict(thaw_value(self.current_values))
        self.available_services = tuple(self.available_services)
        self.expressed_required_fields = set(self.expressed_required_fields)
        self.missing_required_fields = set(self.missing_required_fields)
        self.used_deviations = list(self.used_deviations)

    @property
    def selected_services(self) -> Tuple[str, ...]:
        """Read-only view derived from canonical ``current_values``."""

        services = self.current_values.get(SELECTED_SERVICES_FIELD, ())
        if not isinstance(services, (list, tuple)):
            raise TypeError("current selected_services must be a list or tuple")
        return tuple(services)

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_values": thaw_value(self.original_values),
            "current_values": thaw_value(self.current_values),
            "available_services": list(self.available_services),
            "expressed_required_fields": sorted(self.expressed_required_fields),
            "missing_required_fields": sorted(self.missing_required_fields),
            "used_deviations": [item.value for item in self.used_deviations],
        }


@dataclass
class ProcessingState:
    batch_id: str
    template_id: str
    total_turns: int
    booking: BookingState
    source_hash: Optional[str] = None
    config_version: Optional[str] = None
    prompt_version: Optional[str] = None
    model_version: Optional[str] = None
    status: ProcessingStatus = ProcessingStatus.PENDING
    current_phase: Optional[Phase] = None
    completed_turns: int = 0
    plan: Optional[TurnPlan] = None
    generated_exchanges: list[ConversationExchange] = field(default_factory=list)
    validation_history: list[ValidationReport] = field(default_factory=list)
    regeneration_attempts: dict[str, int] = field(default_factory=dict)
    checkpoint_version: int = 1

    def __post_init__(self) -> None:
        if self.total_turns < 1:
            raise ValueError("total_turns must be at least 1")
        if not 0 <= self.completed_turns <= self.total_turns:
            raise ValueError("completed_turns must be between zero and total_turns")

    @property
    def remaining_turns(self) -> int:
        return self.total_turns - self.completed_turns

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "template_id": self.template_id,
            "source_hash": self.source_hash,
            "config_version": self.config_version,
            "prompt_version": self.prompt_version,
            "model_version": self.model_version,
            "status": self.status.value,
            "current_phase": self.current_phase.value if self.current_phase else None,
            "total_turns": self.total_turns,
            "completed_turns": self.completed_turns,
            "remaining_turns": self.remaining_turns,
            "booking": self.booking.to_dict(),
            "plan": self.plan.to_dict() if self.plan else None,
            "generated_exchanges": [item.to_dict() for item in self.generated_exchanges],
            "validation_history": [item.to_dict() for item in self.validation_history],
            "regeneration_attempts": dict(self.regeneration_attempts),
            "checkpoint_version": self.checkpoint_version,
        }
