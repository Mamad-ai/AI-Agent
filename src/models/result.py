"""Final per-template processing result."""

from dataclasses import dataclass
from typing import Any, Optional, Tuple

from .dialogue import Dialogue
from .enums import Classification
from .state import BookingState
from .validation import ValidationReport


@dataclass(frozen=True)
class ProcessingResult:
    template_id: str
    classification: Classification
    booking_state: BookingState
    validation_reports: Tuple[ValidationReport, ...]
    dialogue: Optional[Dialogue] = None
    generation_attempts: int = 0
    audited: bool = False

    def __post_init__(self) -> None:
        if self.generation_attempts < 0:
            raise ValueError("generation_attempts cannot be negative")
        object.__setattr__(self, "validation_reports", tuple(self.validation_reports))

    def to_dict(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "classification": self.classification.value,
            "booking_state": self.booking_state.to_dict(),
            "validation_reports": [item.to_dict() for item in self.validation_reports],
            "dialogue": self.dialogue.to_dict() if self.dialogue else None,
            "generation_attempts": self.generation_attempts,
            "audited": self.audited,
        }
