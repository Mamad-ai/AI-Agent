"""Immutable end-to-end application result."""

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple

from src.agent.semantic_validation_service import SemanticValidationResult
from src.models.classification import ClassificationResult
from src.models.dialogue import Dialogue
from src.models.enums import Classification
from src.models.state import BookingState
from src.models.template import freeze_value, thaw_value
from src.validators.deterministic_pipeline import DeterministicValidationResult


@dataclass(frozen=True)
class AgentRunError:
    code: str
    stage: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "stage": self.stage, "message": self.message}


@dataclass(frozen=True)
class AgentRunResult:
    template_id: str
    classification: Classification
    status: Classification
    dialogue: Optional[Dialogue] = None
    final_booking_state: Optional[BookingState] = None
    deterministic_validation: Optional[DeterministicValidationResult] = None
    semantic_validation: Optional[SemanticValidationResult] = None
    classification_result: Optional[ClassificationResult] = None
    errors: Tuple[AgentRunError, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status is not self.classification:
            raise ValueError("status must equal classification")
        object.__setattr__(self, "errors", tuple(self.errors))
        object.__setattr__(self, "metadata", freeze_value(self.metadata))

    @property
    def output_label(self) -> str:
        return {
            Classification.SAFE: "safe",
            Classification.WEAK: "tvek",
            Classification.ERROR: "fel",
        }[self.classification]

    def to_dict(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "classification": self.classification.value,
            "status": self.status.value,
            "output_label": self.output_label,
            "dialogue": self.dialogue.to_dict() if self.dialogue else None,
            "final_booking_state": (
                self.final_booking_state.to_dict() if self.final_booking_state else None
            ),
            "deterministic_validation": (
                self.deterministic_validation.to_dict()
                if self.deterministic_validation
                else None
            ),
            "semantic_validation": (
                self.semantic_validation.to_dict() if self.semantic_validation else None
            ),
            "classification_result": (
                self.classification_result.to_dict() if self.classification_result else None
            ),
            "errors": [error.to_dict() for error in self.errors],
            "metadata": thaw_value(self.metadata),
        }
