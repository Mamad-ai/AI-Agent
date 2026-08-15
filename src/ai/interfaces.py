"""Provider-neutral AI worker interfaces."""

from typing import Protocol

from .schemas import (
    GenerationRequest,
    GenerationResponse,
    SemanticDialogueValidationRequest,
    SemanticDialogueValidationResponse,
    SemanticTurnValidationRequest,
    SemanticTurnValidationResponse,
)


class ConversationGenerator(Protocol):
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate exactly one structured customer-to-assistant exchange."""


class SemanticTurnValidator(Protocol):
    def validate_turn(
        self, request: SemanticTurnValidationRequest
    ) -> SemanticTurnValidationResponse:
        """Semantically assess one generated exchange."""


class SemanticDialogueValidator(Protocol):
    def validate_dialogue(
        self, request: SemanticDialogueValidationRequest
    ) -> SemanticDialogueValidationResponse:
        """Semantically assess a complete dialogue."""
