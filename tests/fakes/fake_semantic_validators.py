"""Scripted fake semantic validators with no language understanding."""

from collections import deque
from typing import Callable, Iterable, Optional, Union

from src.ai.schemas import (
    SemanticDialogueValidationRequest,
    SemanticDialogueValidationResponse,
    SemanticTurnValidationRequest,
    SemanticTurnValidationResponse,
)


TurnOutcome = Union[SemanticTurnValidationResponse, Exception]
DialogueOutcome = Union[SemanticDialogueValidationResponse, Exception]


class FakeSemanticTurnValidator:
    def __init__(
        self,
        outcomes: Iterable[TurnOutcome] = (),
        response_factory: Optional[
            Callable[[SemanticTurnValidationRequest], SemanticTurnValidationResponse]
        ] = None,
    ) -> None:
        self._outcomes = deque(outcomes)
        self._factory = response_factory
        self.received_requests: list[SemanticTurnValidationRequest] = []

    def validate_turn(
        self, request: SemanticTurnValidationRequest
    ) -> SemanticTurnValidationResponse:
        self.received_requests.append(request)
        if self._outcomes:
            outcome = self._outcomes.popleft()
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        if self._factory is not None:
            return self._factory(request)
        raise RuntimeError("fake semantic turn validator has no configured response")


class FakeSemanticDialogueValidator:
    def __init__(
        self,
        outcome: DialogueOutcome,
    ) -> None:
        self._outcome = outcome
        self.received_requests: list[SemanticDialogueValidationRequest] = []

    def validate_dialogue(
        self, request: SemanticDialogueValidationRequest
    ) -> SemanticDialogueValidationResponse:
        self.received_requests.append(request)
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome
