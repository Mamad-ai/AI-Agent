"""Scripted fake implementing the ConversationGenerator protocol."""

from collections import deque
from typing import Callable, Iterable, Optional, Union

from src.ai.schemas import GenerationRequest, GenerationResponse


FakeOutcome = Union[GenerationResponse, Exception]


class FakeConversationGenerator:
    def __init__(
        self,
        outcomes: Iterable[FakeOutcome] = (),
        response_factory: Optional[Callable[[GenerationRequest], GenerationResponse]] = None,
    ) -> None:
        self._outcomes = deque(outcomes)
        self._response_factory = response_factory
        self.received_requests: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.received_requests.append(request)
        if self._outcomes:
            outcome = self._outcomes.popleft()
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        if self._response_factory is not None:
            return self._response_factory(request)
        raise RuntimeError("fake generator has no configured response")

    @property
    def unused_outcome_count(self) -> int:
        return len(self._outcomes)
