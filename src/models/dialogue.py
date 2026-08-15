"""Generated dialogue models with explicit utterances and exchanges."""

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple

from .enums import LanguageProfile, Phase, PhaseVariant, Speaker
from .template import freeze_value, thaw_value


@dataclass(frozen=True)
class Utterance:
    speaker: Speaker
    text: str

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("utterance text must be non-empty")

    def to_dict(self) -> dict[str, str]:
        return {"speaker": self.speaker.value, "text": self.text}


@dataclass(frozen=True)
class ConversationExchange:
    """One complete turn consisting of Customer -> AI."""

    exchange_number: int
    customer: Utterance
    assistant: Utterance
    phase: Optional[Phase] = None
    phase_variant: Optional[PhaseVariant] = None
    customer_mlp1: Optional[str] = None
    customer_mlp2: Optional[str] = None
    customer_facts: Mapping[str, Any] = field(default_factory=dict)
    assistant_facts: Mapping[str, Any] = field(default_factory=dict)
    summary_facts: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.exchange_number < 1:
            raise ValueError("exchange_number must be at least 1")
        if self.customer.speaker is not Speaker.CUSTOMER:
            raise ValueError("customer utterance must have customer speaker")
        if self.assistant.speaker is not Speaker.ASSISTANT:
            raise ValueError("assistant utterance must have assistant speaker")
        object.__setattr__(self, "customer_facts", freeze_value(self.customer_facts))
        object.__setattr__(self, "assistant_facts", freeze_value(self.assistant_facts))
        object.__setattr__(self, "summary_facts", freeze_value(self.summary_facts))

    def to_dict(self) -> dict[str, Any]:
        return {
            "exchange_number": self.exchange_number,
            "customer": self.customer.to_dict(),
            "assistant": self.assistant.to_dict(),
            "phase": self.phase.value if self.phase else None,
            "phase_variant": self.phase_variant.value if self.phase_variant else None,
            "customer_mlp1": self.customer_mlp1,
            "customer_mlp2": self.customer_mlp2,
            "customer_facts": thaw_value(self.customer_facts),
            "assistant_facts": thaw_value(self.assistant_facts),
            "summary_facts": thaw_value(self.summary_facts),
        }


@dataclass(frozen=True)
class Dialogue:
    template_id: str
    language_profile: LanguageProfile
    exchanges: Tuple[ConversationExchange, ...]
    plan_version: str = "1"
    generation_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        exchanges = tuple(self.exchanges)
        if tuple(item.exchange_number for item in exchanges) != tuple(range(1, len(exchanges) + 1)):
            raise ValueError("exchange numbers must be consecutive and start at 1")
        object.__setattr__(self, "exchanges", exchanges)
        object.__setattr__(self, "generation_metadata", freeze_value(self.generation_metadata))

    @property
    def total_turns(self) -> int:
        """Number of complete Customer -> AI exchanges."""

        return len(self.exchanges)

    @property
    def utterance_count(self) -> int:
        return self.total_turns * 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "language_profile": self.language_profile.value,
            "plan_version": self.plan_version,
            "total_turns": self.total_turns,
            "utterance_count": self.utterance_count,
            "exchanges": [exchange.to_dict() for exchange in self.exchanges],
            "generation_metadata": thaw_value(self.generation_metadata),
        }
