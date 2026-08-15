"""Deterministic construction of machine-readable generation context."""

from dataclasses import dataclass
from typing import Any, Mapping

from src.models.template import freeze_value, thaw_value

from .language_profiles import LanguageProfileConfig
from .schemas import GenerationRequest


@dataclass(frozen=True)
class PromptContext:
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", freeze_value(self.payload))

    def to_dict(self) -> dict[str, Any]:
        return thaw_value(self.payload)


class PromptContextBuilder:
    def build(
        self, request: GenerationRequest, profile: LanguageProfileConfig
    ) -> PromptContext:
        if request.language_profile is not profile.profile:
            raise ValueError("language profile config does not match generation request")
        request_data = request.to_dict()
        return PromptContext(
            {
                "task": "generate_exactly_one_customer_to_assistant_exchange",
                "constraints": {
                    "required_facts_must_be_expressed": True,
                    "allowed_facts_may_be_expressed": True,
                    "facts_not_in_allowed_are_forbidden": True,
                    "do_not_create_state_changes": True,
                    "do_not_change_plan_structure": True,
                    "language": "svenska",
                    "mlp_labeling_instruction": (
                        "Generate the customer utterance first according to the locked plan and facts. "
                        "Then label that utterance using exactly one MLP1 from mlp_taxonomy and its "
                        "corresponding MLP2 when the taxonomy requires one; otherwise use null."
                    ),
                },
                "style": profile.to_dict(),
                "generation": request_data,
            }
        )
