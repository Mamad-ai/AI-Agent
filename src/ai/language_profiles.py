"""Loading and representation of style-only language profile configuration."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Tuple

from src.models.enums import LanguageProfile
from src.models.template import freeze_value, thaw_value


@dataclass(frozen=True)
class LanguageProfileConfig:
    profile: LanguageProfile
    formality: str
    typical_sentence_length: str
    address_style: str
    colloquiality: str
    grammatical_features: Tuple[str, ...]
    avoid_expressions: Tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "grammatical_features", tuple(self.grammatical_features))
        object.__setattr__(self, "avoid_expressions", tuple(self.avoid_expressions))

    def style_context(self) -> Mapping[str, Any]:
        return freeze_value(
            {
                "profile": self.profile.value,
                "formality": self.formality,
                "typical_sentence_length": self.typical_sentence_length,
                "address_style": self.address_style,
                "colloquiality": self.colloquiality,
                "grammatical_features": list(self.grammatical_features),
                "avoid_expressions": list(self.avoid_expressions),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return thaw_value(self.style_context())


class LanguageProfileLoader:
    def __init__(self, config_directory: Path | None = None) -> None:
        self._directory = config_directory or (
            Path(__file__).resolve().parents[2] / "config" / "language_profiles"
        )

    def load(self, profile: LanguageProfile) -> LanguageProfileConfig:
        path = self._directory / f"{profile.value}.json"
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if data.get("profile") != profile.value:
            raise ValueError("language profile file does not match requested profile")
        return LanguageProfileConfig(
            profile=profile,
            formality=data["formality"],
            typical_sentence_length=data["typical_sentence_length"],
            address_style=data["address_style"],
            colloquiality=data["colloquiality"],
            grammatical_features=tuple(data["grammatical_features"]),
            avoid_expressions=tuple(data["avoid_expressions"]),
        )
