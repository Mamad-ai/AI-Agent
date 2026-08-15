"""Immutable application input models."""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Tuple

from .template import ConversationTemplate, freeze_value, thaw_value


@dataclass(frozen=True)
class PoolData:
    """Provider-neutral validation/source metadata; never a selection engine."""

    services: Tuple[str, ...] = ()
    days: Tuple[str, ...] = ()
    time_expressions: Tuple[str, ...] = ()
    stylists: Tuple[str, ...] = ()
    booking_actions: Tuple[str, ...] = ()
    authentication_values: Mapping[str, Any] = field(default_factory=dict)
    additional_values: Mapping[str, Any] = field(default_factory=dict)
    version: str = "1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "services", tuple(self.services))
        object.__setattr__(self, "days", tuple(self.days))
        object.__setattr__(self, "time_expressions", tuple(self.time_expressions))
        object.__setattr__(self, "stylists", tuple(self.stylists))
        object.__setattr__(self, "booking_actions", tuple(self.booking_actions))
        object.__setattr__(self, "authentication_values", freeze_value(self.authentication_values))
        object.__setattr__(self, "additional_values", freeze_value(self.additional_values))

    def to_dict(self) -> dict[str, Any]:
        return {
            "services": list(self.services),
            "days": list(self.days),
            "time_expressions": list(self.time_expressions),
            "stylists": list(self.stylists),
            "booking_actions": list(self.booking_actions),
            "authentication_values": thaw_value(self.authentication_values),
            "additional_values": thaw_value(self.additional_values),
            "version": self.version,
        }


@dataclass(frozen=True)
class MLPTaxonomy:
    """Single source of truth for allowed MLP1 -> MLP2 relations."""

    relations: Mapping[str, Tuple[str, ...]]
    version: str = "1"

    def __post_init__(self) -> None:
        normalized = {
            str(mlp1): frozenset(str(mlp2) for mlp2 in mlp2_values)
            for mlp1, mlp2_values in self.relations.items()
        }
        object.__setattr__(self, "relations", MappingProxyType(normalized))

    def as_validator_mapping(self) -> dict[str, frozenset[str]]:
        return {mlp1: frozenset(values) for mlp1, values in self.relations.items()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "relations": {
                mlp1: sorted(mlp2_values)
                for mlp1, mlp2_values in sorted(self.relations.items())
            },
            "version": self.version,
        }


@dataclass(frozen=True)
class AgentInputBundle:
    template: ConversationTemplate
    pool: PoolData
    mlp_taxonomy: MLPTaxonomy
    input_versions: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_versions", freeze_value(self.input_versions))

    def to_dict(self) -> dict[str, Any]:
        return {
            "template": self.template.to_dict(),
            "pool": self.pool.to_dict(),
            "mlp_taxonomy": self.mlp_taxonomy.to_dict(),
            "input_versions": thaw_value(self.input_versions),
        }
