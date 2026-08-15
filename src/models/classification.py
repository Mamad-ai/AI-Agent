"""Immutable configuration and result models for deterministic classification."""

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Tuple

from .enums import Classification


class ClassificationReasonCode(str, Enum):
    DETERMINISTIC_HARD_FAILURE = "DETERMINISTIC_HARD_FAILURE"
    DETERMINISTIC_VALIDATION_FAILED = "DETERMINISTIC_VALIDATION_FAILED"
    SEMANTIC_HARD_FAILURE = "SEMANTIC_HARD_FAILURE"
    SEMANTIC_BLOCKING_FAILURE = "SEMANTIC_BLOCKING_FAILURE"
    CONFIDENCE_BELOW_WEAK_THRESHOLD = "CONFIDENCE_BELOW_WEAK_THRESHOLD"
    CONFIDENCE_BELOW_SAFE_THRESHOLD = "CONFIDENCE_BELOW_SAFE_THRESHOLD"
    QUALITY_ISSUES_PRESENT = "QUALITY_ISSUES_PRESENT"
    UNCERTAIN_ISSUES_PRESENT = "UNCERTAIN_ISSUES_PRESENT"
    LOW_NATURALNESS = "LOW_NATURALNESS"
    CLEAN_HIGH_CONFIDENCE = "CLEAN_HIGH_CONFIDENCE"


@dataclass(frozen=True)
class ClassificationReason:
    code: ClassificationReasonCode
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code.value, "message": self.message}


@dataclass(frozen=True)
class ClassificationConfig:
    version: str
    safe_min_confidence: float
    weak_min_confidence: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.weak_min_confidence <= self.safe_min_confidence <= 1.0:
            raise ValueError(
                "confidence thresholds must satisfy 0 <= weak <= safe <= 1"
            )
        if not self.version:
            raise ValueError("classification config version must be non-empty")

    @classmethod
    def load(cls, path: Path | None = None) -> "ClassificationConfig":
        config_path = path or Path(__file__).resolve().parents[2] / "config" / "classification.json"
        with config_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls(
            version=data["version"],
            safe_min_confidence=float(data["safe_min_confidence"]),
            weak_min_confidence=float(data["weak_min_confidence"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "safe_min_confidence": self.safe_min_confidence,
            "weak_min_confidence": self.weak_min_confidence,
        }


@dataclass(frozen=True)
class ClassificationResult:
    classification: Classification
    reasons: Tuple[ClassificationReason, ...]
    deterministic_pass: bool
    semantic_pass: bool
    deterministic_hard_failures: int
    semantic_hard_failures: int
    quality_issue_count: int
    uncertain_issue_count: int
    minimum_semantic_confidence: float
    requires_audit: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "reasons", tuple(self.reasons))

    def to_dict(self) -> dict[str, Any]:
        return {
            "classification": self.classification.value,
            "reasons": [reason.to_dict() for reason in self.reasons],
            "deterministic_pass": self.deterministic_pass,
            "semantic_pass": self.semantic_pass,
            "deterministic_hard_failures": self.deterministic_hard_failures,
            "semantic_hard_failures": self.semantic_hard_failures,
            "quality_issue_count": self.quality_issue_count,
            "uncertain_issue_count": self.uncertain_issue_count,
            "minimum_semantic_confidence": self.minimum_semantic_confidence,
            "requires_audit": self.requires_audit,
        }
