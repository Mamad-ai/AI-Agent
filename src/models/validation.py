"""Structured deterministic and semantic validation results."""

from dataclasses import dataclass
from typing import Any, Optional, Tuple

from .enums import ValidationScope, ValidationSeverity, ValidationSource
from .template import freeze_value, thaw_value


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    severity: ValidationSeverity
    source: ValidationSource
    scope: ValidationScope
    exchange_number: Optional[int] = None
    field_name: Optional[str] = None
    expected: Any = None
    observed: Any = None
    repairable: bool = False

    def __post_init__(self) -> None:
        if not self.code or not self.message:
            raise ValueError("validation issue code and message must be non-empty")
        if self.exchange_number is not None and self.exchange_number < 1:
            raise ValueError("exchange_number must be at least 1")
        object.__setattr__(self, "expected", freeze_value(self.expected))
        object.__setattr__(self, "observed", freeze_value(self.observed))

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity.value,
            "source": self.source.value,
            "scope": self.scope.value,
            "exchange_number": self.exchange_number,
            "field_name": self.field_name,
            "expected": thaw_value(self.expected),
            "observed": thaw_value(self.observed),
            "repairable": self.repairable,
        }


@dataclass(frozen=True)
class ValidationReport:
    validator_name: str
    passed: bool
    issues: Tuple[ValidationIssue, ...] = ()

    def __post_init__(self) -> None:
        issues = tuple(self.issues)
        if not self.validator_name:
            raise ValueError("validator_name must be non-empty")
        if self.passed and issues:
            raise ValueError("a passed validation report cannot contain issues")
        if not self.passed and not issues:
            raise ValueError("a failed validation report must contain issues")
        object.__setattr__(self, "issues", issues)

    @property
    def has_hard_failures(self) -> bool:
        return any(issue.severity is ValidationSeverity.HARD for issue in self.issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "validator_name": self.validator_name,
            "passed": self.passed,
            "issues": [issue.to_dict() for issue in self.issues],
        }
