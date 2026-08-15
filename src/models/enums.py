"""Shared enumerations for the conversation domain."""

from enum import Enum


class StringEnum(str, Enum):
    """Python 3.10 compatible string enum."""


class LanguageProfile(StringEnum):
    VARDAGLIG_SVENSKA = "vardaglig_svenska"
    ORTEN_SVENSKA = "orten_svenska"
    BRYTANDE_SVENSKA = "brytande_svenska"
    FORMELL_SVENSKA = "formell_svenska"


class Phase(StringEnum):
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"
    P4 = "P4"
    P5 = "P5"


class PhaseVariant(StringEnum):
    P1_1 = "P1.1"
    P1_2 = "P1.2"
    P1_3 = "P1.3"
    P1_4 = "P1.4"
    P2_1 = "P2.1"
    P2_2 = "P2.2"
    P2_3 = "P2.3"
    P2_4 = "P2.4"
    P3_1 = "P3.1"
    P3_2 = "P3.2"
    P3_3 = "P3.3"
    P4_1 = "P4.1"
    P4_2 = "P4.2"
    P4_3 = "P4.3"
    P5_1 = "P5.1"
    P5_2 = "P5.2"

    @property
    def phase(self) -> Phase:
        """Return the only main phase this variant can belong to."""

        return Phase(self.value.split(".", maxsplit=1)[0])

    def belongs_to(self, phase: Phase) -> bool:
        return self.phase is phase


class DeviationType(StringEnum):
    D1 = "D1"
    D2 = "D2"
    D3 = "D3"
    D5 = "D5"

    @property
    def changes_booking_state(self) -> bool:
        """D1 is informational; D2, D3 and D5 change booking state."""

        return self is not DeviationType.D1


class Speaker(StringEnum):
    CUSTOMER = "customer"
    ASSISTANT = "assistant"


class StateOperation(StringEnum):
    REPLACE = "replace"
    ADD = "add"


class StateChangeSource(StringEnum):
    PHASE_VARIANT = "phase_variant"
    DEVIATION = "deviation"


class ProcessingStatus(StringEnum):
    PENDING = "pending"
    TEMPLATE_VALIDATED = "template_validated"
    PLANNED = "planned"
    GENERATED = "generated"
    VALIDATING = "validating"
    REPAIRING = "repairing"
    FINAL_VALIDATION = "final_validation"
    COMPLETED = "completed"
    ERROR = "error"


class ValidationSeverity(StringEnum):
    HARD = "hard"
    QUALITY = "quality"
    UNCERTAIN = "uncertain"


class ValidationSource(StringEnum):
    DETERMINISTIC = "deterministic"
    SEMANTIC_AI = "semantic_ai"


class ValidationScope(StringEnum):
    TEMPLATE = "template"
    PLAN = "plan"
    EXCHANGE = "exchange"
    UTTERANCE = "utterance"
    STATE = "state"
    DIALOGUE = "dialogue"


class Classification(StringEnum):
    SAFE = "SAFE"
    WEAK = "WEAK"
    ERROR = "ERROR"
