"""End-to-end application composition using provider-neutral workers."""

from typing import Any, Optional

from src.ai.interfaces import (
    ConversationGenerator,
    SemanticDialogueValidator,
    SemanticTurnValidator,
)
from src.ai.language_profiles import LanguageProfileLoader
from src.agent.classifier import ConversationClassifier
from src.agent.generation_service import (
    ConversationGenerationService,
    GenerationServiceError,
)
from src.agent.semantic_validation_service import (
    SemanticValidationError,
    SemanticValidationService,
)
from src.agent.turn_plan_builder import InvalidTemplateError, TurnPlanBuilder
from src.models.agent_run import AgentRunError, AgentRunResult
from src.models.classification import ClassificationConfig
from src.models.enums import Classification
from src.models.input_bundle import AgentInputBundle
from src.validators.deterministic_pipeline import (
    DeterministicValidationPipeline,
    DeterministicValidationResult,
)
from src.validators.template_validator import TemplateValidator


class ApplicationService:
    """Compose one completed template run; no retries, storage or provider knowledge."""

    def __init__(
        self,
        generator: ConversationGenerator,
        semantic_turn_validator: SemanticTurnValidator,
        semantic_dialogue_validator: SemanticDialogueValidator,
        classification_config: Optional[ClassificationConfig] = None,
        profile_loader: Optional[LanguageProfileLoader] = None,
    ) -> None:
        self._generator = generator
        self._semantic_turn = semantic_turn_validator
        self._semantic_dialogue = semantic_dialogue_validator
        self._classification_config = classification_config
        self._profile_loader = profile_loader or LanguageProfileLoader()
        self._template_validator = TemplateValidator()
        self._plan_builder = TurnPlanBuilder(self._template_validator)
        self._classifier = ConversationClassifier()

    def run(self, bundle: AgentInputBundle) -> AgentRunResult:
        template = bundle.template
        metadata = self._metadata(bundle)
        template_report = self._template_validator.validate(template)
        if not template_report.passed:
            deterministic = DeterministicValidationResult((template_report,))
            return self._error(
                bundle,
                "INVALID_TEMPLATE",
                "template_validation",
                "ConversationTemplate failed deterministic validation.",
                metadata,
                deterministic=deterministic,
            )

        try:
            plan = self._plan_builder.build(template)
        except InvalidTemplateError as error:
            return self._error(
                bundle,
                "TURN_PLAN_BUILD_FAILURE",
                "turn_plan",
                str(error),
                metadata,
            )

        try:
            profile = self._profile_loader.load(template.language_profile)
            classification_config = self._classification_config or ClassificationConfig.load()
        except (OSError, KeyError, TypeError, ValueError) as error:
            return self._error(
                bundle,
                "CONFIGURATION_FAILURE",
                "configuration",
                str(error),
                metadata,
            )

        taxonomy_mapping = bundle.mlp_taxonomy.as_validator_mapping()
        pipeline = DeterministicValidationPipeline(taxonomy_mapping)
        generation_service = ConversationGenerationService(
            self._generator, pipeline, bundle.pool, bundle.mlp_taxonomy
        )
        try:
            generation = generation_service.generate(template, plan, profile)
        except GenerationServiceError as error:
            return self._error(
                bundle,
                type(error).__name__,
                "generation",
                str(error),
                metadata,
            )

        if not generation.deterministic_validation.passed:
            return self._error(
                bundle,
                "DETERMINISTIC_VALIDATION_FAILURE",
                "deterministic_validation",
                "Generated dialogue failed deterministic validation.",
                metadata,
                dialogue=generation.dialogue,
                final_state=generation.final_booking_state,
                deterministic=generation.deterministic_validation,
            )

        semantic_service = SemanticValidationService(
            self._semantic_turn,
            self._semantic_dialogue,
            pipeline,
            bundle.mlp_taxonomy,
        )
        try:
            semantic = semantic_service.validate(
                generation.dialogue,
                plan,
                template,
                generation.final_booking_state,
                generation.deterministic_validation,
                profile,
            )
        except SemanticValidationError as error:
            return self._error(
                bundle,
                type(error).__name__,
                "semantic_validation",
                str(error),
                metadata,
                dialogue=generation.dialogue,
                final_state=generation.final_booking_state,
                deterministic=generation.deterministic_validation,
            )

        try:
            classification = self._classifier.classify(
                generation.deterministic_validation,
                semantic,
                classification_config,
            )
        except (TypeError, ValueError) as error:
            return self._error(
                bundle,
                "CLASSIFICATION_FAILURE",
                "classification",
                str(error),
                metadata,
                dialogue=generation.dialogue,
                final_state=generation.final_booking_state,
                deterministic=generation.deterministic_validation,
                semantic=semantic,
            )

        return AgentRunResult(
            template_id=template.template_id,
            classification=classification.classification,
            status=classification.classification,
            dialogue=generation.dialogue,
            final_booking_state=generation.final_booking_state,
            deterministic_validation=generation.deterministic_validation,
            semantic_validation=semantic,
            classification_result=classification,
            metadata=metadata,
        )

    @staticmethod
    def _metadata(bundle: AgentInputBundle) -> dict[str, Any]:
        return {
            "template_schema_version": bundle.template.schema_version,
            "template_source_hash": bundle.template.source_hash,
            "pool_version": bundle.pool.version,
            "mlp_taxonomy_version": bundle.mlp_taxonomy.version,
            "input_versions": dict(bundle.input_versions),
        }

    @staticmethod
    def _error(
        bundle: AgentInputBundle,
        code: str,
        stage: str,
        message: str,
        metadata: dict[str, Any],
        dialogue: Any = None,
        final_state: Any = None,
        deterministic: Any = None,
        semantic: Any = None,
    ) -> AgentRunResult:
        return AgentRunResult(
            template_id=bundle.template.template_id,
            classification=Classification.ERROR,
            status=Classification.ERROR,
            dialogue=dialogue,
            final_booking_state=final_state,
            deterministic_validation=deterministic,
            semantic_validation=semantic,
            errors=(AgentRunError(code, stage, message),),
            metadata=metadata,
        )
