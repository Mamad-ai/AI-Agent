"""Deterministic conversion from a completed template to a turn plan."""

from src.models.plan import PlannedExchange, PlannedSemanticAction, PlannedStateChange, TurnPlan
from src.models.template import ConversationTemplate
from src.validators.template_validator import TemplateValidator


class InvalidTemplateError(ValueError):
    """Raised when a turn plan is requested for an invalid template."""


class TurnPlanBuilder:
    def __init__(self, validator: TemplateValidator | None = None) -> None:
        self._validator = validator or TemplateValidator()

    def build(self, template: ConversationTemplate) -> TurnPlan:
        report = self._validator.validate(template)
        if not report.passed:
            codes = ", ".join(issue.code for issue in report.issues)
            raise InvalidTemplateError(f"cannot build plan from invalid template: {codes}")

        deviations = {item.exchange_number: item.deviation_type for item in template.deviations}
        semantic_actions = {
            item.exchange_number: item for item in template.planned_semantic_actions
        }
        requirements = {
            item.exchange_number: item for item in template.exchange_requirements
        }
        exchanges: list[PlannedExchange] = []
        for number in range(1, template.total_turns + 1):
            semantic_action = semantic_actions.get(number)
            requirement = requirements[number]
            exchanges.append(
                PlannedExchange(
                    exchange_number=number,
                    phase=template.phase_by_exchange[number - 1],
                    phase_variant=template.phase_variant_by_exchange[number - 1],
                    semantic_action=(
                        PlannedSemanticAction(semantic_action.domain, semantic_action.action)
                        if semantic_action is not None
                        else None
                    ),
                    required_customer_facts=requirement.required_customer_facts,
                    required_assistant_facts=requirement.required_assistant_facts,
                    allowed_customer_facts=requirement.allowed_customer_facts,
                    allowed_assistant_facts=requirement.allowed_assistant_facts,
                    deviation=deviations.get(number),
                    state_changes=tuple(
                        PlannedStateChange(
                            field_name=change.field_name,
                            operation=change.operation,
                            value=change.value,
                            source=change.source,
                        )
                        for change in requirement.state_changes
                    ),
                )
            )
        return TurnPlan(template.template_id, template.total_turns, tuple(exchanges))
