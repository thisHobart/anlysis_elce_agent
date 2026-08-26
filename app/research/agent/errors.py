"""Explicit failures for the model-led research workflow."""

from app.research.data.loader import ResearchDataError


class ResearchAgentError(RuntimeError):
    """Base error shown to the desktop research conversation."""


class ResearchModelUnavailableError(ResearchAgentError):
    """Raised when the required research model is not configured or reachable."""


class ResearchPlanValidationError(ResearchAgentError):
    """Raised when model output cannot be compiled into an executable plan."""


class DuplicateResearchFunctionError(ResearchPlanValidationError):
    """Raised when a plan repeats a function that must batch its varying dimension."""

    def __init__(self, function_name: str, recommendation: str) -> None:
        self.function_name = function_name
        self.recommendation = recommendation
        super().__init__(f"研究函数 {function_name} 每个计划最多调用一次。{recommendation}")


class PlanCompatibilityError(ResearchPlanValidationError):
    """Installed Skill, function, or plan version cannot execute the approved plan."""


class DataFingerprintMismatchError(ResearchDataError):
    """Input content no longer matches the fingerprint approved in the plan."""


class InsufficientDataError(ResearchDataError):
    """The current dataset cannot satisfy the minimum deterministic evidence gate."""


class RepairablePlanError(ResearchPlanValidationError):
    """The model may repair function arguments or selected variables without new data."""

