"""Explicit failures for the model-led research workflow."""


class ResearchAgentError(RuntimeError):
    """Base error shown to the desktop research conversation."""


class ResearchModelUnavailableError(ResearchAgentError):
    """Raised when the required research model is not configured or reachable."""


class ResearchPlanValidationError(ResearchAgentError):
    """Raised when model output cannot be compiled into an executable plan."""

