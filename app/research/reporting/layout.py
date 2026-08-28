"""Canonical on-disk layout for one desktop research package."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArtifactLayout:
    """Resolve every package path from one run directory."""

    root: Path

    @property
    def report_markdown(self) -> Path:
        return self.root / "report.md"

    @property
    def methods_markdown(self) -> Path:
        return self.root / "methods.md"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def figures(self) -> Path:
        return self.root / "figures"

    @property
    def evidence(self) -> Path:
        return self.root / "evidence"

    @property
    def provenance(self) -> Path:
        return self.root / "provenance"

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def data_quality(self) -> Path:
        return self.evidence / "data_quality.json"

    @property
    def eda_summary(self) -> Path:
        return self.evidence / "eda_summary.json"

    @property
    def agent_evaluation(self) -> Path:
        return self.evidence / "agent_evaluation.json"

    @property
    def research_plan(self) -> Path:
        return self.provenance / "research_plan.json"

    @property
    def execution_trace(self) -> Path:
        return self.provenance / "execution_trace.json"

    @property
    def conversation(self) -> Path:
        return self.provenance / "conversation.json"

    @property
    def research_loop(self) -> Path:
        return self.provenance / "research_loop.json"

    @property
    def study_context(self) -> Path:
        return self.provenance / "study_context.json"

    @property
    def aligned_data(self) -> Path:
        return self.data / "aligned_data.parquet"

    def create_directories(self) -> None:
        for path in (self.evidence, self.provenance, self.data):
            path.mkdir(parents=True, exist_ok=False)
