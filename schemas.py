from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal
from dataclasses import dataclass, field, asdict


ModificationType = Literal["action_change", "modality_mask", "temporal_shift"]


@dataclass(kw_only=True)
class PredictedOutcome:
    z_score: Optional[float] = None
    raw_value: Optional[float] = None
    raw_name: Optional[str] = None
    confidence: Optional[float] = None
    rationale: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PredictedOutcome":
        raw_value = None
        raw_name = None

        if "raw_value" in data:
            raw_value = data.get("raw_value")
            raw_name = data.get("raw_name")
        elif "raw_WBC" in data:
            raw_value = data.get("raw_WBC")
            raw_name = "WBC"
        elif "raw_mmHg" in data:
            raw_value = data.get("raw_mmHg")
            raw_name = "mmHg"

        return cls(
            z_score=data.get("z_score"),
            raw_value=raw_value,
            raw_name=raw_name,
            confidence=data.get("confidence"),
            rationale=data.get("rationale"),
        )

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)

        if self.raw_value is not None and self.raw_name:
            if self.raw_name == "WBC":
                result["raw_WBC"] = self.raw_value
            elif self.raw_name == "mmHg":
                result["raw_mmHg"] = self.raw_value

        return result


@dataclass(kw_only=True)
class CounterfactualScenario:
    patient_id: str
    intervention_point: float
    X: List[List[float]]
    A_original: List[int]
    A_counterfactual: List[int]
    modification_type: ModificationType
    modification_details: Dict[str, Any] = field(default_factory=dict)
    predicted_outcome: PredictedOutcome
    question: str
    rationale: Optional[str] = None
    ground_truth: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CounterfactualScenario":
        outcome_data = data.get("predicted_outcome", {}) or {}

        return cls(
            patient_id=str(data["patient_id"]),
            intervention_point=float(data["intervention_point"]),
            X=data["X"],
            A_original=data["A_original"],
            A_counterfactual=data["A_counterfactual"],
            modification_type=data["modification_type"],
            modification_details=data.get("modification_details", {}),
            predicted_outcome=PredictedOutcome.from_dict(outcome_data),
            question=str(data["question"]),
            rationale=data.get("rationale"),
            ground_truth=data.get("ground_truth"),
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["predicted_outcome"] = self.predicted_outcome.to_dict()
        return result


@dataclass(kw_only=True)
class EvidenceItem:
    source: str
    chunk_id: str
    text: str
    distance: Optional[float] = None
    source_type: str = "web"
    supports: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(kw_only=True)
class GapItem:
    gap_type: str
    severity: str
    description: str
    suggested_next_step: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(kw_only=True)
class MemoryCandidate:
    should_store: bool
    memory_key: str
    memory_type: str
    summary: str
    tags: List[str] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(kw_only=True)
class MITSExport:
    patient_id: str
    intervention_point: float
    question: str
    action_delta: Dict[str, Any]
    verified_outcome: Dict[str, Any]
    evidence_summary: List[str]
    verifier_verdict: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(kw_only=True)
class VerifierOutput:
    verdict: str
    rationale: str
    references: List[EvidenceItem]
    checks: Dict[str, Any]
    gaps: List[GapItem]
    memory_candidate: Optional[MemoryCandidate] = None
    mits_export: Optional[MITSExport] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "rationale": self.rationale,
            "references": [ref.to_dict() for ref in self.references],
            "checks": self.checks,
            "gaps": [gap.to_dict() for gap in self.gaps],
            "memory_candidate": self.memory_candidate.to_dict() if self.memory_candidate else None,
            "mits_export": self.mits_export.to_dict() if self.mits_export else None,
        }