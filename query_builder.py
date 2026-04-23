"""
Turn structured scenario into two kinds of queries:
1) PMC corpus-building queries
2) Final retrieval query for Chroma
"""

from __future__ import annotations

from typing import Dict, List

from .config import VerifierConfig
from .schemas import CounterfactualScenario


class QueryBuilder:
    def __init__(self, config: VerifierConfig):
        self.config = config

    # --------- shared helpers ---------
    def _action_delta_text(self, scenario: CounterfactualScenario) -> str:
        """
        Summarize A_original -> A_counterfactual, e.g.
        "MRSA_targeted_antibiotics: off -> on; hospital_discharge: on -> off"
        """
        changes = []
        for idx, (a0, a1) in enumerate(zip(scenario.A_original, scenario.A_counterfactual)):
            if a0 != a1:
                action_name = self.config.action_names.get(idx, f"action_{idx}")
                before = "on" if int(a0) == 1 else "off"
                after = "on" if int(a1) == 1 else "off"
                changes.append(f"{action_name}: {before} -> {after}")
        if not changes:
            return "no action change"
        return "; ".join(changes)

    def _latest_state_summary(self, scenario: CounterfactualScenario) -> str:
        """
        Use metadata.feature_names when available; fall back to index-based summary.
        """
        if not scenario.X:
            return ""

        latest = scenario.X[-1]
        feature_names = scenario.metadata.get("feature_names", [])

        if feature_names and len(feature_names) == len(latest):
            parts = [f"{name}={value}" for name, value in zip(feature_names, latest)]
        else:
            parts = [str(v) for v in latest]
        return ", ".join(parts)

    def _infer_changed_targets(self, scenario: CounterfactualScenario) -> List[str]:
        targets = []
        for idx, (a0, a1) in enumerate(zip(scenario.A_original, scenario.A_counterfactual)):
            if a0 != a1:
                targets.append(self.config.action_names.get(idx, f"action_{idx}"))
        return targets

    def _outcome_text(self, scenario: CounterfactualScenario) -> str:
        po = scenario.predicted_outcome
        if getattr(po, "raw_value", None) is not None:
            raw_name = getattr(po, "raw_name", None) or "outcome"
            return f"Predicted {raw_name}: {po.raw_value}"
        if getattr(po, "z_score", None) is not None:
            return f"Predicted outcome z-score {po.z_score}"
        return "Predicted outcome unavailable"

    # --------- A. PMC queries ---------
    def build_pmc_queries(self, scenario: CounterfactualScenario) -> List[str]:
        """
        Queries focused on building a small, infection-centric PMC KB
        for the current demo scenario (MRSA / WBC / discharge).
        """
        question = scenario.question
        action_delta = self._action_delta_text(scenario)
        state_summary = self._latest_state_summary(scenario)
        outcome_text = self._outcome_text(scenario)

        changed_targets = scenario.modification_details.get("changed_targets", []) or self._infer_changed_targets(
            scenario
        )

        rationale = scenario.rationale or ""
        pred_rationale = scenario.predicted_outcome.rationale or ""

        base_queries: List[str] = [
            question,
            f"MRSA targeted antibiotics vs standard prophylaxis around hospital discharge; "
            f"infection control and WBC trends. {state_summary}",
            "Hospital discharge safety when white blood cell count is elevated or rising; "
            "risk of empyema, trunk abscess, and readmission.",
            "Guidelines on switching from cefazolin or other prophylactic antibiotics "
            "to vancomycin or MRSA-targeted therapy in postoperative patients.",
            "Postoperative infection, MRSA empyema, and complications related to premature discharge "
            "with persistent leukocytosis.",
        ]

        # Lightly specialize when we see these targets
        lower_targets = [t.lower() for t in changed_targets]
        if any("discharge" in t for t in lower_targets):
            base_queries.append(
                "Discharge criteria and timing for postoperative patients with suspected infection "
                "or high white blood cell count."
            )

        if any("mrsa" in t for t in lower_targets):
            base_queries.append(
                "MRSA bacteremia / MRSA pneumonia postoperative management, antibiotic choice, "
                "and duration of therapy."
            )

        # Add one scenario-specific synthesis query
        combo = (
            f"Counterfactual intervention: {action_delta}. "
            f"{outcome_text}. "
            f"Scenario rationale: {rationale} {pred_rationale}"
        ).strip()
        base_queries.append(combo)

        # simple dedupe
        seen = set()
        unique_queries: List[str] = []
        for q in base_queries:
            q_norm = q.strip()
            if not q_norm:
                continue
            if q_norm.lower() in seen:
                continue
            seen.add(q_norm.lower())
            unique_queries.append(q_norm)

        print("\n[QueryBuilder] PMC queries generated:")
        for i, q in enumerate(unique_queries, 1):
            print(f"  [{i}] {q}")

        return unique_queries

    # --------- B. Retrieval query ---------
    def build_retrieval_query(self, scenario: CounterfactualScenario) -> str:
        """
        Short, focused query used for final vector retrieval in Chroma.
        """
        action_delta = self._action_delta_text(scenario)
        state_summary = self._latest_state_summary(scenario)
        outcome_text = self._outcome_text(scenario)

        changed_targets = scenario.modification_details.get("changed_targets", []) or self._infer_changed_targets(
            scenario
        )
        changed_targets_text = ", ".join(changed_targets) if changed_targets else ""

        parts = [
            scenario.question,
            f"Intervention: {action_delta}",
            f"Latest state: {state_summary}" if state_summary else "",
            outcome_text,
            f"Targets: {changed_targets_text}" if changed_targets_text else "",
        ]

        query = " | ".join([p for p in parts if p])
        print("\n[QueryBuilder] Retrieval query:")
        print(f"  {query}")
        return query

    # --------- Structured delta (for exports) ---------
    def action_delta_struct(self, scenario: CounterfactualScenario) -> Dict[str, Dict[str, int]]:
        delta: Dict[str, Dict[str, int]] = {}
        for idx, (a0, a1) in enumerate(zip(scenario.A_original, scenario.A_counterfactual)):
            if a0 != a1:
                delta[self.config.action_names.get(idx, f"action_{idx}")] = {
                    "before": int(a0),
                    "after": int(a1),
                }
        return delta

    # --------- C. ACR queries ---------
    def build_acr_topic_query(self, scenario: CounterfactualScenario) -> str:
        """
        Build a simple ACR topic query from scenario.
        ACR queries are guideline-focused, unlike PMC's literature focus.
        """
        question = scenario.question.lower()
        changed_targets = self._infer_changed_targets(scenario)
        rationale = (scenario.rationale or "").lower()

        # Extract imaging-related keywords
        keywords = []
        if "imaging" in question or "ct" in question or "mri" in question:
            keywords.extend(["breast cancer", "lung cancer", "abdominal imaging"])  # Example expansions
        if any("discharge" in t.lower() for t in changed_targets):
            keywords.append("postoperative imaging")
        if rationale:
            if "infection" in rationale:
                keywords.append("infection imaging")

        # Fallback to general medical imaging if no specific keywords
        if not keywords:
            keywords = ["medical imaging guidelines"]

        query = " ".join(keywords)
        print(f"\n[QueryBuilder] ACR topic query: {query}")
        return query

    def build_acr_keywords(self, scenario: CounterfactualScenario) -> List[str]:
        """
        Extract keywords for ACR search from scenario.
        """
        keywords = []
        question_words = scenario.question.lower().split()
        keywords.extend([w for w in question_words if len(w) > 3])  # Filter short words

        changed_targets = self._infer_changed_targets(scenario)
        keywords.extend(changed_targets)

        if scenario.rationale:
            rationale_words = scenario.rationale.lower().split()
            keywords.extend([w for w in rationale_words if len(w) > 3])

        # Dedupe
        return list(set(keywords))
