from typing import List

from sentence_transformers import CrossEncoder

from .config import VerifierConfig
from .schemas import CounterfactualScenario, EvidenceItem, VerifierOutput, GapItem
from .kb import KnowledgeBase
from .judge import QwenJudge
from .query_builder import QueryBuilder


class RAGVerifier:
    def __init__(self, config: VerifierConfig):
        self.config = config

        print("[Verifier] Initializing components...")

        self.kb = KnowledgeBase(config)
        self.judge = QwenJudge(config)
        self.query_builder = QueryBuilder(config)

        try:
            self.reranker = CrossEncoder("BAAI/bge-reranker-base")
            self.use_reranker = True
            print("[Verifier] Reranker loaded")
        except Exception as e:
            print(f"[Verifier] Reranker failed, fallback to no-rerank: {e}")
            self.use_reranker = False

    def build_kb(self, urls: List[str] | None = None) -> int:
        """
        Legacy helper to build a web KB from URLs.
        New PMC flow is controlled from verify() via QueryBuilder + build_from_pmc_queries.
        """
        return self.kb.build_from_urls(urls)

    def _build_query_from_scenario(self, scenario: CounterfactualScenario) -> str:
        """
        Thin wrapper so existing code paths use the new QueryBuilder retrieval query.
        """
        return self.query_builder.build_retrieval_query(scenario)

    def _filter_evidence(
        self,
        evidence: List[EvidenceItem],
        scenario: CounterfactualScenario,
    ) -> List[EvidenceItem]:
        keywords = [
            "infection", "infectious", "sepsis", "empyema", "abscess",
            "mrsa", "antibiotic", "vancomycin", "cefazolin",
            "wbc", "white blood cell", "leukocytosis",
            "discharge", "readmission", "complication",
            "postoperative", "post-operative",
            # Add imaging keywords for ACR support
            "imaging", "ct", "mri", "ultrasound", "x-ray",
            "mammography", "tomosynthesis", "diagnostic imaging",
        ]

        filtered = []
        for ev in evidence:
            text = ev.text.lower()
            keyword_hit = any(k in text for k in keywords)
            distance_ok = (ev.distance is not None and ev.distance < 0.9)

            if keyword_hit or distance_ok:
                filtered.append(ev)

        print(f"[Verifier] Filter kept {len(filtered)}/{len(evidence)} evidence")
        return filtered

    def _rerank(
        self,
        scenario: CounterfactualScenario,
        evidence: List[EvidenceItem],
        top_k: int = 8,
    ) -> List[EvidenceItem]:
        if not self.use_reranker or not evidence:
            return evidence[:top_k]

        query = self._build_query_from_scenario(scenario)
        pairs = [[query, ev.text] for ev in evidence]
        scores = self.reranker.predict(pairs)

        ranked = sorted(
            zip(evidence, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        refined = [ev for ev, score in ranked if score > 0.3]

        if not refined:
            refined = [ev for ev, _ in ranked]

        print("[Verifier] Rerank scores (top 5):")
        for i, (ev, score) in enumerate(ranked[:5], 1):
            snippet = ev.text.replace("\n", " ")[:120]
            print(f"  [{i}] score={score:.3f} source={ev.source_type}:{ev.source} text={snippet}...")

        return refined[:top_k]

    def verify(self, scenario: CounterfactualScenario) -> VerifierOutput:
        print("\n[Verifier] Starting verification pipeline...")
        print(f"[Verifier] Scenario patient_id={scenario.patient_id}")
        print(f"[Verifier] Question: {scenario.question}")
        # 1) Optionally build / refresh scenario-specific PMC KB
        pmc_queries: List[str] = []
        if getattr(self.config, "build_pmc_kb_on_demand", False):
            pmc_queries = self.query_builder.build_pmc_queries(scenario)
            if pmc_queries:
                if getattr(self.config, "force_rebuild_pmc_kb", False):
                    print("[Verifier] force_rebuild_pmc_kb=True -> resetting collection")
                    self.kb.reset_collection()
                added_chunks = self.kb.build_from_pmc_queries(pmc_queries)
                print(f"[Verifier] PMC KB build: +{added_chunks} chunks")

        # 1.5) Optionally build ACR KB
        if getattr(self.config, "build_acr_kb_on_demand", False) and getattr(self.config, "include_acr_kb", True):
            try:
                from .acr.acr_builder import ACRBuilder
                acr_query = self.query_builder.build_acr_topic_query(scenario)
                print(f"[Verifier] ACR query: {acr_query}")
                acr_builder = ACRBuilder()
                acr_items = acr_builder.build_evidence_from_query(
                    acr_query,
                    top_k_topics=getattr(self.config, "acr_top_k_topics", 3),
                    max_scenarios_per_topic=getattr(self.config, "acr_max_scenarios_per_topic", 3)
                )
                if acr_items:
                    added_acr_chunks = self.kb.build_from_acr_items(acr_items)
                    print(f"[Verifier] ACR KB build: +{added_acr_chunks} chunks")
                else:
                    print("[Verifier] No ACR items found")
            except ImportError as e:
                print(f"[Verifier] ACR module not available: {e}")
            except Exception as e:
                print(f"[Verifier] ACR build failed: {e}")

        # 2) Final retrieval query for RAG
        query_text = self._build_query_from_scenario(scenario)
        print(f"[Verifier] Retrieval query text length={len(query_text)}")

        n_results = max(
            12,
            int(getattr(self.config, "retrieval_top_k", 8)),
        )
        raw_evidence = self.kb.query([query_text], n_results=n_results)

        print(f"[Verifier] Retrieved {len(raw_evidence)} evidence")
        for i, ev in enumerate(raw_evidence[:5], 1):
            snippet = ev.text.replace("\n", " ")[:120]
            print(
                f"  [Raw {i}] source={ev.source_type}:{ev.source} "
                f"dist={ev.distance if ev.distance is not None else 'NA'} text={snippet}..."
            )

        if not raw_evidence:
            return VerifierOutput(
                verdict="REJECT",
                rationale="No evidence retrieved from knowledge base.",
                references=[],
                checks={
                    "stage": "retrieval",
                    "reason": "no_evidence",
                    "query_text": query_text,
                },
                gaps=[
                    GapItem(
                        gap_type="retrieval_gap",
                        severity="high",
                        description="No evidence retrieved from knowledge base.",
                        suggested_next_step="Rebuild KB or improve retrieval query.",
                    )
                ],
            )

        filtered_evidence = self._filter_evidence(raw_evidence, scenario)
        print(f"[Verifier] After filter: {len(filtered_evidence)} evidence")

        if len(filtered_evidence) < 2:
            print("[Verifier] Fallback to raw evidence")
            filtered_evidence = raw_evidence

        top_k = int(getattr(self.config, "retrieval_top_k", 8))
        refined_evidence = self._rerank(scenario, filtered_evidence, top_k=top_k)
        print(f"[Verifier] After rerank: {len(refined_evidence)} evidence")

        if not refined_evidence:
            print("[Verifier] Rerank empty -> fallback")
            refined_evidence = filtered_evidence[:8]

        if not refined_evidence:
            return VerifierOutput(
                verdict="REJECT",
                rationale="No relevant medical evidence after filtering.",
                references=[],
                checks={
                    "stage": "filtering",
                    "reason": "no_relevant_evidence",
                    "query_text": query_text,
                },
                gaps=[
                    GapItem(
                        gap_type="evidence_gap",
                        severity="high",
                        description="No relevant evidence remained after filtering and reranking.",
                        suggested_next_step="Add more domain-specific sources or relax thresholds.",
                    )
                ],
            )

        valid_evidence = [
            ev for ev in refined_evidence
            if ev.distance is None or ev.distance < 1.2
        ]

        if not valid_evidence:
            return VerifierOutput(
                verdict="REJECT",
                rationale="Retrieved evidence was not sufficiently relevant to the scenario.",
                references=refined_evidence,
                checks={
                    "stage": "screening",
                    "reason": "domain_mismatch",
                    "query_text": query_text,
                },
                gaps=[
                    GapItem(
                        gap_type="domain_mismatch",
                        severity="high",
                        description="Retrieved evidence appears mismatched to the scenario.",
                        suggested_next_step="Use more scenario-specific sources and improve retrieval chunks.",
                    )
                ],
            )

        print(f"[Verifier] Valid evidence count before judge: {len(valid_evidence)}")
        verdict, rationale, checks, gaps = self.judge.judge(
            scenario,
            valid_evidence,
        )

        merged_checks = {
            "stage": "complete",
            "query_text": query_text,
            "retrieved_count": len(raw_evidence),
            "filtered_count": len(filtered_evidence),
            "final_evidence_count": len(valid_evidence),
            **checks,
        }

        return VerifierOutput(
            verdict=verdict,
            rationale=rationale,
            references=valid_evidence,
            checks=merged_checks,
            gaps=gaps,
        )
