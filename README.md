# RAG Verifier v3 – Clinical Counterfactual Verifier

This repository implements a scenario‑focused Retrieval‑Augmented Generation (RAG) verifier for clinical counterfactual questions.  
Given a structured counterfactual scenario `(X, A_original, A_counterfactual, outcome, metadata)`, the system:
- builds or queries a Chroma knowledge base from web and/or PMC full‑text articles,
- retrieves and optionally re‑ranks relevant evidence,
- calls a local Qwen LLM judge on GPU, and
- outputs a structured JSON verdict for downstream analysis, memory, and MITS export.

The code is designed to plug into the broader `evo-med-agent` pipeline but can also be used as a standalone verifier.

**ACR Integration (v1)**: The system now supports ACR (American College of Radiology Appropriateness Criteria) as a second knowledge source alongside PMC. ACR provides guideline-level decision support through structured rating tables (procedure + appropriateness). PMC remains literature-focused. ACR v1 extracts core rating-table recommendations but does not fully parse narrative paragraphs.

---

## Repository structure

```text
rag_verifier_v3/
├── .gitignore
├── README.md
├── __init__.py
├── config.py              # VerifierConfig: URLs, PMC / Chroma / model settings
├── schemas.py             # Input/Output dataclasses and JSON schemas
├── kb.py                  # Web + PMC knowledge base builder and Chroma client
├── query_builder.py       # Scenario → PMC queries + retrieval query
├── judge.py               # Qwen-based LLM judge (PASS / FLAG / REJECT)
├── verifier.py            # End-to-end orchestration (KB + retrieval + judge)
├── build_pmc_kb_v2.py     # Standalone PMC KB builder CLI
├── main.py                # CLI entrypoint for running the verifier
├── demo_scenario.json     # Example counterfactual scenario input
├── example_output_schema.json  # Example of structured verifier output
├── acr/
│   ├── __init__.py
│   ├── acr_search.py       # Search ACR topics from public list
│   ├── acr_parser.py       # Parse ACR pages for rating tables
│   ├── acr_builder.py      # Orchestrate ACR search + parse + chunk
│   ├── acr_test.py         # ACR testing utilities
│   └── samples/
│       └── acr_topics_breast_cancer.json  # Example ACR search results
└── outputs/
    └── verifier_output.json  # Example verifier output
```

---

## High-level pipeline

1. **Input scenario**  
   A structured JSON scenario is parsed into `CounterfactualScenario` (see `schemas.py`).

2. **PMC KB building (optional, scenario‑focused)**  
   - `QueryBuilder.build_pmc_queries` generates multiple PMC search queries based on the scenario (question, action delta, latest state, outcome text).  
   - `KnowledgeBase.build_from_pmc_queries` (invoked from `RAGVerifier.verify`) uses NCBI E-utilities to:
     - search PMC,
     - fetch full‑text XML,
     - parse abstracts and body sections,
     - chunk and embed, and
     - upsert into a Chroma collection.

3. **ACR KB building (optional, guideline‑focused)**  
   - `QueryBuilder.build_acr_topic_query` generates an ACR topic search query.  
   - `ACRBuilder.build_evidence_from_query` (invoked from `RAGVerifier.verify`) uses the ACR module to:
     - search ACR topics,
     - parse rating table pages,
     - extract procedure + appropriateness recommendations,
     - chunk and embed, and
     - upsert into the same Chroma collection.

4. **Retrieval**  
   - `QueryBuilder.build_retrieval_query` constructs a short, focused retrieval query.  
   - `KnowledgeBase.query` encodes queries using a sentence‑transformer (e.g. `Qwen/Qwen3-Embedding-0.6B`) and queries a Chroma HTTP collection.  
   - Raw retrieved chunks are converted into `EvidenceItem` objects with source URL, text, distance, and metadata.

5. **Filtering and reranking**  
   - `RAGVerifier._filter_evidence` keeps evidence with infection/imaging‑related keywords or sufficiently good embedding distance.  
   - If a CrossEncoder reranker (`BAAI/bge-reranker-base`) is available, `RAGVerifier._rerank` reorders evidence by relevance and trims to top‑k.

6. **Judging**  
   - `QwenJudge` builds a strict JSON‑only prompt that:
     - enumerates retrieved evidence (including PMC literature and ACR guidelines),
     - instructs the model to decompose predicted outcome into claims,
     - assigns each claim a status (`supported` / `unsupported` / `contradicted`), and
     - chooses a global verdict (`PASS` / `FLAG` / `REJECT`).  
   - The Qwen model runs locally on GPU via `transformers` + `bitsandbytes` 4‑bit quantization.

7. **Structured output**  
   - `RAGVerifier.verify` packages the verdict, rationale, references, internal checks, gaps, and optional memory / MITS exports into a `VerifierOutput` dataclass and finally into JSON.

---

## Input schema (scenario JSON)

The verifier consumes a single scenario JSON file, which is parsed into `CounterfactualScenario`.  
An example is provided in `demo_scenario.json`.

Key fields:
- `patient_id: str`  
  Unique patient identifier used for logging and exports.

- `intervention_point: float`  
  Time index (e.g. hours since admission) at which the counterfactual intervention is applied.

- `X: List[List[float]]`  
  Longitudinal feature matrix; each inner list is a time step.  
  Feature names are optionally given in `metadata.feature_names`.

- `A_original: List[int]`  
  Original binary/multi‑hot action vector at `intervention_point`.

- `A_counterfactual: List[int]`  
  Counterfactual action vector proposed by the agent.

- `modification_type: Literal["action_change", "modality_mask", "temporal_shift"]`  
  Type of counterfactual modification.

- `modification_details: Dict[str, Any]`  
  Additional structured metadata about the intervention, typically including:
  - `changed_dims: List[int]`
  - `changed_targets: List[str]` (e.g. `"MRSA_targeted_antibiotics"`, `"hospital_discharge"`)
  - `description: str` – human‑readable description of the intervention.

- `predicted_outcome: object`  
  Parsed into `PredictedOutcome`:
  - `z_score: Optional[float]`
  - `raw_value: Optional[float]`  
    Internally encoded as `raw_WBC` or `raw_mmHg` when relevant.
  - `raw_name: Optional[str]`
  - `confidence: Optional[float]`
  - `rationale: Optional[str]` – free‑text explanation from the proposer.

- `question: str`  
  The clinical counterfactual question (e.g. “What would be the patient’s infection status if …?”).

- `rationale: Optional[str]`  
  Scenario‑level rationale (distinct from `predicted_outcome.rationale`).

- `ground_truth: Optional[Any]`  
  Structured ground truth from real data (e.g. rehospitalization reason, true lab values).

- `metadata: Dict[str, Any]`  
  Arbitrary scenario metadata. Common keys:
  - `dataset: str`
  - `feature_names: List[str]`
  - `split: str`

All fields in `demo_scenario.json` are supported and used by the query builder and judge prompt.

---

## Output schema (verifier JSON)

The main output is a JSON serialized `VerifierOutput`.  
An example format is provided in `example_output_schema.json`.

Top‑level fields:

- `verdict: str`  
  One of:
  - `"PASS"` – all key outcome claims are supported by retrieved evidence.  
  - `"FLAG"` – partially supported, mixed, or incomplete evidence.  
  - `"REJECT"` – key claims are unsupported or contradicted.

- `rationale: str`  
  Natural‑language explanation from the Qwen judge summarizing why the verdict was chosen.

- `references: List[EvidenceItem]`  
  Evidence actually passed to the judge. Each item has:
  - `source: str` – URL or provenance (web / PMC / local case).  
  - `chunk_id: str` – stable identifier in the Chroma collection.  
  - `text: str` – cleaned chunk text, often prefixed with topic/section headers.  
  - `distance: Optional[float]` – vector distance from retrieval.  
  - `source_type: str` – e.g. `"web"` or `"pmc"`.  
  - `supports: List[str]` – reserved for downstream claim mapping.

- `checks: Dict[str, Any]`  
  Internal diagnostics from the verifier and judge, for example:
  - `"stage"` – `"retrieval"`, `"filtering"`, `"screening"`, `"complete"`, etc.  
  - `"query_text"` – final retrieval query.  
  - `"retrieved_count"`, `"filtered_count"`, `"final_evidence_count"`.  
  - `"judge_model"`, `"prompt_tokens"`, `"parsed"`, `"claim_count"`, `"evidence_count"`.

- `gaps: List[GapItem]`  
  Structured descriptions of gaps identified during retrieval or judgment. Each gap includes:
  - `gap_type: str` – e.g. `"retrieval_gap"`, `"evidence_gap"`, `"domain_mismatch"`, `"evidence_mismatch"`.  
  - `severity: str` – `"low"`, `"medium"`, `"high"`.  
  - `description: str` – what is missing or misaligned.  
  - `suggested_next_step: str` – how to improve (e.g. rebuild KB, adjust queries, add sources).

- `memory_candidate: Optional[MemoryCandidate]`  
  Optional structured summary for long‑term memory:
  - `should_store: bool`  
  - `memory_key: str`  
  - `memory_type: str` – e.g. `"verified_counterfactual"`.  
  - `summary: str` – concise textual summary.  
  - `tags: List[str]` – structured tags (e.g. actions, outcomes).  
  - `provenance: Dict[str, Any]` – includes patient id, intervention point, evidence count, verifier version, etc.

- `mits_export: Optional[MITSExport]`  
  Optional export for MITS or downstream systems, including:
  - `patient_id: str`  
  - `intervention_point: float`  
  - `question: str`  
  - `action_delta: Dict[str, Dict[str, int]]` – action name → `{before, after}`.  
  - `verified_outcome: Dict[str, Any]` – structured outcome (z‑score, raw value, confidence).  
  - `evidence_summary: List[str]` – concise references, e.g. `"[1] ..."`  
  - `verifier_verdict: str`

When no evidence is retrieved or filtering removes all evidence, the verifier returns a `"REJECT"` verdict with a corresponding gap entry documenting why.

---

## Configuration (VerifierConfig)

`config.VerifierConfig` centralizes runtime configuration. Important fields:

- **Knowledge base sources**
  - `url_list: List[str]` – default MedlinePlus or other clinical URLs for web KB.  
  - `local_case_dir: Optional[str]` / `local_case_glob: str` – optional directory of case JSONs.  
  - `include_web_kb: bool` / `include_local_case_kb: bool` – toggle source types.

- **Chroma / embedding**
  - `chroma_host: str` / `chroma_port: int` – HTTP endpoint for Chroma server.  
  - `collection_name: str` – main retrieval collection name.  
  - `embedding_model: str` – sentence‑transformer model name (e.g. `Qwen/Qwen3-Embedding-0.6B`).  
  - `chunk_size`, `chunk_overlap`, `min_chunk_len`, `max_title_len` – text chunking parameters.  
  - `retrieval_top_k` / `top_k` – number of evidence items to retrieve.

- **PMC / NCBI**
  - `pmc_email: str` – email used for NCBI E‑utilities (required).  
  - `pmc_api_key: Optional[str]` – optional NCBI API key for higher rate limits.  
  - `pmc_collection_name: str` – Chroma collection for PMC chunks (often same as `collection_name`).  
  - `pmc_max_per_query`, `pmc_max_total_articles` – upper bounds for PMC fetch.  
  - `build_pmc_kb_on_demand: bool` – whether `verify()` builds PMC KB from scenario queries.  
  - `force_rebuild_pmc_kb: bool` – if true, drop and recreate the collection before building.

- **Actions and judge**
  - `action_names: Dict[int, str]` – mapping from action indices to human‑readable names.  
  - `min_evidence_for_pass: int` – minimal evidence count before allowing a `"PASS"` verdict.  
  - `qwen_model_name: str` – HF model id for Qwen judge (e.g. `Qwen/Qwen2-1.5B-Instruct`).  
  - `qwen_temperature`, `qwen_max_tokens` – generation parameters.  
  - `qwen_base_url`, `qwen_api_key` – reserved for HTTP‑based variants; currently the judge uses local HF loading.

Adjust these fields either by modifying `config.py` directly or by constructing a `VerifierConfig` in your own integration.

---

## Dependencies and environment

Core Python dependencies (non‑exhaustive):
- `torch` (with CUDA support)
- `transformers`
- `bitsandbytes`
- `sentence-transformers`
- `chromadb`
- `requests`
- `beautifulsoup4`

Install example (you may need to adapt versions to your environment):

```bash
pip install torch transformers bitsandbytes sentence-transformers chromadb beautifulsoup4 requests
```

Additional requirements:
- A running **Chroma** server accessible at `http://{chroma_host}:{chroma_port}`.  
- A GPU capable of hosting the Qwen judge model (e.g. `Qwen/Qwen2-1.5B-Instruct` with 4‑bit quantization).  
- Valid NCBI email (and ideally an API key) configured in `VerifierConfig` for PMC access.

---

## Running the verifier

From inside the `rag_verifier_v3` directory, with dependencies installed and a Chroma server running:

```bash
python main.py \
  --scenario demo_scenario.json \
  --output outputs/verifier_output.json \
  --rebuild_kb \
  --chroma_host localhost \
  --chroma_port 8000
```

Arguments:
- `--scenario` (required): path to the scenario JSON file.  
- `--output` (optional): output path for the verifier JSON (defaults to `outputs/verifier_output.json`).  
- `--rebuild_kb` (flag): if set, forces rebuilding the PMC / KB for this run (sets `force_rebuild_pmc_kb=True`).  
- `--chroma_host`, `--chroma_port`: override Chroma HTTP endpoint (default `localhost:8000`).

Alternatively, if you install this folder as a module on the Python path:

```bash
python -m rag_verifier_v3.main \
  --scenario path/to/scenario.json \
  --output path/to/output.json \
  --chroma_host your_chroma_host \
  --chroma_port your_chroma_port
```

The CLI prints progress logs and finally reports where the output JSON has been written.

---

## Standalone PMC KB builder

In addition to on‑demand KB building inside `RAGVerifier.verify`, you can pre‑build a PMC KB using `build_pmc_kb_v2.py`:

```bash
python build_pmc_kb_v2.py \
  --queries_file queries.txt \
  --email your_email@example.com \
  --collection_name pmc_kb_demo \
  --max_per_query 50 \
  --max_total_articles 200 \
  --embedding_model Qwen/Qwen3-Embedding-0.6B \
  --chroma_host localhost \
  --chroma_port 8000
```

This flow:
- reads one PMC query per line from `queries.txt`,  
- searches PMC, deduplicates article IDs across queries,  
- fetches full‑text XML, parses sections, chunks, and embeds, and  
- upserts all chunks into the specified Chroma collection.

You can then run the verifier with `build_pmc_kb_on_demand=False` and point it at this prebuilt collection.

---

## Integration notes

- The verifier is intentionally modular:
  - `KnowledgeBase` can be reused in other clinical RAG tasks.  
  - `QwenJudge` can be swapped out for another LLM or served model as long as it follows the same JSON contract.  
  - `QueryBuilder` encodes assumptions about infection‑focused scenarios but can be extended to other domains.

- For large‑scale evaluation:
  - Run `main.py` in a loop over many scenario JSON files.  
  - Persist both the verifier outputs and internal `checks`/`gaps` for audit and error analysis.

- For safety‑critical use, treat the verifier as **decision support** only; it does not replace clinical judgment.
