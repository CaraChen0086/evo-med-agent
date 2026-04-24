# evo-med-agent Clinical Counterfactual Verifier

This project is an intelligent medical decision verification system. Its purpose is to take a counterfactual treatment recommendation from a model—"what happens if we change treatment?"—search medical evidence, and produce a clear, evidence-based verdict.

It is especially useful for:
- auditing medical AI conclusions
- validating counterfactual treatment recommendations
- adding a safety layer to clinical decision support
- generating structured, auditable judgment outputs

## How it works in three steps

1. **Understand the question**
   The system receives a structured clinical scenario: patient data, current treatment, proposed counterfactual treatment, predicted outcome, and related metadata. The system uses this input as a formal counterfactual question.

2. **Find evidence**
   It automatically searches two types of medical knowledge sources:
   - **PMC literature**: extracts relevant paragraphs from full-text medical articles
   - **ACR guidelines**: extracts recommendations from the American College of Radiology appropriateness criteria

   These results are converted into evidence chunks and stored in a vector search database for later judgment.

3. **Make a judgment**
   A local Qwen language model acts as the judge and evaluates only the retrieved evidence to answer:
   - Is the predicted outcome supported by evidence?
   - Is there any contradiction between evidence and prediction?
   - What is the structured final verdict: `PASS` / `FLAG` / `REJECT`?


## What makes this project special

- It is not a general chatbot. It is a medical counterfactual evidence verifier.
- It turns complex clinical reasoning into a process of retrieval + evidence evaluation + structured conclusion.
- For non-technical users, it behaves like an automated medical literature and guideline review tool.
- For developers and products, it outputs machine-readable JSON for analysis, memory storage, or system integration.


## Key features

- Supports two knowledge sources:
  - **PMC full-text papers** for research evidence
  - **ACR guideline recommendations** for standard imaging and appropriateness guidance
- Automatically generates scenario-specific search queries, so users do not need to write complicated search strings
- Applies evidence filtering and reranking to avoid using irrelevant content for judgment
- Runs the Qwen judge locally on GPU, reducing cloud dependency
- Produces structured JSON output suitable for audit, storage, and downstream analysis


## Explanation for non-technical users

Imagine a medical model says: “If this patient switches to MRSA-targeted antibiotics and leaves the hospital earlier, the infection risk will decrease.”

This project will automatically:
- read the patient state, current treatment, and model prediction
- search medical papers and professional guidelines for relevant evidence
- verify whether that evidence actually supports the prediction
- tell you:
  - `PASS`: evidence supports the prediction
  - `FLAG`: evidence is incomplete or mixed
  - `REJECT`: evidence does not support the prediction

That means it is not guessing. It is making a judgment based on evidence.


## What this project actually does

### 1. `main.py`: the command-line entrypoint

This is the script you run. It accepts a scenario file, executes the verification pipeline, and saves the result as JSON.

### 2. `config.py`: central configuration

All key runtime settings are defined here:
- Chroma vector database address
- whether to build PMC queries automatically
- whether to include ACR guidelines
- evidence retrieval limits, chunk size, and model choice
- Qwen judge settings

### 3. `query_builder.py`: turns a scenario into search queries

This module is one of the core logics. It:
- reads the original and counterfactual treatment changes
- summarizes the latest patient state
- generates multiple PMC search queries
- builds the final vector retrieval query
- extracts suitable keywords for ACR guideline search

### 4. `kb.py`: knowledge base construction and retrieval

It does two main jobs:
- downloads PMC medical papers, parses abstracts and body text, chunks them, and saves them into Chroma
- converts retrieval queries into vectors and returns the most relevant evidence chunks from Chroma

### 5. `verifier.py`: orchestration of the verification flow

This module sequences the entire process:
- optionally rebuilds the PMC / ACR knowledge base
- retrieves evidence and filters out unrelated material
- reranks evidence if a reranker is available
- passes the final evidence set to the judge for decision-making

### 6. `judge.py`: evidence judgment module

The judge uses Qwen to perform the final review:
- it is only allowed to reason from retrieved evidence
- it breaks the predicted outcome into smaller claims
- it labels each claim as supported / unsupported / contradicted
- it outputs an overall verdict and reasoning

### 7. `acr/`: ACR guideline support

The ACR folder implements:
- searching ACR topics by query
- parsing guideline pages and recommendation tables
- converting structured recommendations into retrieval-friendly evidence chunks


## Project structure overview

- `README.md`: project guide
- `config.py`: runtime configuration
- `schemas.py`: input/output data structure definitions
- `kb.py`: knowledge base creation and Chroma retrieval
- `query_builder.py`: scenario-to-query conversion
- `judge.py`: Qwen judgment logic
- `verifier.py`: end-to-end verification flow
- `main.py`: command-line entrypoint
- `demo_scenario.json`: example input scenario
- `example_output_schema.json`: example output format
- `acr/`: ACR search and parsing modules
- `outputs/`: sample output directory


## How to run

Assuming dependencies are installed and Chroma is running:

```bash
python main.py \
  --scenario demo_scenario.json \
  --output outputs/verifier_output.json \
  --rebuild_kb \
  --chroma_host localhost \
  --chroma_port 8000
```

Arguments:
- `--scenario`: path to the scenario file to verify
- `--output`: JSON output path for the verifier result
- `--rebuild_kb`: force rebuilding the knowledge base
- `--chroma_host` / `--chroma_port`: Chroma vector database address


## What you need before running

- a GPU with CUDA support
- installed packages: `torch`, `transformers`, `bitsandbytes`, `sentence-transformers`, `chromadb`, `requests`, `beautifulsoup4`
- a running local Chroma vector search service
- a configured NCBI email for PMC access


## The big goal of this project

This project is not about answering a medical question directly. It aims to:
- build a retrievable evidence chain for each counterfactual recommendation
- make the decision process verifiable and auditable
- add an evidence review layer to medical AI decision workflows
- let non-technical users understand whether a recommendation is supported by medical evidence


## How you can use it

- as a second audit layer for clinical model outputs
- as a validation tool during medical AI development
- as the backend for automated counterfactual report generation
- as a traceable component within a clinical decision support system

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
