# evo-med-agent Clinical Counterfactual Verifier

A practical, evidence-first tool that checks whether a medical AI's “what if we change treatment?” prediction is actually supported by real research and clinical guidelines.

This project is designed so that non-technical users, clinicians, and decision makers can understand a model’s recommendation through evidence, not guesswork.

## Why this exists

When a medical AI proposes a counterfactual treatment—"what if we change the treatment plan?"—it is easy to produce a convincing answer without proof. This project adds a second layer: it verifies those recommendations by searching trusted medical sources and returns a clear verdict.

## What it does in plain terms

- Reads a structured clinical scenario: patient details, current treatment, proposed change, and the model’s predicted outcome.
- Searches medical evidence automatically from two sources:
  - PMC research articles
  - ACR appropriateness guidelines
- Judges the prediction using only the evidence it finds.
- Produces a simple, structured result:
  - `PASS` if evidence supports the prediction
  - `FLAG` if the evidence is mixed or incomplete
  - `REJECT` if evidence does not support the claim

## Big picture: the mission

This is not a medical chatbot.
It is an automated evidence verifier for counterfactual medical decisions.

It is built to:
- make model recommendations auditable
- reduce hidden risks in clinical AI outputs
- turn complex medical reasoning into a verifiable workflow
- help people understand whether a recommendation is backed by real evidence

## Why this project is special

- Focused on counterfactual medical decisions, not general chat.
- Uses both medical research papers and clinical guideline recommendations.
- Converts clinical questions into search queries automatically.
- Runs a local judge model, so it can work offline without sending data to the cloud.
- Outputs machine-readable JSON for audits, reports, or system integration.

## Simple workflow

1. **Understand the case**
   - The system reads patient status, current treatment, the suggested change, and the expected outcome.

2. **Find evidence**
   - It searches PMC articles and ACR guidelines.
   - It converts matching content into searchable evidence chunks.

3. **Make the call**
   - The judge model evaluates only that evidence.
   - It generates a verdict and explains why.

## Core implementation

### `main.py`

The command-line entry point.
- Loads a scenario file.
- Runs the full verification pipeline.
- Saves the final JSON result.

### `config.py`

Central settings for the whole app:
- Chroma vector database connection
- whether to build PMC knowledge on demand
- whether to include ACR guidance
- retrieval limits, chunking, and model options
- judge model settings

### `query_builder.py`

Turns scenarios into search-ready queries.
- Reads patient state and treatment change.
- Summarizes the clinical question.
- Builds PMC search queries.
- Extracts keywords for ACR guideline lookup.

### `kb.py`

Builds and queries the knowledge base.
- Downloads PMC articles and parses full text.
- Chunks articles into evidence pieces.
- Stores them in Chroma.
- Retrieves the most relevant evidence chunks for a question.

### `verifier.py`

Orchestrates the full flow.
- Rebuilds or loads PMC/ACR knowledge.
- Retrieves and filters evidence.
- Optionally reranks evidence.
- Sends final evidence to the judge.

### `judge.py`

The evidence reviewer.
- Evaluates only retrieved evidence.
- Breaks the prediction into smaller claims.
- Labels claims as supported / unsupported / contradicted.
- Produces an overall verdict and reasoning.

### `acr/`

Handles ACR guideline support.
- Searches ACR topics.
- Parses recommendation tables.
- Converts guideline content into evidence chunks.

## What is included in the repo

- `README.md` — this guide
- `main.py` — CLI entrypoint
- `config.py` — runtime configuration
- `query_builder.py` — scenario-to-search logic
- `kb.py` — knowledge base build and retrieval
- `verifier.py` — verification orchestration
- `judge.py` — evidence judgment logic
- `schemas.py` — input/output JSON schema definitions
- `demo_scenario.json` — example input file
- `example_output_schema.json` — example output format
- `acr/` — ACR guideline tools
- `outputs/` — sample outputs

## How to run

Make sure dependencies are installed and Chroma is running.

```bash
python main.py \
  --scenario demo_scenario.json \
  --output outputs/verifier_output.json \
  --rebuild_kb \
  --chroma_host localhost \
  --chroma_port 8000
```

Arguments:
- `--scenario` — path to the clinical scenario input
- `--output` — path to write the verifier result JSON
- `--rebuild_kb` — force rebuilding the knowledge base
- `--chroma_host` / `--chroma_port` — Chroma service address

## Requirements

- GPU with CUDA support
- Python packages: `torch`, `transformers`, `bitsandbytes`, `sentence-transformers`, `chromadb`, `requests`, `beautifulsoup4`
- Local Chroma vector search service
- Configured NCBI email for PMC access

## Standalone PMC KB builder

You can prebuild the PMC knowledge base with `build_pmc_kb_v2.py`:

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
- reads PMC queries from `queries.txt`
- searches PMC and deduplicates article IDs
- downloads and parses full-text XML
- chunks and embeds content
- saves the result into Chroma

Then run the verifier with `build_pmc_kb_on_demand=False` to use the prebuilt collection.

## Integration notes

- The verifier is modular and reusable.
- `KnowledgeBase` can be used for other clinical retrieval tasks.
- `QwenJudge` can be replaced with another model that follows the same JSON contract.
- `QueryBuilder` can be extended beyond infection-focused scenarios.

> For safety, this tool is decision support only and does not replace clinician judgment.

## My Learning: Results and Insights 📌

- The project shows how to turn a counterfactual clinical question into a verifiable evidence workflow.
- It proves that medical AI outputs can be audited by combining retrieval and local judgment.
- The most important insight is that evidence-driven review can make clinical recommendations clearer and safer.
- Building a structured pipeline around PMC and ACR makes the verification process both explainable and actionable.
