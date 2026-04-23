from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class VerifierConfig:
    url_list: List[str] = field(
        default_factory=lambda: [
            "https://medlineplus.gov/lab-tests/white-blood-count-wbc/",
            "https://medlineplus.gov/druginfo/meds/a604038.html",
            "https://medlineplus.gov/ency/article/000123.htm",
        ]
    )

    # local case knowledge
    local_case_dir: Optional[str] = None
    local_case_glob: str = "*.json"
    include_web_kb: bool = True
    include_local_case_kb: bool = True

    chroma_host: str = "localhost"
    chroma_port: int = 8000
    # main retrieval collection; will typically hold scenario-specific PMC KB
    collection_name: str = "pmc_kb_demo_test"

    # PMC / NCBI settings for scenario-focused KB building
    pmc_email: str = "carachen@gmail.com"
    pmc_api_key: Optional[str] = None
    pmc_collection_name: str = "pmc_kb_demo_test"
    pmc_max_per_query: int = 10
    pmc_max_total_articles: int = 50

    # control whether verifier builds PMC KB on demand
    build_pmc_kb_on_demand: bool = True
    force_rebuild_pmc_kb: bool = False

    # ACR settings for guideline-based KB building
    include_acr_kb: bool = True
    build_acr_kb_on_demand: bool = True
    acr_top_k_topics: int = 3
    acr_max_scenarios_per_topic: int = 3
    acr_use_topic_level_only: bool = True
    acr_collection_name: str = "acr_kb_demo"

    chunk_size: int = 900
    chunk_overlap: int = 150
    min_chunk_len: int = 200
    max_title_len: int = 120

    top_k: int = 5
    retrieval_top_k: int = 8
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"

    min_evidence_for_pass: int = 2

    action_names: Dict[int, str] = field(
        default_factory=lambda: {
            0: "MRSA_targeted_antibiotics",
            1: "hospital_discharge",
        }
    )

    qwen_base_url: str = "http://localhost:8001/v1"
    qwen_api_key: str = "EMPTY"
    qwen_model_name : str= "Qwen/Qwen2-1.5B-Instruct"
    qwen_temperature: float = 0.1
    qwen_max_tokens: int = 1024
