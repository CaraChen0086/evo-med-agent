from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .config import VerifierConfig
    from .schemas import CounterfactualScenario
    from .verifier import RAGVerifier
except ImportError:
    from config import VerifierConfig
    from schemas import CounterfactualScenario
    from verifier import RAGVerifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAG Verifier")

    parser.add_argument("--scenario", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs/verifier_output.json")
    parser.add_argument("--rebuild_kb", action="store_true")

    # 👉 新增：Chroma server
    parser.add_argument("--chroma_host", type=str, default="localhost")
    parser.add_argument("--chroma_port", type=int, default=8000)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # server config，
    config = VerifierConfig(
        chroma_host=args.chroma_host,
        chroma_port=args.chroma_port,
        force_rebuild_pmc_kb=args.rebuild_kb,
    )

    verifier = RAGVerifier(config)

    # scenario
    scenario_data = json.loads(Path(args.scenario).read_text())
    scenario = CounterfactualScenario.from_dict(scenario_data)

    # verifier
    output = verifier.verify(scenario)

    # server 
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output.to_dict(), indent=2))

    print(f"[Done] Output saved to {output_path.resolve()}")


if __name__ == "__main__":
    main()
