from typing import List, Tuple, Dict, Any
import json
import re

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from .config import VerifierConfig
from .schemas import CounterfactualScenario, EvidenceItem, GapItem


class QwenJudge:
    def __init__(self, config: VerifierConfig):
        self.config = config
        model_name = config.qwen_model_name

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        print("[Judge] Loading Qwen model...")

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        print("[Judge] Qwen judge ready on GPU!")

    def _build_prompt(self, scenario: CounterfactualScenario, evidence: List[EvidenceItem]) -> str:
        evidence_text = "\n\n".join(
            [f"[E{i+1}] {ev.text}" for i, ev in enumerate(evidence)]
        )

        return f"""
You are a STRICT evidence-based clinical judge for counterfactual patient management review.

Your job is to verify whether the predicted counterfactual outcome is supported by the retrieved evidence.

You must use BOTH:
1. general medical evidence
2. patient-specific longitudinal evidence, if provided

IMPORTANT RULES
- Your ONLY allowed basis is the RETRIEVED EVIDENCE below.
- You MUST NOT use outside medical knowledge.
- You MUST NOT use common sense or assumptions that are not grounded in the evidence.
- If an important claim is not supported by evidence, you must not return PASS.
- Future patient outcomes (such as readmission, complications, later labs, or later diagnoses) may be used as supporting context, but they DO NOT automatically prove the counterfactual is correct.

When reasoning, consider:
- whether the intervention is clinically indicated based on the patient's state
- whether the timing of the intervention is important
- whether the intervention avoids unnecessary or excessive treatment
- whether the overall reasoning aligns with standard clinical care

Do NOT output the above considerations as separate fields.
Use them internally to support your final judgment.

TASK

Step 1:
Break the predicted outcome into atomic claims.

Step 2:
For each claim, output:
- claim
- status: supported / unsupported / contradicted
- evidence_ids: list of evidence ids such as E1, E2

Step 3:
Give an overall verdict:
- PASS: all key claims are supported by evidence
- FLAG: partially supported, mixed, or incomplete evidence
- REJECT: key claims are unsupported or contradicted by evidence

Return JSON only.

SCENARIO

Question:
{scenario.question}

Action:
{scenario.modification_details.get('description', '')}

Predicted Outcome:
{scenario.predicted_outcome.rationale}

Patient Context:
- patient_id: {scenario.patient_id}
- intervention_point: {scenario.intervention_point}
- scenario_rationale: {scenario.rationale or ""}
- metadata: {scenario.metadata}

RETRIEVED EVIDENCE
Evidence may include:
- guideline-level recommendations (e.g., ACR Appropriateness Criteria)
- literature evidence (e.g., PMC articles)

{evidence_text if evidence else "NO EVIDENCE"}

{{
  "claims": [
    {{
      "claim": "...",
      "status": "supported | unsupported | contradicted",
      "evidence_ids": ["E1"]
    }}
  ],
  "final_verdict": "PASS | FLAG | REJECT",
  "reason": "..."
}}
""".strip()

    def _safe_parse(self, text: str) -> Dict[str, Any] | None:
        try:
            return json.loads(text)
        except Exception:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except Exception:
                    pass
        return None

    def _enforce_claims(self, claims: List[Dict[str, Any]], verdict: str) -> str:
        if verdict == "PASS":
            for c in claims:
                if c.get("status") != "supported":
                    return "FLAG"
                if not c.get("evidence_ids"):
                    return "FLAG"
        return verdict

    def judge(
        self,
        scenario: CounterfactualScenario,
        evidence: List[EvidenceItem]
    ) -> Tuple[str, str, Dict[str, Any], List[GapItem]]:

        prompt = self._build_prompt(scenario, evidence)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_length = inputs.input_ids.shape[-1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.0,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        print(f"[Judge] Raw response:\n{response}\n")

        parsed = self._safe_parse(response)

        if parsed:
            claims = parsed.get("claims", [])
            verdict = parsed.get("final_verdict", "FLAG")
            reason = parsed.get("reason", "")
        else:
            claims = []
            verdict = "FLAG"
            reason = response

        verdict = self._enforce_claims(claims, verdict)

        gaps: List[GapItem] = []
        if verdict != "PASS":
            gaps.append(
                GapItem(
                    gap_type="evidence_mismatch",
                    severity="high",
                    description="Evidence does not fully support the predicted outcome.",
                    suggested_next_step="Improve retrieval or refine scenario."
                )
            )

        checks = {
            "judge_model": "qwen_local_gpu",
            "prompt_tokens": str(input_length),
            "parsed": str(bool(parsed)),
            "claim_count": len(claims),
            "evidence_count": len(evidence),
        }

        return verdict, reason, checks, gaps
