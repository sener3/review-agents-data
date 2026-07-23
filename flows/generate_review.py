from __future__ import annotations

import json

from schemas import SpecialistReviewOutput
from utils.prompt_renderer import render_prompt
from utils.prompts import (
    SPECIALIST_REVIEW_SYSTEM_PROMPT,
    SPECIALIST_REVIEW_USER_PROMPT,
)

ROLE_INSTRUCTIONS = {
    "structure_analyst": """
Reconstruct the manuscript's logical argument graph.
Test section continuity, claim-to-evidence linkage, and argument coherence.
Do not just summarize sections; identify where the reasoning strengthens or breaks.
""".strip(),
    "methodology_auditor": """
Validate assumptions, model choices, evaluation setup, reproducibility, and robustness step by step.
Focus on what must be true for the results to hold and what evidence is missing.
""".strip(),
    "novelty_assessor_incremental": """
Assess how the manuscript extends, refines, or reorganizes existing paradigms.
Look for semantic overlap, implicit inspirations, and whether the contribution is incremental but meaningful.
""".strip(),
    "novelty_assessor_disruptive": """
Assess whether the manuscript introduces a new conceptual frame even when direct precedent is sparse.
Protect unconventional but coherent ideas from being penalized merely for divergence from mainstream literature.
""".strip(),
    "claim_verifier": """
Cross-check assertions against manuscript evidence and retrieved prior work.
Distinguish supported, partially supported, and overstated claims inside your analytical discussion without turning the output into a binary verdict.
""".strip(),
}


def _build_manuscript_context(manuscript) -> str:
    if hasattr(manuscript, "to_prompt_context"):
        return manuscript.to_prompt_context()

    if hasattr(manuscript, "model_dump_json"):
        return manuscript.model_dump_json(indent=2)

    if hasattr(manuscript, "model_dump"):
        return json.dumps(manuscript.model_dump(), indent=2, ensure_ascii=False)

    return str(manuscript)


def generate_specialist_review(
    provider,
    manuscript,
    paper_context,
    shared_analysis,
    retrieval_context,
    reviewer_id,
    cognitive_role,
    epistemic_variant,
    prior_stage_context,
    temperature,
):
    manuscript_context = _build_manuscript_context(manuscript)

    user_prompt = render_prompt(
        SPECIALIST_REVIEW_USER_PROMPT,
        reviewer_id=reviewer_id,
        reviewer_label=provider.reviewer_label,
        cognitive_role=cognitive_role,
        epistemic_variant=epistemic_variant,
        role_instructions=ROLE_INSTRUCTIONS[cognitive_role],
        paper_context=paper_context.model_dump_json(indent=2),
        shared_analysis=shared_analysis.model_dump_json(indent=2),
        manuscript_context=manuscript_context,
        retrieval_context=retrieval_context,
        prior_stage_context=prior_stage_context,
    )

    response = provider.generate_structured(
        system_prompt=SPECIALIST_REVIEW_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_model=SpecialistReviewOutput,
        temperature=temperature,
    )

    response.reviewer_id = reviewer_id
    response.provider = provider.provider_name
    response.model = provider.model
    response.reviewer_label = provider.reviewer_label
    response.cognitive_role = cognitive_role
    response.epistemic_variant = epistemic_variant

    if not response.prior_stage_inputs:
        response.prior_stage_inputs = [
            "paper_context",
            "shared_analysis",
            "retrieval_context",
        ]
        if prior_stage_context.strip() and prior_stage_context.strip() != "none":
            response.prior_stage_inputs.append("prior_stage_digest")

    return response
