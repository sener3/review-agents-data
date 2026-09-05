from __future__ import annotations

import json
from contextlib import nullcontext
from typing import Any, Dict, List, Optional

from flows.classify_paper import classify_paper
from flows.generate_review import generate_specialist_review
from flows.human_facing_meta_review import generate_human_facing_meta_review
from flows.meta_review import generate_meta_review
from flows.shared_analysis import generate_shared_analysis
from providers.base import LLMProvider
from schemas import (
    ParsedManuscript,
    PluralisticReviewOutput,
    PromptExecutionTrace,
    ReviewAudit,
    SpecialistReviewOutput,
)
from utils.corpus_index import (
    build_guideline_corpus_context,
    build_literature_corpus_context,
    merge_corpus_contexts,
)

REVIEWER_BLUEPRINTS = [
    {
        "reviewer_id": "reviewer_1",
        "cognitive_role": "structure_analyst",
        "epistemic_variant": "logical-coherence-first",
        "temperature": 0.15,
        "retrieval_query": "paper structure logical flow section consistency argument continuity",
    },
    {
        "reviewer_id": "reviewer_2",
        "cognitive_role": "methodology_auditor",
        "epistemic_variant": "reproducibility-and-validity-first",
        "temperature": 0.2,
        "retrieval_query": "assumptions model choices statistical validity reproducibility robustness evaluation",
    },
    {
        "reviewer_id": "reviewer_3",
        "cognitive_role": "novelty_assessor_incremental",
        "epistemic_variant": "incremental-contribution lens",
        "temperature": 0.35,
        "retrieval_query": "extends existing paradigms semantic overlap comparable works incremental contribution",
    },
    {
        "reviewer_id": "reviewer_4",
        "cognitive_role": "novelty_assessor_disruptive",
        "epistemic_variant": "disruptive-thinking lens",
        "temperature": 0.45,
        "retrieval_query": "new conceptual frame divergent ideas sparse precedent creative contribution",
    },
    {
        "reviewer_id": "reviewer_5",
        "cognitive_role": "claim_verifier",
        "epistemic_variant": "evidence-and-uncertainty-first",
        "temperature": 0.1,
        "retrieval_query": "claim verification evidence support overstatement literature cross-check",
    },
]


def _build_prior_review_digest(reviews: List[SpecialistReviewOutput]) -> str:
    if not reviews:
        return "none"

    digest: List[Dict[str, Any]] = []
    for review in reviews:
        digest.append(
            {
                "reviewer_id": review.reviewer_id,
                "cognitive_role": review.cognitive_role,
                "analytical_summary": review.analytical_summary,
                "top_strengths": [item.title for item in review.strengths[:3]],
                "top_concerns": [item.title for item in review.concerns[:3]],
                "uncertainty_flags": review.uncertainty_flags[:5],
            }
        )
    return json.dumps(digest, indent=2, ensure_ascii=False)


def _reporting_stage(llm_reporter: Optional[Any], stage_name: str, **metadata: Any):
    if llm_reporter is None or not hasattr(llm_reporter, "stage"):
        return nullcontext()
    return llm_reporter.stage(stage_name, **metadata)


def run_prompt_orchestration_review_flow(
    provider: LLMProvider,
    manuscript: ParsedManuscript,
    include_audit: bool = True,
    llm_reporter: Optional[Any] = None,
    use_retrieval: bool = True,
) -> Dict[str, Any]:
    with _reporting_stage(llm_reporter, "meta_classification", role="paper_classifier"):
        paper_context = classify_paper(provider, manuscript)
    with _reporting_stage(
        llm_reporter, "shared_analysis", role="shared_claims_and_assumptions_extractor"
    ):
        shared_analysis = generate_shared_analysis(provider, manuscript, paper_context)

    corpus_contexts: Dict[str, Any] = {}
    specialist_reviews: List[SpecialistReviewOutput] = []
    execution_trace: List[PromptExecutionTrace] = [
        PromptExecutionTrace(
            stage="meta_classification",
            provider=provider.provider_name,
            model=provider.model,
            temperature=0.1,
            role="paper_classifier",
            input_stages=["manuscript"],
            notes=[
                "Infers paper type, epistemic stance, and contribution style before downstream prompting."
            ],
        ),
        PromptExecutionTrace(
            stage="shared_analysis",
            provider=provider.provider_name,
            model=provider.model,
            temperature=0.1,
            role="shared_claims_and_assumptions_extractor",
            input_stages=["meta_classification", "manuscript"],
            notes=[
                "Extracts shared claims and assumptions for downstream prompt chaining."
            ],
        ),
    ]

    for blueprint in REVIEWER_BLUEPRINTS:
        retrieval_query = (
            f"{paper_context.research_question} {blueprint['retrieval_query']}".strip()
        )
        if use_retrieval:
            literature_context = build_literature_corpus_context(
                manuscript,
                query=retrieval_query,
            )
            guideline_context = build_guideline_corpus_context(
                manuscript,
                query=f"{blueprint['cognitive_role']} {blueprint['epistemic_variant']} {blueprint['retrieval_query']}",
            )
            combined_context = merge_corpus_contexts(
                retrieval_query,
                literature_context,
                guideline_context,
            )

            retrieval_context = json.dumps(
                combined_context.model_dump(), indent=2, ensure_ascii=False
            )

            corpus_contexts[f"{blueprint['reviewer_id']}_literature"] = (
                literature_context
            )
            corpus_contexts[f"{blueprint['reviewer_id']}_guidelines"] = (
                guideline_context
            )
            corpus_contexts[f"{blueprint['reviewer_id']}_combined"] = combined_context

        else:
            retrieval_context = json.dumps(
                {
                    "ablation_mode": "no_retrieval",
                    "retrieval_enabled": False,
                    "literature_context": [],
                    "guideline_context": [],
                    "instruction": (
                        "No external literature or reviewer-guideline retrieval context is provided. "
                        "Do not invent retrieved evidence. Evaluate only from the manuscript, "
                        "paper classification, shared analysis, and prior specialist outputs if available."
                    ),
                },
                indent=2,
                ensure_ascii=False,
            )

        prior_stage_context = "none"
        if blueprint["cognitive_role"] == "claim_verifier":
            prior_stage_context = _build_prior_review_digest(specialist_reviews)

        with _reporting_stage(
            llm_reporter,
            "specialist_review",
            reviewer_id=blueprint["reviewer_id"],
            role=blueprint["cognitive_role"],
            epistemic_variant=blueprint["epistemic_variant"],
        ):
            review = generate_specialist_review(
                provider=provider,
                manuscript=manuscript,
                paper_context=paper_context,
                shared_analysis=shared_analysis,
                retrieval_context=retrieval_context,
                reviewer_id=blueprint["reviewer_id"],
                cognitive_role=blueprint["cognitive_role"],
                epistemic_variant=blueprint["epistemic_variant"],
                prior_stage_context=prior_stage_context,
                temperature=blueprint["temperature"],
            )
        specialist_reviews.append(review)

        input_stages = [
            "meta_classification",
            "shared_analysis",
            "external_q1q2_retrieval_context",
            "review_guidelines_retrieval_context",
        ]
        if blueprint["cognitive_role"] == "claim_verifier":
            input_stages.append("specialist_review_digests")

        execution_trace.append(
            PromptExecutionTrace(
                stage="specialist_review",
                provider=provider.provider_name,
                model=provider.model,
                temperature=blueprint["temperature"],
                role=blueprint["cognitive_role"],
                input_stages=input_stages,
                notes=[
                    f"Variant: {blueprint['epistemic_variant']}",
                    f"Reviewer label: {provider.reviewer_label}",
                    "Shared extraction stage is injected to preserve cross-review coherence.",
                    f"Literature query: {retrieval_query}",
                    f"Guideline query: {blueprint['cognitive_role']} {blueprint['epistemic_variant']} {blueprint['retrieval_query']}",
                    "Retrieved evidence comes from a dual runtime path: curated Q1/Q2 literature plus reviewer-guideline corpora in Qdrant.",
                ],
            )
        )

    with _reporting_stage(
        llm_reporter, "meta_review", role="pluralistic_meta_reviewer"
    ):
        meta_review = generate_meta_review(
            provider=provider,
            paper_context=paper_context,
            shared_analysis=shared_analysis,
            reviewer_outputs=specialist_reviews,
        )
    with _reporting_stage(
        llm_reporter, "human_facing_meta_review", role="human_style_meta_reviewer"
    ):
        human_facing_meta_review = generate_human_facing_meta_review(
            provider=provider,
            paper_context=paper_context,
            shared_analysis=shared_analysis,
            reviewer_outputs=specialist_reviews,
            meta_review=meta_review,
        )
    execution_trace.append(
        PromptExecutionTrace(
            stage="meta_review",
            provider=provider.provider_name,
            model=provider.model,
            temperature=0.2,
            role="pluralistic_meta_reviewer",
            input_stages=[
                "meta_classification",
                "shared_analysis",
                "specialist_reviews",
            ],
            notes=[
                "Synthesizes agreement, contradiction, and uncertainty rather than collapsing to a single verdict."
            ],
        )
    )
    execution_trace.append(
        PromptExecutionTrace(
            stage="human_facing_meta_review",
            provider=provider.provider_name,
            model=provider.model,
            temperature=0.15,
            role="human_style_meta_reviewer",
            input_stages=[
                "meta_classification",
                "shared_analysis",
                "specialist_reviews",
                "meta_review",
            ],
            notes=[
                "Exports a final expert-facing analytical synthesis with no scores and explicit transparency notes."
            ],
        )
    )

    audit = None
    if include_audit:
        audit = ReviewAudit(
            corpus_contexts=corpus_contexts,
            prompt_execution_trace=execution_trace,
        )

    final_output = PluralisticReviewOutput(
        paper_context=paper_context,
        shared_analysis=shared_analysis,
        specialist_reviews=specialist_reviews,
        meta_review=meta_review,
        human_facing_meta_review=human_facing_meta_review,
        audit=audit,
    )

    return final_output.model_dump()
