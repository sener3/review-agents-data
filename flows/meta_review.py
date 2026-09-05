import json

from providers.base import LLMProvider
from schemas import MetaReviewOutput, PaperClassification, SharedAnalysisOutput, SpecialistReviewOutput
from utils.prompt_renderer import render_prompt
from utils.prompts import META_REVIEW_PROMPT, SYSTEM_REVIEW_POLICY


def generate_meta_review(
    provider: LLMProvider,
    paper_context: PaperClassification,
    shared_analysis: SharedAnalysisOutput,
    reviewer_outputs: list[SpecialistReviewOutput],
) -> MetaReviewOutput:
    prompt = render_prompt(
        META_REVIEW_PROMPT,
        paper_context=paper_context.model_dump_json(indent=2),
        shared_analysis=shared_analysis.model_dump_json(indent=2),
        reviews_json=json.dumps([x.model_dump() for x in reviewer_outputs], indent=2),
    )
    return provider.generate_structured(
        system_prompt=SYSTEM_REVIEW_POLICY,
        user_prompt=prompt,
        response_model=MetaReviewOutput,
        temperature=0.2,
    )
