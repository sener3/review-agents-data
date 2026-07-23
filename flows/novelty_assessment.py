from providers.base import LLMProvider
from schemas import NoveltyAssessmentOutput, ParsedManuscript
from utils.manuscript import build_manuscript_context
from utils.prompt_renderer import render_prompt
from utils.prompts import NOVELTY_ASSESSMENT_PROMPT, SYSTEM_REVIEW_POLICY


def generate_novelty_assessment(
    provider: LLMProvider,
    manuscript: ParsedManuscript,
    shared_context: str,
    retrieval_context: str,
) -> NoveltyAssessmentOutput:
    manuscript_context = build_manuscript_context(manuscript)
    prompt = render_prompt(
        NOVELTY_ASSESSMENT_PROMPT,
        manuscript_context=manuscript_context,
        shared_context=shared_context,
        retrieval_context=retrieval_context,
    )
    return provider.generate_structured(
        system_prompt=SYSTEM_REVIEW_POLICY,
        user_prompt=prompt,
        response_model=NoveltyAssessmentOutput,
        temperature=0.35,
    )
