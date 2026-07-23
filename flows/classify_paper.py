from providers.base import LLMProvider
from utils.prompt_renderer import render_prompt

from schemas import PaperClassification, ParsedManuscript
from utils.manuscript import build_manuscript_context
from utils.prompts import CLASSIFY_PAPER_PROMPT, SYSTEM_REVIEW_POLICY


def classify_paper(
    provider: LLMProvider, manuscript: ParsedManuscript
) -> PaperClassification:
    manuscript_context = build_manuscript_context(manuscript)
    prompt = render_prompt(CLASSIFY_PAPER_PROMPT, manuscript_context=manuscript_context)
    return provider.generate_structured(
        system_prompt=SYSTEM_REVIEW_POLICY,
        user_prompt=prompt,
        response_model=PaperClassification,
        temperature=0.1,
    )
