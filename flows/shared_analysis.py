from providers.base import LLMProvider
from schemas import PaperClassification, ParsedManuscript, SharedAnalysisOutput
from utils.manuscript import build_manuscript_context
from utils.prompt_renderer import render_prompt
from utils.prompts import SHARED_ANALYSIS_PROMPT, SYSTEM_REVIEW_POLICY


def generate_shared_analysis(
    provider: LLMProvider,
    manuscript: ParsedManuscript,
    paper_context: PaperClassification,
) -> SharedAnalysisOutput:
    manuscript_context = build_manuscript_context(manuscript)
    prompt = render_prompt(
        SHARED_ANALYSIS_PROMPT,
        paper_context=paper_context.model_dump_json(indent=2),
        manuscript_context=manuscript_context,
    )
    return provider.generate_structured(
        system_prompt=SYSTEM_REVIEW_POLICY,
        user_prompt=prompt,
        response_model=SharedAnalysisOutput,
        temperature=0.1,
    )
