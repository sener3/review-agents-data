import json

from providers.base import LLMProvider
from schemas import GeneratedARRScores, ReviewerOutput
from utils.prompt_renderer import render_prompt
from utils.prompts import ARR_SCORING_PROMPT, SYSTEM_REVIEW_POLICY


def score_arr_review(
    provider: LLMProvider,
    reviewer_outputs: list[ReviewerOutput],
) -> GeneratedARRScores:
    payload = json.dumps([x.model_dump() for x in reviewer_outputs], indent=2)
    prompt = render_prompt(
        ARR_SCORING_PROMPT,
        reviewer_outputs_json=payload,
    )
    return provider.generate_structured(
        system_prompt=SYSTEM_REVIEW_POLICY,
        user_prompt=prompt,
        response_model=GeneratedARRScores,
        temperature=0.1,
    )
