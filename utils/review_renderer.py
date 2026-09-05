from __future__ import annotations

from schemas import (
    ClaimVerificationOutput,
    FinalHumanReviewPacket,
    FinalPaperContext,
    GeneratedARRReview,
    GeneratedARRScores,
    NoveltyAssessmentOutput,
    ReviewerOutput,
)
from utils.dedup import (
    collect_claim_alerts,
    collect_key_strengths,
    collect_key_weaknesses,
    collect_questions,
)


def shorten_text(text: str, max_words: int = 160) -> str:
    words = text.strip().split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).strip() + "..."


def build_final_summary(
    reviewer_outputs: list[ReviewerOutput],
    claim_verification: ClaimVerificationOutput,
) -> str:
    summaries = [r.summary.strip() for r in reviewer_outputs if r.summary.strip()]
    base = summaries[0] if summaries else ""

    if claim_verification.unsupported_or_overstated_claims:
        base += " Several manuscript claims should be narrowed or supported with additional evidence."

    return shorten_text(base.strip(), max_words=170)


def build_novelty_note(novelty_assessment: NoveltyAssessmentOutput) -> str:
    return (
        f"Novelty assessment: {novelty_assessment.verdict} "
        f"(level: {novelty_assessment.novelty_level})."
    ).strip()


def build_arr_like_review(
    reviewer_outputs: list[ReviewerOutput],
    claim_verification: ClaimVerificationOutput,
    novelty_assessment: NoveltyAssessmentOutput,
    scores: GeneratedARRScores,
) -> GeneratedARRReview:
    return GeneratedARRReview(
        scores=scores,
        summary=build_final_summary(reviewer_outputs, claim_verification),
        key_strengths=collect_key_strengths(reviewer_outputs, limit=4),
        key_weaknesses=collect_key_weaknesses(reviewer_outputs, limit=5),
        questions_for_authors=collect_questions(reviewer_outputs, limit=5),
        claim_alerts=collect_claim_alerts(claim_verification, limit=3),
        novelty_note=build_novelty_note(novelty_assessment),
    )


def build_final_human_review_packet(
    research_question: str,
    main_claims: list[str],
    arr_review: GeneratedARRReview,
    editor_summary: str,
) -> FinalHumanReviewPacket:
    return FinalHumanReviewPacket(
        paper_context=FinalPaperContext(
            research_question=research_question,
            main_claims=main_claims[:4],
        ),
        review=arr_review,
        editor_summary=shorten_text(editor_summary, max_words=140),
    )
