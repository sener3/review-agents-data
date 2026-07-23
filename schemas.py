from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

PaperType = Literal[
    "empirical",
    "methodological",
    "review",
    "theoretical",
    "hybrid",
    "resource",
    "unknown",
]
ConfidenceLevel = Literal["low", "medium", "high"]
SeverityLevel = Literal["minor", "moderate", "major"]
SpecialistRole = Literal[
    "structure_analyst",
    "methodology_auditor",
    "novelty_assessor_incremental",
    "novelty_assessor_disruptive",
    "claim_verifier",
]


class ParsedSection(BaseModel):
    heading: Optional[str] = None
    text: str = ""


class ParsedReference(BaseModel):
    title: Optional[str] = None
    author: List[str] = Field(default_factory=list)
    venue: Optional[str] = None
    citeRegEx: Optional[str] = None
    shortCiteRegEx: Optional[str] = None
    year: Optional[int] = None


class ParsedReferenceMention(BaseModel):
    referenceID: int
    context: str
    startOffset: Optional[int] = None
    endOffset: Optional[int] = None


class ParsedMetadata(BaseModel):
    source: Optional[str] = None
    title: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    emails: List[str] = Field(default_factory=list)
    sections: List[ParsedSection] = Field(default_factory=list)
    references: List[ParsedReference] = Field(default_factory=list)
    referenceMentions: List[ParsedReferenceMention] = Field(default_factory=list)
    year: Optional[int] = None
    abstractText: Optional[str] = None
    creator: Optional[str] = None


class ParsedManuscript(BaseModel):
    name: Optional[str] = None
    metadata: ParsedMetadata


class PaperClassification(BaseModel):
    paper_type: PaperType
    epistemic_stance: Literal["empirical", "theoretical", "hybrid", "unknown"] = (
        "unknown"
    )
    research_question: str
    contribution_type: List[str] = Field(default_factory=list)
    contribution_style: List[str] = Field(default_factory=list)
    main_claims: List[str] = Field(default_factory=list)
    expected_evaluation_dimensions: List[str] = Field(default_factory=list)
    confidence: ConfidenceLevel


class CorpusDocument(BaseModel):
    document_id: str
    title: str
    authors: List[str] = Field(default_factory=list)
    venue: Optional[str] = None
    year: Optional[int] = None
    abstract_or_snippet: str = ""
    similarity_score: float = 0.0
    source: str = "external_q1q2_qdrant"
    source_trace: Optional[str] = None


class CorpusContext(BaseModel):
    query: str
    top_documents: List[CorpusDocument] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)


class ClaimItem(BaseModel):
    claim_id: str
    claim_label: str
    claim_text: str
    section_anchor: str
    evidence: List[str] = Field(default_factory=list)
    uncertainty_flags: List[str] = Field(default_factory=list)


class AssumptionItem(BaseModel):
    assumption_id: str
    assumption_text: str
    assumption_type: Literal["explicit", "implicit", "operational", "evaluation"] = (
        "explicit"
    )
    section_anchor: str
    supporting_evidence: List[str] = Field(default_factory=list)
    risk_if_unmet: str = ""


class SharedAnalysisOutput(BaseModel):
    extracted_claims: List[ClaimItem] = Field(default_factory=list)
    identified_assumptions: List[AssumptionItem] = Field(default_factory=list)
    argument_focus: List[str] = Field(default_factory=list)
    contribution_targets: List[str] = Field(default_factory=list)
    evaluation_targets: List[str] = Field(default_factory=list)
    uncertainty_flags: List[str] = Field(default_factory=list)
    chaining_notes: List[str] = Field(default_factory=list)


class ComparableWork(BaseModel):
    citation_label: str
    relevance: str
    similarity_or_difference: str
    source_trace: Optional[str] = None


class ReviewStrength(BaseModel):
    title: str
    description: str
    evidence: List[str] = Field(default_factory=list)
    confidence: ConfidenceLevel = "medium"

    @field_validator("title")
    @classmethod
    def validate_title(cls, v: str) -> str:
        v = (v or "").strip()
        if not v or v.lower() == "unnamed strength":
            raise ValueError("Strength title must be specific.")
        return v

    @field_validator("description")
    @classmethod
    def validate_description(cls, v: str) -> str:
        v = (v or "").strip()
        if len(v) < 20:
            raise ValueError("Strength description is too short.")
        return v


class ReviewConcern(BaseModel):
    issue_code: str
    title: str
    description: str
    severity: SeverityLevel = "moderate"
    evidence: List[str] = Field(default_factory=list)
    actionable_suggestion: str
    novelty_risk: bool = False
    methodological_flaw: bool = False
    confidence: ConfidenceLevel = "medium"

    @field_validator("issue_code")
    @classmethod
    def validate_issue_code(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("Concern issue_code is required.")
        return v

    @field_validator("title")
    @classmethod
    def validate_title(cls, v: str) -> str:
        v = (v or "").strip()
        if not v or v.lower() == "unnamed concern":
            raise ValueError("Concern title must be specific.")
        return v

    @field_validator("description")
    @classmethod
    def validate_description(cls, v: str) -> str:
        v = (v or "").strip()
        if len(v) < 20:
            raise ValueError("Concern description is too short.")
        return v

    @field_validator("actionable_suggestion")
    @classmethod
    def validate_suggestion(cls, v: str) -> str:
        v = (v or "").strip()
        if len(v) < 12:
            raise ValueError("Concern actionable_suggestion is too short.")
        return v


class CreativitySafeguardAssessment(BaseModel):
    lacks_close_analogs: bool = False
    treat_as_innovation_signal: bool = False
    novelty_not_validity_note: str = ""
    counterfactual_conditions_for_success: List[str] = Field(default_factory=list)


class ConfidenceInterval(BaseModel):
    lower: float = 0.35
    upper: float = 0.75
    rationale: str = ""


class NoveltyLensView(BaseModel):
    lens: str
    verdict: str
    rationale: str
    evidence: List[str] = Field(default_factory=list)
    uncertainty_flags: List[str] = Field(default_factory=list)


class SpecialistReviewOutput(BaseModel):
    reviewer_id: str
    provider: str
    model: str
    reviewer_label: str
    cognitive_role: SpecialistRole
    epistemic_variant: str
    paper_type_assumed: PaperType = "unknown"
    prior_stage_inputs: List[str] = Field(default_factory=list)
    analytical_summary: str
    logical_argument_graph: List[str] = Field(default_factory=list)
    identified_claims: List[ClaimItem] = Field(default_factory=list)
    strengths: List[ReviewStrength] = Field(default_factory=list)
    concerns: List[ReviewConcern] = Field(default_factory=list)
    comparable_works: List[ComparableWork] = Field(default_factory=list)
    uncertainty_flags: List[str] = Field(default_factory=list)
    alternative_interpretations: List[str] = Field(default_factory=list)
    transparency_notes: List[str] = Field(default_factory=list)
    creativity_safeguard: CreativitySafeguardAssessment = Field(
        default_factory=CreativitySafeguardAssessment
    )
    novelty_lenses: List[NoveltyLensView] = Field(default_factory=list)
    confidence_interval: ConfidenceInterval = Field(default_factory=ConfidenceInterval)


class VerifiedClaim(BaseModel):
    claim: str
    status: Literal["supported", "partially_supported", "unsupported_or_overstated"]
    evidence: List[str] = Field(default_factory=list)
    literature_context: List[str] = Field(default_factory=list)
    note: Optional[str] = None
    confidence: ConfidenceLevel = "medium"


class ClaimVerificationOutput(BaseModel):
    supported_claims: List[VerifiedClaim] = Field(default_factory=list)
    partially_supported_claims: List[VerifiedClaim] = Field(default_factory=list)
    unsupported_or_overstated_claims: List[VerifiedClaim] = Field(default_factory=list)
    missing_evidence: List[str] = Field(default_factory=list)
    transparency_notes: List[str] = Field(default_factory=list)
    confidence_interval: ConfidenceInterval = Field(default_factory=ConfidenceInterval)


class NoveltyAssessmentOutput(BaseModel):
    verdict: str
    novelty_level: Literal["low", "moderate", "high"]
    main_novelty_claim: str
    closest_prior_art_risk: List[str] = Field(default_factory=list)
    why_incremental_or_distinct: List[str] = Field(default_factory=list)
    creativity_safeguard: CreativitySafeguardAssessment = Field(
        default_factory=CreativitySafeguardAssessment
    )
    comparable_works: List[ComparableWork] = Field(default_factory=list)
    transparency_notes: List[str] = Field(default_factory=list)
    confidence_interval: ConfidenceInterval = Field(default_factory=ConfidenceInterval)


class MetaReviewOutput(BaseModel):
    consensus_findings: List[str] = Field(default_factory=list)
    disagreements: List[str] = Field(default_factory=list)
    uncertainty_hotspots: List[str] = Field(default_factory=list)
    model_diversity_effects: List[str] = Field(default_factory=list)
    editor_guidance: str
    author_guidance: str
    transparency_report: List[str] = Field(default_factory=list)
    epistemic_limits: List[str] = Field(default_factory=list)




class HumanFacingMetaReview(BaseModel):
    is_meta_review: bool = True
    summary: str
    strengths: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)
    consensus_snapshot: List[str] = Field(default_factory=list)
    disagreement_snapshot: List[str] = Field(default_factory=list)
    uncertainty_snapshot: List[str] = Field(default_factory=list)
    source_trace: List[str] = Field(default_factory=list)
    transparency_notes: List[str] = Field(default_factory=list)
    editor_guidance: str
    author_guidance: str
    comments: str
    confidence_interval: ConfidenceInterval = Field(default_factory=ConfidenceInterval)

class PromptExecutionTrace(BaseModel):
    stage: str
    provider: str
    model: str
    temperature: float
    role: str
    input_stages: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class ReviewAudit(BaseModel):
    corpus_contexts: Dict[str, CorpusContext] = Field(default_factory=dict)
    prompt_execution_trace: List[PromptExecutionTrace] = Field(default_factory=list)


class PluralisticReviewOutput(BaseModel):
    paper_context: PaperClassification
    shared_analysis: SharedAnalysisOutput
    specialist_reviews: List[SpecialistReviewOutput] = Field(default_factory=list)
    meta_review: MetaReviewOutput
    human_facing_meta_review: HumanFacingMetaReview
    audit: Optional[ReviewAudit] = None
