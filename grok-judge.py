from __future__ import annotations

import html
import json
import logging
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable

import requests

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency fallback
    load_dotenv = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Hardcoded project values. No command-line arguments are used.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
PDFS_DIR = DATA_DIR / "pdfs"
HUMAN_REVIEW_DIR = DATA_DIR / "human-reviews"
GENERATED_REVIEW_DIRS = {
    "gpt": DATA_DIR / "gpt-reviews",
    "gemini": DATA_DIR / "gemini-reviews",
    "claude": DATA_DIR / "claude-reviews",
    "gpt-no-rag": DATA_DIR / "gpt-reviews-no-rag",
}

# Modify this value to control how many discovered manuscripts are scanned.
MAX_MANUSCRIPTS = 137

# Local source-to-anonymized-label mapping. This mapping is never sent to Grok;
# it is used only after the judgment is returned to create aggregate reports.
SOURCE_ORDER = ("human", "gpt", "gemini", "claude", "gpt-no-rag")

SOURCE_TO_LABEL = {
    "human": "A",
    "gpt": "B",
    "gemini": "C",
    "claude": "D",
    "gpt-no-rag": "E",
}
LABEL_TO_SOURCE = {label: source for source, label in SOURCE_TO_LABEL.items()}
REVIEW_LABELS = tuple(SOURCE_TO_LABEL[source] for source in SOURCE_ORDER)

OUTPUT_DIR = PROJECT_ROOT / "outputs-no-rag-100-sample-v2"
PAYLOAD_OUTPUT_DIR = OUTPUT_DIR / "sanitized-payloads-no-rag"
FINAL_REPORT_OUTPUT_PATH = OUTPUT_DIR / "final_report_no_rag.json"
BATCH_SUMMARY_OUTPUT_PATH = OUTPUT_DIR / "batch_summary_no_rag.json"

# Optional custom regex redactions. If this file is absent, defaults are used.
REDACTION_PATTERNS_PATH = DATA_DIR / "redactions.txt"

# Hardcoded runtime switches. Keep DRY_RUN=True for extraction/privacy testing;
# set to False to call Grok. The API key still must come from XAI_API_KEY.
DRY_RUN = False
STOP_ON_FIRST_ERROR = False
SKIP_EXISTING_JUDGMENTS = True
LOG_LEVEL = "INFO"

DEFAULT_XAI_BASE_URL = "https://api.x.ai/v1"
DEFAULT_XAI_MODEL = "grok-4.3"

ALLOWED_TOP_LEVEL_KEYS = {"reviews", "manuscript_text"}
REQUIRED_REVIEW_KEYS = set(REVIEW_LABELS)
QUALITY_SCORE_KEYS = {
    "factual_accuracy",
    "methodological_insight",
    "specificity",
    "actionability",
    "coverage_of_relevant_issues",
    "constructiveness",
}
PENALTY_KEYS = {
    "hallucinated_or_unsupported_claims",
    "unnecessary_verbosity_or_repetition",
}

# Generated review extraction is intentionally path-specific. The current archive
# uses human_facing_meta_review.comments, while some versions may use
# human_facing_synthesis.comments. Both are explicit allowed paths; arbitrary
# nested comments keys in generated review files are not extracted.
GENERATED_REVIEW_COMMENT_PATHS = (
    ("human_facing_synthesis", "comments"),
    ("human_facing_meta_review", "comments"),
)

# Conservative defaults. Add manuscript-/journal-specific names, affiliations,
# grant numbers, or other sensitive terms through data/redactions.txt.
DEFAULT_REDACTION_PATTERNS = [
    r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",  # email
    r"\b\d{4}-\d{4}-\d{4}-\d{3}[0-9X]\b",  # ORCID
]

# Used only when the value of an exact "comments" key is itself an object.
# This avoids forwarding reviewer metadata embedded under a comments object.
SENSITIVE_COMMENT_SUBKEY_RE = re.compile(
    r"(?:^|_)(?:id|identifier|name|email|affiliation|institution|department|"
    r"score|rating|grade|recommendation|decision|date|time|timestamp|reviewer|"
    r"editor|metadata|meta|status|round|role|country|ip|url|link)(?:_|$)",
    re.IGNORECASE,
)
TEXT_LIKE_COMMENT_SUBKEY_RE = re.compile(
    r"(?:comment|comments|text|body|content|summary|major|minor|strength|"
    r"weakness|confidential|public|message|note|notes|remark|remarks|question|"
    r"answer|value|description|critique|review)",
    re.IGNORECASE,
)


def _one_review_schema() -> dict[str, Any]:
    return {
        "scores": {
            "factual_accuracy": 0,
            "methodological_insight": 0,
            "specificity": 0,
            "actionability": 0,
            "coverage_of_relevant_issues": 0,
            "constructiveness": 0,
        },
        "penalties": {
            "hallucinated_or_unsupported_claims": 0,
            "unnecessary_verbosity_or_repetition": 0,
        },
        "base_score": 0,
        "adjusted_overall_score": 0,
        "major_strengths": [
            {
                "point": "",
                "evidence_from_review": "",
                "evidence_from_manuscript": "",
            }
        ],
        "major_weaknesses": [
            {
                "point": "",
                "evidence_from_review": "",
                "evidence_from_manuscript": "",
            }
        ],
        "hallucinated_or_unsupported_points": [
            {
                "claim": "",
                "why_problematic": "",
                "evidence_from_review": "",
                "evidence_from_manuscript": "",
            }
        ],
        "short_rationale": "",
    }


OUTPUT_SCHEMA_TEMPLATE = {
    "reviews": {label: _one_review_schema() for label in REVIEW_LABELS},
    "final_ranking": [
        {
            "rank": 1,
            "review": "",
            "adjusted_overall_score": 0,
            "rationale": "",
        }
    ],
    "ties": [
        {
            "rank": 0,
            "reviews": [],
            "reason": "",
        }
    ],
    "overall_decision_rationale": "",
    "confidence": "low | medium | high",
    "manuscript_verification": "available",
}


class ReviewJudgeError(Exception):
    """Base class for user-facing errors."""


class MissingInputError(ReviewJudgeError):
    """Raised when an input path is missing."""


class InvalidJSONError(ReviewJudgeError):
    """Raised when JSON input is invalid."""


class CommentsExtractionError(ReviewJudgeError):
    """Raised for missing, empty, fewer, or more than four review inputs."""


class PDFExtractionError(ReviewJudgeError):
    """Raised when manuscript PDF/text extraction fails."""


class RedactionPatternError(ReviewJudgeError):
    """Raised when custom redaction patterns cannot be loaded or compiled."""


class PrivacyValidationError(ReviewJudgeError):
    """Raised when the final outbound payload violates privacy constraints."""


class GrokAPIError(ReviewJudgeError):
    """Raised when the Grok API call fails."""


class GrokResponseError(ReviewJudgeError):
    """Raised when Grok returns malformed or schema-invalid JSON."""


# ---------------------------------------------------------------------------
# Required modular functions
# ---------------------------------------------------------------------------
def load_json(path: str | Path) -> Any:
    """Load a JSON file from disk. The returned object must remain local."""
    json_path = Path(path)
    if not json_path.is_file():
        raise MissingInputError(f"JSON file not found: {json_path}")

    try:
        with json_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        raise InvalidJSONError(f"Invalid JSON in {json_path}: {exc}") from exc
    except OSError as exc:
        raise MissingInputError(f"Unable to read JSON {json_path}: {exc}") from exc


def extract_comments_only(data: Any) -> list[Any]:
    """Recursively extract only values of exact keys named "comments".

    Privacy enforcement point for human reviews:
    - This function ignores every key except the exact, case-sensitive key
      "comments".
    - Sibling fields such as reviewer_id, reviewer_name, dates, scores,
      recommendations, decisions, affiliations, and metadata are not returned.
    - The original JSON object is never returned or forwarded.
    """
    extracted: list[Any] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "comments":
                    extracted.append(value)
                # Continue recursion because human review files may contain
                # multiple reviews, each with its own exact comments field.
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(data)
    return extracted


def normalize_comment_value(value: Any) -> str:
    """Normalize a comment value into a clean string.

    Supported comment values:
    - string
    - list of strings
    - nested list
    - dictionary/object containing comment text

    If an exact "comments" value is an object, this function extracts only
    text-like subfields and skips sensitive metadata-looking subfields.
    """
    parts = list(_iter_comment_text(value))
    cleaned = [_clean_text_fragment(part) for part in parts]
    return "\n\n".join(part for part in cleaned if part)


def build_reviews_payload(comments: list[Any]) -> dict[str, str]:
    expected = len(REVIEW_LABELS)

    if len(comments) != expected:
        raise CommentsExtractionError(
            f"Expected exactly four source review values; found {len(comments)}."
        )

    normalized = [normalize_comment_value(value) for value in comments]
    empty_indices = [
        idx + 1 for idx, comment in enumerate(normalized) if not comment.strip()
    ]
    if empty_indices:
        raise CommentsExtractionError(
            f"Empty comments after normalization at source position(s): {empty_indices}."
        )

    return dict(zip(REVIEW_LABELS, normalized, strict=True))


def extract_pdf_text(path: str | Path) -> str:
    """Extract plain manuscript text locally from a mandatory manuscript path.

    Supports both actual PDFs and this archive's manuscript text artifacts named
    *.pdf.json. For *.pdf.json, only metadata.sections[*].text is extracted;
    artifact metadata such as title, authors, emails, source, and file name is
    deliberately ignored.

    Privacy enforcement point:
    - The raw PDF is not sent to Grok.
    - PDF metadata is not included in the outbound payload.
    - Only locally extracted manuscript text is used.
    """
    manuscript_path = Path(path)
    if not manuscript_path.is_file():
        raise MissingInputError(
            f"Mandatory manuscript file not found: {manuscript_path}"
        )

    if manuscript_path.name.endswith(".pdf.json"):
        return _extract_text_from_pdf_json_artifact(manuscript_path)

    if manuscript_path.suffix.lower() != ".pdf":
        raise PDFExtractionError(
            f"Unsupported manuscript file type: {manuscript_path}. Expected .pdf or .pdf.json."
        )

    try:
        reader = PdfReader(str(manuscript_path))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as exc:
                raise PDFExtractionError(
                    f"PDF is encrypted and cannot be decrypted: {manuscript_path}"
                ) from exc

        page_texts: list[str] = []
        for page_index, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as exc:
                raise PDFExtractionError(
                    f"Text extraction failed on PDF page {page_index}: {exc}"
                ) from exc
            if text.strip():
                page_texts.append(text)

        manuscript_text = _clean_text_fragment("\n\n".join(page_texts))
        if not manuscript_text:
            raise PDFExtractionError(
                f"No extractable text found in manuscript PDF: {manuscript_path}"
            )
        return manuscript_text
    except PdfReadError as exc:
        raise PDFExtractionError(
            f"Unreadable manuscript PDF {manuscript_path}: {exc}"
        ) from exc
    except OSError as exc:
        raise PDFExtractionError(
            f"Unable to read manuscript PDF {manuscript_path}: {exc}"
        ) from exc


def load_redaction_patterns(path: str | Path | None) -> list[str]:
    """Load custom redaction regex patterns.

    Supported formats:
    - JSON list: ["pattern1", "pattern2"]
    - JSON object: {"patterns": ["pattern1", "pattern2"]}
    - Plain text: one regex per line; blank lines and # comments are ignored.

    Default redactions are always applied first.
    """
    patterns = list(DEFAULT_REDACTION_PATTERNS)
    if path is None:
        return _validate_regex_patterns(patterns)

    pattern_path = Path(path)
    if not pattern_path.is_file():
        raise RedactionPatternError(
            f"Redaction patterns file not found: {pattern_path}"
        )

    try:
        raw = pattern_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RedactionPatternError(
            f"Unable to read redaction patterns file {pattern_path}: {exc}"
        ) from exc

    custom_patterns: list[str]
    if pattern_path.suffix.lower() == ".json":
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RedactionPatternError(
                f"Invalid JSON redaction pattern file: {exc}"
            ) from exc

        if isinstance(parsed, list):
            custom_patterns = parsed
        elif isinstance(parsed, dict) and isinstance(parsed.get("patterns"), list):
            custom_patterns = parsed["patterns"]
        else:
            raise RedactionPatternError(
                "JSON redaction pattern file must be a list or an object with a 'patterns' list."
            )
    else:
        custom_patterns = [
            line.strip()
            for line in raw.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]

    if not all(isinstance(pattern, str) for pattern in custom_patterns):
        raise RedactionPatternError("Every redaction pattern must be a string regex.")

    patterns.extend(custom_patterns)
    return _validate_regex_patterns(patterns)


def sanitize_text(text: str, redaction_patterns: list[str]) -> str:
    """Apply local redactions and whitespace cleanup before any Grok call."""
    sanitized = _clean_text_fragment(text)
    for pattern in redaction_patterns:
        sanitized = re.sub(pattern, "[REDACTED]", sanitized)
    return _clean_text_fragment(sanitized)


def build_outbound_payload(
    reviews_payload: dict[str, str], manuscript_text: str
) -> dict[str, Any]:
    """Build the exact data-bearing payload embedded in the Grok user message.

    No keys beyond "reviews" and "manuscript_text" are allowed.
    """
    payload = {
        "reviews": reviews_payload,
        "manuscript_text": manuscript_text,
    }
    validate_outbound_payload(payload)
    return payload


def validate_outbound_payload(payload: dict[str, Any]) -> None:
    """Validate privacy/data-minimization constraints immediately before API use."""
    if not isinstance(payload, dict):
        raise PrivacyValidationError("Outbound payload must be a dictionary.")

    actual_top_keys = set(payload.keys())
    if actual_top_keys != ALLOWED_TOP_LEVEL_KEYS:
        raise PrivacyValidationError(
            f"Outbound payload top-level keys must be exactly {sorted(ALLOWED_TOP_LEVEL_KEYS)}; "
            f"got {sorted(actual_top_keys)}."
        )

    reviews = payload.get("reviews")
    if not isinstance(reviews, dict):
        raise PrivacyValidationError('Outbound payload "reviews" must be an object.')

    actual_review_keys = set(reviews.keys())
    if actual_review_keys != REQUIRED_REVIEW_KEYS:
        raise PrivacyValidationError(
            f"Reviews must contain exactly keys {sorted(REQUIRED_REVIEW_KEYS)}; got {sorted(actual_review_keys)}."
        )

    for label in REVIEW_LABELS:
        value = reviews.get(label)
        if not isinstance(value, str) or not value.strip():
            raise PrivacyValidationError(f"Review {label} must be a non-empty string.")

    manuscript_text = payload.get("manuscript_text")
    if not isinstance(manuscript_text, str) or not manuscript_text.strip():
        raise PrivacyValidationError(
            'Outbound payload "manuscript_text" must be a non-empty string.'
        )


def build_grok_judge_prompt() -> str:
    """Build the Grok LLM-as-judge instruction prompt.

    The user message contains the only document/review data available to Grok.
    It must be a JSON object with only "reviews" and "manuscript_text".
    """
    output_schema = json.dumps(OUTPUT_SCHEMA_TEMPLATE, ensure_ascii=False, indent=2)
    labels_text = ", ".join(REVIEW_LABELS)
    review_phrase = (
        "one anonymized peer-review report"
        if len(REVIEW_LABELS) == 1
        else f"{len(REVIEW_LABELS)} anonymized peer-review reports"
    )

    return f"""
You are evaluating the quality of {review_phrase} for the same scientific manuscript.

The next user message will contain a JSON payload with exactly two top-level keys:
- reviews
- manuscript_text

Use only that payload. The review label(s) are: {labels_text}. The manuscript_text is sanitized plain text extracted locally from the reviewed manuscript PDF or from the local *.pdf.json manuscript text artifact. Do not ask for the raw PDF, PDF metadata, reviewer identities, scores, recommendations, dates, affiliations, or any other metadata.

Judging rules:
- Do not try to infer whether a review was written by a human or by an AI model.
- Judge each review only by how useful it would be to the manuscript authors and editors.
- Do not reward length, fluency, confidence, or politeness unless the review is also accurate, specific, and actionable.
- Do not reward generic comments that could apply to many papers.
- Use the manuscript text to check whether reviewer criticisms are supported, contradicted, or not verifiable.
- Penalize criticisms that are contradicted by the manuscript.
- Penalize claims about missing content if the manuscript actually contains that content.
- Penalize hallucinated or unsupported claims.
- Reward comments that accurately identify real weaknesses in the manuscript.
- Reward comments that are specific, methodological, actionable, and relevant.
- Allow ties when differences are small.
- Provide a confidence level: low, medium, or high.

Score each review from 0 to 5 on:
1. factual_accuracy
2. methodological_insight
3. specificity
4. actionability
5. coverage_of_relevant_issues
6. constructiveness

Assign penalties from 0 to 5 for:
1. hallucinated_or_unsupported_claims
2. unnecessary_verbosity_or_repetition

Penalty meaning:
- 0 = no penalty
- 1 = minor issue
- 2 = noticeable issue
- 3 = significant issue
- 4 = severe issue
- 5 = extremely severe issue

Use this formula:
base_score = average of the six quality scores
penalty_deduction = 0.35 * hallucinated_or_unsupported_claims + 0.20 * unnecessary_verbosity_or_repetition
adjusted_overall_score = max(0, min(5, base_score - penalty_deduction))

Round all numeric scores to two decimals.

Return only valid JSON. No markdown. No explanations outside the JSON. Use exactly this output schema:
{output_schema}
""".strip()


def call_grok_judge(payload: dict[str, Any]) -> str:
    """Call Grok through a small adapter function.

    Change this adapter to switch endpoint, SDK, or model. Extraction,
    sanitization, and privacy validation remain independent.
    """
    validate_outbound_payload(payload)

    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        raise GrokAPIError("Missing XAI_API_KEY environment variable.")

    base_url = os.getenv("XAI_BASE_URL", DEFAULT_XAI_BASE_URL).rstrip("/")
    model = os.getenv("XAI_MODEL", DEFAULT_XAI_MODEL)
    timeout_seconds = float(os.getenv("XAI_TIMEOUT_SECONDS", "3600"))
    endpoint = f"{base_url}/responses"

    request_body = {
        "model": model,
        "input": [
            {"role": "system", "content": build_grok_judge_prompt()},
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False),
            },
        ],
        "text": {"format": {"type": "json_object"}},
        "temperature": 0,
        "store": False,
    }

    try:
        response = requests.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=request_body,
            timeout=timeout_seconds,
        )
    except requests.RequestException as exc:
        raise GrokAPIError(f"Grok API request failed: {exc}") from exc

    if response.status_code >= 400:
        safe_body = response.text[:2000]
        raise GrokAPIError(
            f"Grok API returned HTTP {response.status_code}: {safe_body}"
        )

    try:
        response_json = response.json()
    except ValueError as exc:
        raise GrokResponseError("Grok API response was not JSON.") from exc

    return _extract_text_from_xai_response(response_json)


def parse_grok_response(response_text: str) -> dict[str, Any]:
    """Parse Grok response text as JSON and validate expected high-level shape."""
    parsed = _strict_or_fenced_json_loads(response_text)

    if not isinstance(parsed, dict):
        raise GrokResponseError("Grok judgment report must be a JSON object.")

    reviews = parsed.get("reviews")
    if not isinstance(reviews, dict) or set(reviews.keys()) != REQUIRED_REVIEW_KEYS:
        raise GrokResponseError(
            f"Grok response must contain reviews {sorted(REQUIRED_REVIEW_KEYS)}."
        )

    for label, review in reviews.items():
        if not isinstance(review, dict):
            raise GrokResponseError(f"Grok response review {label} must be an object.")
        scores = review.get("scores")
        penalties = review.get("penalties")
        if not isinstance(scores, dict) or set(scores.keys()) != QUALITY_SCORE_KEYS:
            raise GrokResponseError(
                f"Grok response review {label} has invalid score keys."
            )
        if not isinstance(penalties, dict) or set(penalties.keys()) != PENALTY_KEYS:
            raise GrokResponseError(
                f"Grok response review {label} has invalid penalty keys."
            )
        for score_name, value in {**scores, **penalties}.items():
            if not _is_number_in_range(value, 0, 5):
                raise GrokResponseError(
                    f"Grok response review {label} field {score_name} must be numeric in [0, 5]."
                )
        for score_name in ["base_score", "adjusted_overall_score"]:
            if not _is_number_in_range(review.get(score_name), 0, 5):
                raise GrokResponseError(
                    f"Grok response review {label} field {score_name} must be numeric in [0, 5]."
                )

    if not isinstance(parsed.get("final_ranking"), list):
        raise GrokResponseError("Grok response final_ranking must be a list.")
    if not isinstance(parsed.get("ties"), list):
        raise GrokResponseError("Grok response ties must be a list.")
    if parsed.get("manuscript_verification") != "available":
        raise GrokResponseError(
            'Grok response must set manuscript_verification to "available".'
        )
    if parsed.get("confidence") not in {"low", "medium", "high"}:
        raise GrokResponseError(
            'Grok response confidence must be "low", "medium", or "high".'
        )

    return parsed


def save_report(report: dict[str, Any], output_path: str | Path) -> None:
    """Save a JSON report to disk."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.write("\n")
    except OSError as exc:
        raise ReviewJudgeError(f"Unable to save report to {out_path}: {exc}") from exc


# ---------------------------------------------------------------------------
# Batch discovery and orchestration functions
# ---------------------------------------------------------------------------
def main() -> int:
    if load_dotenv is not None:
        load_dotenv(PROJECT_ROOT / ".env")

    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL), format="%(levelname)s: %(message)s"
    )

    summary: dict[str, Any] = {
        "project_root": str(PROJECT_ROOT),
        "data_dir": str(DATA_DIR),
        "pdfs_dir": str(PDFS_DIR),
        "human_review_dir": str(HUMAN_REVIEW_DIR),
        "generated_review_dirs": {
            source: str(path) for source, path in GENERATED_REVIEW_DIRS.items()
        },
        "max_manuscripts": MAX_MANUSCRIPTS,
        "dry_run": DRY_RUN,
        "skip_existing_judgments": SKIP_EXISTING_JUDGMENTS,
        "label_to_source_local_only": LABEL_TO_SOURCE,
        "processed": [],
        "skipped_existing": [],
        "skipped_missing_or_invalid_inputs": [],
        "failed": [],
    }

    selected_manuscript_ids: list[str] = []

    try:
        logging.warning(
            "Privacy notice: only human exact 'comments' fields, generated explicit "
            "human-facing comments fields, and sanitized extracted manuscript text are sent "
            "to Grok. Full review JSON, raw PDFs, PDF metadata, reviewer identifiers, "
            "scores, dates, affiliations, recommendations, and editor metadata are not sent."
        )

        if not DATA_DIR.is_dir():
            raise MissingInputError(f"Hardcoded data directory not found: {DATA_DIR}")
        if not PDFS_DIR.is_dir():
            raise MissingInputError(
                f"Hardcoded manuscripts directory not found: {PDFS_DIR}"
            )
        if not HUMAN_REVIEW_DIR.is_dir():
            raise MissingInputError(
                f"Hardcoded human review directory not found: {HUMAN_REVIEW_DIR}"
            )
        for source, path in GENERATED_REVIEW_DIRS.items():
            if not path.is_dir():
                raise MissingInputError(
                    f"Hardcoded generated review directory for {source} not found: {path}"
                )

        redaction_path = (
            REDACTION_PATTERNS_PATH if REDACTION_PATTERNS_PATH.is_file() else None
        )
        redaction_patterns = load_redaction_patterns(redaction_path)
        if redaction_path is None:
            logging.info(
                "No custom redaction file found at %s; using default redactions only.",
                REDACTION_PATTERNS_PATH,
            )
        else:
            logging.info("Loaded redaction patterns from %s.", redaction_path)

        manuscripts = discover_manuscripts(PDFS_DIR)
        if not manuscripts:
            raise MissingInputError(
                f"No manuscripts found in {PDFS_DIR}; expected *.pdf or *.pdf.json."
            )

        selected = list(manuscripts.items())[:MAX_MANUSCRIPTS]
        selected_manuscript_ids = [manuscript_id for manuscript_id, _path in selected]
        logging.info(
            "Discovered %d manuscript artifact(s); scanning first %d.",
            len(manuscripts),
            len(selected),
        )

        for manuscript_id, manuscript_path in selected:
            judgment_path = judgment_output_path(manuscript_id)
            if SKIP_EXISTING_JUDGMENTS and judgment_path.is_file():
                logging.info(
                    "Skipping manuscript %s because judgment already exists: %s",
                    manuscript_id,
                    judgment_path,
                )
                summary["skipped_existing"].append(
                    {
                        "manuscript_id": manuscript_id,
                        "manuscript_path": str(manuscript_path),
                        "judgment_path": str(judgment_path),
                        "status": "skipped_existing_judgment",
                    }
                )
                continue

            logging.info(
                "Preparing manuscript %s from %s", manuscript_id, manuscript_path
            )
            try:
                process_one_manuscript(
                    manuscript_id=manuscript_id,
                    manuscript_path=manuscript_path,
                    redaction_patterns=redaction_patterns,
                    summary=summary,
                )
            except (
                CommentsExtractionError,
                MissingInputError,
                InvalidJSONError,
                PDFExtractionError,
            ) as exc:
                record = {
                    "manuscript_id": manuscript_id,
                    "manuscript_path": str(manuscript_path),
                    "reason": str(exc),
                    "status": "skipped_missing_or_invalid_inputs",
                }
                summary["skipped_missing_or_invalid_inputs"].append(record)
                logging.warning("Skipping manuscript %s: %s", manuscript_id, exc)
                if STOP_ON_FIRST_ERROR:
                    break
            except ReviewJudgeError as exc:
                record = {
                    "manuscript_id": manuscript_id,
                    "manuscript_path": str(manuscript_path),
                    "reason": str(exc),
                    "status": "failed",
                }
                summary["failed"].append(record)
                logging.error("Failed manuscript %s: %s", manuscript_id, exc)
                if STOP_ON_FIRST_ERROR:
                    break

        final_report = build_aggregate_report(selected_manuscript_ids, summary)
        save_report(final_report, FINAL_REPORT_OUTPUT_PATH)
        save_report(summary, BATCH_SUMMARY_OUTPUT_PATH)
        logging.info("Saved final aggregate report to %s", FINAL_REPORT_OUTPUT_PATH)
        logging.info("Saved batch summary to %s", BATCH_SUMMARY_OUTPUT_PATH)

        logging.info(
            "Finished. New processed=%d, skipped_existing=%d, skipped_inputs=%d, failed=%d, judgments_available_for_report=%d.",
            len(summary["processed"]),
            len(summary["skipped_existing"]),
            len(summary["skipped_missing_or_invalid_inputs"]),
            len(summary["failed"]),
            final_report.get("judgments_found", 0),
        )
        return 0 if not summary["failed"] else 1

    except ReviewJudgeError as exc:
        summary["failed"].append({"reason": str(exc), "status": "fatal"})
        try:
            final_report = build_aggregate_report(selected_manuscript_ids, summary)
            save_report(final_report, FINAL_REPORT_OUTPUT_PATH)
            save_report(summary, BATCH_SUMMARY_OUTPUT_PATH)
        except Exception:
            pass
        logging.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        logging.error("Interrupted.")
        return 130


def process_one_manuscript(
    manuscript_id: str,
    manuscript_path: Path,
    redaction_patterns: list[str],
    summary: dict[str, Any],
) -> None:
    raw_review_values, source_counts, source_files = (
        collect_review_values_for_manuscript(manuscript_id)
    )

    # Review JSON stays local; only permitted comment values are normalized and sanitized.
    reviews_payload = build_reviews_payload(
        [raw_review_values[source] for source in SOURCE_ORDER]
    )
    reviews_payload = {
        label: sanitize_text(comment, redaction_patterns)
        for label, comment in reviews_payload.items()
    }

    # Manuscript file stays local; only sanitized extracted text is used.
    manuscript_text = extract_pdf_text(manuscript_path)
    sanitized_manuscript_text = sanitize_text(manuscript_text, redaction_patterns)

    payload = build_outbound_payload(reviews_payload, sanitized_manuscript_text)
    validate_outbound_payload(payload)

    payload_path = PAYLOAD_OUTPUT_DIR / f"{manuscript_id}.sanitized_payload.json"
    save_report(payload, payload_path)

    # Required exact sanitized outbound payload printing before the API call.
    print(
        f"\n=== Sanitized outbound payload to Grok for manuscript {manuscript_id} ==="
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"=== End sanitized outbound payload for manuscript {manuscript_id} ===\n")

    if DRY_RUN:
        logging.info(
            "Dry run complete for manuscript %s. Grok was not called.", manuscript_id
        )
        summary["processed"].append(
            {
                "manuscript_id": manuscript_id,
                "manuscript_path": str(manuscript_path),
                "review_files_used": {
                    source: str(path) for source, path in source_files.items()
                },
                "source_comment_counts": source_counts,
                "payload_path": str(payload_path),
                "status": "dry_run_validated",
            }
        )
        return

    response_text = call_grok_judge(payload)
    report = parse_grok_response(response_text)
    output_path = judgment_output_path(manuscript_id)
    save_report(report, output_path)

    summary["processed"].append(
        {
            "manuscript_id": manuscript_id,
            "manuscript_path": str(manuscript_path),
            "review_files_used": {
                source: str(path) for source, path in source_files.items()
            },
            "source_comment_counts": source_counts,
            "payload_path": str(payload_path),
            "judgment_path": str(output_path),
            "status": "evaluated",
        }
    )
    logging.info(
        "Saved Grok judgment report for manuscript %s to %s", manuscript_id, output_path
    )


def discover_manuscripts(pdfs_dir: str | Path) -> dict[str, Path]:
    """Return manuscript_id -> manuscript path, preferring raw .pdf over .pdf.json."""
    root = Path(pdfs_dir)
    candidates = [
        p
        for p in root.iterdir()
        if p.is_file() and (p.suffix.lower() == ".pdf" or p.name.endswith(".pdf.json"))
    ]
    by_id: dict[str, Path] = {}

    for path in sorted(candidates, key=lambda p: _natural_sort_key(p.name)):
        manuscript_id = _manuscript_id_from_path(path)
        existing = by_id.get(manuscript_id)
        if existing is None:
            by_id[manuscript_id] = path
        elif existing.name.endswith(".pdf.json") and path.suffix.lower() == ".pdf":
            by_id[manuscript_id] = path

    return dict(sorted(by_id.items(), key=lambda item: _natural_sort_key(item[0])))


def collect_review_values_for_manuscript(
    manuscript_id: str,
) -> tuple[dict[str, Any], dict[str, int], dict[str, Path]]:
    raw_values: dict[str, Any] = {}
    source_counts: dict[str, int] = {}
    source_files: dict[str, Path] = {}

    if "human" in SOURCE_ORDER:
        human_file = HUMAN_REVIEW_DIR / f"{manuscript_id}.json"
        if not human_file.is_file():
            raise MissingInputError(
                f"Missing exact human review file for manuscript {manuscript_id}: {human_file}"
            )

        human_data = load_json(human_file)
        human_comments = extract_comments_only(human_data)

        if not human_comments:
            raise CommentsExtractionError(
                f'Manuscript {manuscript_id}: no exact "comments" fields found in {human_file}.'
            )

        if not normalize_comment_value(human_comments).strip():
            raise CommentsExtractionError(
                f"Manuscript {manuscript_id}: human comments are empty after normalization."
            )

        raw_values["human"] = human_comments
        source_counts["human"] = len(human_comments)
        source_files["human"] = human_file

    for source in SOURCE_ORDER:
        if source == "human":
            continue

        review_dir = GENERATED_REVIEW_DIRS[source]
        generated_file = review_dir / f"{manuscript_id}.review.json"

        if not generated_file.is_file():
            raise MissingInputError(
                f"Missing exact generated review file for source {source}, manuscript {manuscript_id}: {generated_file}"
            )

        generated_data = load_json(generated_file)
        generated_comment = extract_generated_human_facing_comment(
            generated_data, generated_file
        )

        if not normalize_comment_value(generated_comment).strip():
            raise CommentsExtractionError(
                f"Manuscript {manuscript_id}: generated comments for {source} are empty after normalization."
            )

        raw_values[source] = generated_comment
        source_counts[source] = 1
        source_files[source] = generated_file

    missing_sources = [source for source in SOURCE_ORDER if source not in raw_values]
    if missing_sources:
        raise CommentsExtractionError(
            f"Manuscript {manuscript_id}: missing review values for {missing_sources}."
        )

    return raw_values, source_counts, source_files


def extract_generated_human_facing_comment(data: Any, review_file: Path) -> Any:
    """Extract only an explicit human-facing generated comments path.

    This intentionally does not recursively search generated review JSONs.
    """
    matches: list[tuple[tuple[str, ...], Any]] = []
    for path in GENERATED_REVIEW_COMMENT_PATHS:
        found, value = _get_nested_value(data, path)
        if found and normalize_comment_value(value).strip():
            matches.append((path, value))

    if not matches:
        allowed = [".".join(path) for path in GENERATED_REVIEW_COMMENT_PATHS]
        raise CommentsExtractionError(
            f"Generated review file {review_file} does not contain a non-empty allowed comments path. "
            f"Allowed paths: {allowed}."
        )

    if len(matches) > 1:
        allowed_found = [".".join(path) for path, _value in matches]
        raise CommentsExtractionError(
            f"Generated review file {review_file} contains multiple non-empty allowed comments paths: "
            f"{allowed_found}. Refusing to choose implicitly."
        )

    return matches[0][1]


def build_aggregate_report(
    manuscript_ids: list[str], batch_summary: dict[str, Any]
) -> dict[str, Any]:
    """Create final local report with best-review counts by source.

    This report maps anonymized Grok labels back to local sources. That mapping is
    never sent to Grok and is only used after each judgment file exists locally.
    """
    counts = {
        source: {
            "best_or_tied_count": 0,
            "sole_best_count": 0,
            "tied_best_count": 0,
        }
        for source in SOURCE_ORDER
    }
    manuscript_results: list[dict[str, Any]] = []
    parse_failures: list[dict[str, str]] = []

    for manuscript_id in manuscript_ids:
        path = judgment_output_path(manuscript_id)
        if not path.is_file():
            continue
        try:
            report = load_json(path)
            winners = determine_winner_labels(report)
        except ReviewJudgeError as exc:
            parse_failures.append(
                {
                    "manuscript_id": manuscript_id,
                    "judgment_path": str(path),
                    "reason": str(exc),
                }
            )
            continue

        winner_sources = [
            LABEL_TO_SOURCE[label] for label in winners if label in LABEL_TO_SOURCE
        ]
        if not winner_sources:
            parse_failures.append(
                {
                    "manuscript_id": manuscript_id,
                    "judgment_path": str(path),
                    "reason": f"No winner labels could be mapped from {winners!r}.",
                }
            )
            continue

        best_type = "tie" if len(winner_sources) > 1 else "sole"
        for source in winner_sources:
            counts[source]["best_or_tied_count"] += 1
            if best_type == "sole":
                counts[source]["sole_best_count"] += 1
            else:
                counts[source]["tied_best_count"] += 1

        manuscript_results.append(
            {
                "manuscript_id": manuscript_id,
                "judgment_path": str(path),
                "winner_labels": winners,
                "winner_sources": winner_sources,
                "best_type": best_type,
            }
        )

    return {
        "max_manuscripts": MAX_MANUSCRIPTS,
        "manuscripts_considered": len(manuscript_ids),
        "judgments_found": len(manuscript_results),
        "label_to_source_local_only": LABEL_TO_SOURCE,
        "best_counts": counts,
        "manuscript_results": manuscript_results,
        "judgment_parse_failures": parse_failures,
        "batch_summary_counts": {
            "newly_processed": len(batch_summary.get("processed", [])),
            "skipped_existing": len(batch_summary.get("skipped_existing", [])),
            "skipped_missing_or_invalid_inputs": len(
                batch_summary.get("skipped_missing_or_invalid_inputs", [])
            ),
            "failed": len(batch_summary.get("failed", [])),
        },
        "outputs": {
            "final_report": str(FINAL_REPORT_OUTPUT_PATH),
            "batch_summary": str(BATCH_SUMMARY_OUTPUT_PATH),
            "judgments_folder": str(OUTPUT_DIR),
            "sanitized_payloads_folder": str(PAYLOAD_OUTPUT_DIR),
        },
    }


def determine_winner_labels(report: dict[str, Any]) -> list[str]:
    """Determine rank-1 labels from a Grok judgment report."""
    labels: list[str] = []

    # Prefer explicit rank-1 rows.
    final_ranking = report.get("final_ranking")
    if isinstance(final_ranking, list):
        for item in final_ranking:
            if not isinstance(item, dict):
                continue
            rank = item.get("rank")
            if _safe_int(rank) == 1:
                labels.extend(_extract_review_labels(item.get("review")))

    # Also account for a rank-1 tie section if Grok used it.
    ties = report.get("ties")
    if isinstance(ties, list):
        for item in ties:
            if not isinstance(item, dict):
                continue
            if _safe_int(item.get("rank")) == 1:
                labels.extend(_extract_review_labels(item.get("reviews")))

    labels = _dedupe_keep_order(
        [label.upper() for label in labels if label.upper() in REQUIRED_REVIEW_KEYS]
    )
    if labels:
        return labels

    # Fallback: use adjusted_overall_score if final_ranking/ties are not usable.
    reviews = report.get("reviews")
    if not isinstance(reviews, dict):
        raise GrokResponseError(
            "Judgment report has no reviews object for fallback winner detection."
        )

    scores: dict[str, float] = {}
    for label in REVIEW_LABELS:
        review = reviews.get(label)
        if isinstance(review, dict) and _is_number_in_range(
            review.get("adjusted_overall_score"), 0, 5
        ):
            scores[label] = float(review["adjusted_overall_score"])

    if len(scores) != len(REVIEW_LABELS):
        raise GrokResponseError(
            "Could not determine winners from final_ranking or adjusted_overall_score."
        )

    max_score = max(scores.values())
    return [label for label, score in scores.items() if score == max_score]


def judgment_output_path(manuscript_id: str) -> Path:
    return OUTPUT_DIR / f"{manuscript_id}.judge.json"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _extract_text_from_pdf_json_artifact(path: Path) -> str:
    """Extract only local manuscript section text from a *.pdf.json artifact."""
    try:
        artifact = load_json(path)
    except InvalidJSONError as exc:
        raise PDFExtractionError(
            f"Invalid manuscript text artifact {path}: {exc}"
        ) from exc
    except MissingInputError as exc:
        raise PDFExtractionError(str(exc)) from exc

    sections = None
    if isinstance(artifact, dict):
        metadata = artifact.get("metadata")
        if isinstance(metadata, dict):
            sections = metadata.get("sections")

    if not isinstance(sections, list):
        raise PDFExtractionError(
            f"Manuscript artifact {path} does not contain metadata.sections as a list."
        )

    section_texts: list[str] = []
    for section in sections:
        if not isinstance(section, dict):
            continue
        text = section.get("text")
        if isinstance(text, str) and text.strip():
            section_texts.append(text)

    manuscript_text = _clean_text_fragment("\n\n".join(section_texts))
    if not manuscript_text:
        raise PDFExtractionError(
            f"No extractable manuscript text found in artifact: {path}"
        )
    return manuscript_text


def _iter_comment_text(value: Any) -> Iterable[str]:
    if value is None:
        return

    if isinstance(value, str):
        yield value
        return

    if isinstance(value, (int, float, bool)):
        # Numeric and boolean values are not review prose and may be scores/flags.
        return

    if isinstance(value, list):
        for item in value:
            yield from _iter_comment_text(item)
        return

    if isinstance(value, dict):
        text_like_parts: list[str] = []
        fallback_parts: list[str] = []

        for key, child in value.items():
            key_text = str(key)
            if SENSITIVE_COMMENT_SUBKEY_RE.search(key_text):
                continue

            child_parts = list(_iter_comment_text(child))
            if not child_parts:
                continue

            if TEXT_LIKE_COMMENT_SUBKEY_RE.search(key_text):
                text_like_parts.extend(child_parts)
            else:
                # Fallback supports simple objects like {"public": "..."} while
                # still avoiding short labels that are likely metadata.
                fallback_parts.extend(
                    part
                    for part in child_parts
                    if isinstance(part, str) and len(part.strip()) >= 20
                )

        yield from (text_like_parts or fallback_parts)
        return


def _clean_text_fragment(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = html.unescape(text)
    text = text.replace("\x00", " ")
    # Remove common HTML/XML tags from comment systems while preserving prose.
    text = re.sub(r"<[^>]{1,200}>", " ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\t\f\v ]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _validate_regex_patterns(patterns: list[str]) -> list[str]:
    for pattern in patterns:
        try:
            re.compile(pattern)
        except re.error as exc:
            raise RedactionPatternError(
                f"Invalid redaction regex {pattern!r}: {exc}"
            ) from exc
    return patterns


def _extract_text_from_xai_response(response_json: dict[str, Any]) -> str:
    """Extract text from common xAI/OpenAI-compatible response shapes."""
    if (
        isinstance(response_json.get("output_text"), str)
        and response_json["output_text"].strip()
    ):
        return response_json["output_text"]

    # Responses API shape: output[*].content[*].text or output_text.
    output = response_json.get("output")
    if isinstance(output, list):
        chunks: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, list):
                for content_item in content:
                    if not isinstance(content_item, dict):
                        continue
                    for key in ("text", "output_text"):
                        value = content_item.get(key)
                        if isinstance(value, str):
                            chunks.append(value)
            elif isinstance(content, str):
                chunks.append(content)
        if chunks:
            return "".join(chunks)

    # Chat Completions fallback shape.
    choices = response_json.get("choices")
    if isinstance(choices, list) and choices:
        first_choice = choices[0] if isinstance(choices[0], dict) else {}
        message = first_choice.get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"]

    raise GrokResponseError(
        "Could not find generated text in Grok API response. Response keys: "
        f"{sorted(response_json.keys())}"
    )


def _strict_or_fenced_json_loads(response_text: str) -> dict[str, Any]:
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        # Helpful fallback for APIs that unexpectedly wrap JSON in code fences.
        stripped = response_text.strip()
        fenced = re.fullmatch(
            r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE
        )
        if fenced:
            try:
                return json.loads(fenced.group(1))
            except json.JSONDecodeError as exc:
                raise GrokResponseError(f"Malformed Grok JSON response: {exc}") from exc
        raise GrokResponseError(
            "Malformed Grok JSON response; expected valid JSON only."
        )


def _is_number_in_range(value: Any, low: float, high: float) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and low <= float(value) <= high
    )


def _manuscript_id_from_path(path: Path) -> str:
    name = path.name
    if name.endswith(".pdf.json"):
        return name[: -len(".pdf.json")]
    if name.endswith(".pdf"):
        return name[: -len(".pdf")]
    return path.stem


def _natural_sort_key(value: str) -> list[Any]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    ]


def _get_nested_value(data: Any, path: tuple[str, ...]) -> tuple[bool, Any]:
    node = data
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return False, None
        node = node[key]
    return True, node


def _extract_review_labels(value: Any) -> list[str]:
    if isinstance(value, str):
        stripped = value.strip()
        upper = stripped.upper()

        if upper in REQUIRED_REVIEW_KEYS:
            return [upper]

        labels_pattern = "|".join(
            re.escape(label)
            for label in sorted(REQUIRED_REVIEW_KEYS, key=len, reverse=True)
        )

        return [
            match.group(1).upper()
            for match in re.finditer(
                rf"\b(?:Review\s*)?({labels_pattern})\b",
                stripped,
                flags=re.IGNORECASE,
            )
        ]

    if isinstance(value, list):
        labels: list[str] = []
        for item in value:
            labels.extend(_extract_review_labels(item))
        return labels

    return []


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        value = value.upper()
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


if __name__ == "__main__":
    raise SystemExit(main())
