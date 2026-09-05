import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from llm_cost_reporting import CostPerformanceReporter, instrument_provider
from orchestrator import run_prompt_orchestration_review_flow
from providers.claude_provider import ClaudeProvider
from schemas import ParsedManuscript

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

PDFS_DIR = ROOT / "data" / "pdfs"
OUTPUT_DIR = ROOT / "data" / "claude-reviews"
SAMPLE_MANIFEST = ROOT / "data" / "evaluation-sample-100.json"

MODEL_NAME = "claude-haiku-4-5"
OUTPUT_SUFFIX = ".review.json"

EXPECTED_ORIGINAL = 0
EXPECTED_ADDITIONAL = 100
EXPECTED_TOTAL = 100


def manuscript_id_from_path(path: Path) -> str:
    return path.name.removesuffix(".pdf.json")


def make_run_report_path(output_dir: Path, model_name: str) -> Path:
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_model_name = re.sub(r"[^a-zA-Z0-9_.-]+", "-", model_name)

    return (
        reports_dir
        / f"{safe_model_name}-{run_id}.cost-performance.report.json"
    )


def load_sample_manifest() -> dict:
    if not SAMPLE_MANIFEST.exists():
        raise FileNotFoundError(
            f"Selection manifest not found: {SAMPLE_MANIFEST}"
        )

    manifest = json.loads(SAMPLE_MANIFEST.read_text(encoding="utf-8"))

    original_ids = [
        str(value) for value in manifest.get("original_ids", [])
    ]
    additional_ids = [
        str(value) for value in manifest.get("additional_ids", [])
    ]
    selected_ids = [
        str(value) for value in manifest.get("selected_ids", [])
    ]

    if len(original_ids) != EXPECTED_ORIGINAL:
        raise ValueError(
            f"Expected {EXPECTED_ORIGINAL} original IDs, "
            f"found {len(original_ids)}."
        )

    if len(additional_ids) != EXPECTED_ADDITIONAL:
        raise ValueError(
            f"Expected {EXPECTED_ADDITIONAL} additional IDs, "
            f"found {len(additional_ids)}."
        )

    if len(selected_ids) != EXPECTED_TOTAL:
        raise ValueError(
            f"Expected {EXPECTED_TOTAL} total IDs, "
            f"found {len(selected_ids)}."
        )

    if len(set(selected_ids)) != EXPECTED_TOTAL:
        raise ValueError("The selected manuscript IDs contain duplicates.")

    if set(original_ids).intersection(additional_ids):
        raise ValueError("The original and additional cohorts overlap.")

    if set(selected_ids) != set(original_ids).union(additional_ids):
        raise ValueError(
            "selected_ids must equal original_ids plus additional_ids."
        )

    return manifest


def build_manuscript_index() -> dict[str, Path]:
    manuscript_index: dict[str, Path] = {}

    for path in PDFS_DIR.glob("*.pdf.json"):
        manuscript_id = manuscript_id_from_path(path)

        if manuscript_id in manuscript_index:
            raise ValueError(f"Duplicate manuscript ID: {manuscript_id}")

        manuscript_index[manuscript_id] = path

    return manuscript_index


def validate_original_outputs(original_ids: list[str]) -> None:
    missing_outputs = []

    for manuscript_id in original_ids:
        output_path = OUTPUT_DIR / f"{manuscript_id}{OUTPUT_SUFFIX}"

        if not output_path.exists():
            missing_outputs.append(manuscript_id)

    if missing_outputs:
        raise FileNotFoundError(
            "Original Claude reviews are missing for IDs: "
            + ", ".join(missing_outputs)
        )


def main() -> None:
    api_key = os.getenv("CLAUDE_API_KEY")

    if not api_key:
        raise ValueError("CLAUDE_API_KEY is missing")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_sample_manifest()

    original_ids = [
        str(value) for value in manifest["original_ids"]
    ]
    additional_ids = [
        str(value) for value in manifest["additional_ids"]
    ]

    manuscript_index = build_manuscript_index()

    missing_pdf_ids = [
        str(manuscript_id)
        for manuscript_id in manifest["selected_ids"]
        if str(manuscript_id) not in manuscript_index
    ]

    if missing_pdf_ids:
        raise FileNotFoundError(
            "Selected manuscript files were not found for IDs: "
            + ", ".join(missing_pdf_ids)
        )

    # Preserve the original cohort by requiring its existing outputs.
    validate_original_outputs(original_ids)

    # Only manuscripts in the additional cohort are eligible for generation.
    manuscript_paths = [
        manuscript_index[manuscript_id]
        for manuscript_id in additional_ids
    ]

    pending_paths = []

    for manuscript_path in manuscript_paths:
        manuscript_id = manuscript_id_from_path(manuscript_path)
        output_path = OUTPUT_DIR / f"{manuscript_id}{OUTPUT_SUFFIX}"

        if output_path.exists():
            print(
                f"Skipping {manuscript_path.name} -> "
                f"review already exists: {output_path.name}"
            )
        else:
            pending_paths.append(manuscript_path)

    print(f"Sampling seed: {manifest.get('seed')}")
    print(f"Original Claude reviews retained: {len(original_ids)}")
    print(f"Additional manuscripts selected: {len(additional_ids)}")
    print(
        "Additional reviews already generated: "
        f"{len(manuscript_paths) - len(pending_paths)}"
    )
    print(f"Additional reviews still pending: {len(pending_paths)}")

    if not pending_paths:
        print("No additional Claude reviews need to be generated.")
        return

    provider = ClaudeProvider(api_key=api_key, model=MODEL_NAME)
    reporter = CostPerformanceReporter(
        model_name=MODEL_NAME,
        provider_name=provider.provider_name,
    )
    provider = instrument_provider(provider, reporter)

    try:
        for manuscript_path in pending_paths:
            manuscript_id = manuscript_id_from_path(manuscript_path)
            output_path = OUTPUT_DIR / f"{manuscript_id}{OUTPUT_SUFFIX}"

            print(f"Processing {manuscript_path.name}...")

            try:
                manuscript_data = json.loads(
                    manuscript_path.read_text(encoding="utf-8")
                )
                manuscript = ParsedManuscript.model_validate(manuscript_data)

                with reporter.workflow_run(
                    workflow_id=manuscript_id,
                    metadata={
                        "manuscript_file": manuscript_path.name,
                        "output_file": output_path.name,
                        "sample_cohort": "additional_50",
                        "sample_seed": manifest.get("seed"),
                        "sampling_method": manifest.get("method"),
                        "literature_collection": os.getenv(
                            "QDRANT_Q1Q2_COLLECTION",
                            "q1q2_openalex_pdf_articles",
                        ),
                        "guidelines_collection": os.getenv(
                            "QDRANT_GUIDELINES_COLLECTION",
                            "guideline_chunks",
                        ),
                    },
                ):
                    result = run_prompt_orchestration_review_flow(
                        provider,
                        manuscript,
                        include_audit=True,
                        llm_reporter=reporter,
                    )

                output_path.write_text(
                    json.dumps(result, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

                print(f"Saved Claude review to: {output_path}")

            except Exception as exc:
                print(
                    f"Failed for {manuscript_path.name}: "
                    f"{type(exc).__name__}: {exc}"
                )

    finally:
        report_path = make_run_report_path(OUTPUT_DIR, MODEL_NAME)
        reporter.write_report(report_path)
        print(f"Saved Claude cost/performance report to: {report_path}")

    print("Done.")


if __name__ == "__main__":
    main()
