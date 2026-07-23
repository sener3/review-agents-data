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

ROOT = Path(__file__).resolve().parents[0]
load_dotenv(ROOT / ".env")

PDFS_DIR = ROOT / "data" / "pdfs"
OUTPUT_DIR = ROOT / "data" / "claude-reviews"
MAX_FILES = 50
MODEL_NAME = "claude-haiku-4-5"
OUTPUT_SUFFIX = ".review.json"


def manuscript_sort_key(path: Path):
    match = re.match(r"(\d+)\.pdf\.json$", path.name)
    if match:
        return (0, int(match.group(1)))
    return (1, path.name.lower())


def manuscript_id_from_path(path: Path) -> str:
    return path.stem.replace(".pdf", "")


def make_run_report_path(output_dir: Path, model_name: str) -> Path:
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_model_name = re.sub(r"[^a-zA-Z0-9_.-]+", "-", model_name)

    return reports_dir / f"{safe_model_name}-{run_id}.cost-performance.report.json"


def main() -> None:
    api_key = os.getenv("CLAUDE_API_KEY")
    if api_key is None:
        raise ValueError("CLAUDE_API_KEY is missing")

    provider = ClaudeProvider(api_key=api_key, model=MODEL_NAME)
    reporter = CostPerformanceReporter(
        model_name=MODEL_NAME,
        provider_name=provider.provider_name,
    )
    provider = instrument_provider(provider, reporter)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manuscript_paths = sorted(PDFS_DIR.glob("*.json"), key=manuscript_sort_key)[
        :MAX_FILES
    ]

    if not manuscript_paths:
        print(f"No manuscript JSON files found in: {PDFS_DIR}")
        return

    print(f"Found {len(manuscript_paths)} manuscript(s) to consider.")

    for manuscript_path in manuscript_paths:
        manuscript_id = manuscript_id_from_path(manuscript_path)
        output_path = OUTPUT_DIR / f"{manuscript_id}{OUTPUT_SUFFIX}"

        if output_path.exists():
            print(
                f"Skipping {manuscript_path.name} -> review already exists: {output_path.name}"
            )
            continue

        print(f"Processing {manuscript_path.name}...")

        try:
            manuscript_data = json.loads(manuscript_path.read_text(encoding="utf-8"))
            manuscript = ParsedManuscript.model_validate(manuscript_data)

            with reporter.workflow_run(
                workflow_id=manuscript_id,
                metadata={
                    "manuscript_file": manuscript_path.name,
                    "output_file": output_path.name,
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

            print(f"Saved pluralistic review JSON to: {output_path}")

        except Exception as exc:
            print(f"Failed for {manuscript_path.name}: {exc}")

    report_path = make_run_report_path(OUTPUT_DIR, MODEL_NAME)
    reporter.write_report(report_path)
    print(f"Saved LLM cost/performance report to: {report_path}")

    print("Done.")


if __name__ == "__main__":
    main()
