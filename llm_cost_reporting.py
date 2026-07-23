from __future__ import annotations

import json
import math
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

# Central pricing config. Values are USD per 1M tokens.
# Update this dict when your provider contract or model routing changes.
MODEL_PRICING_USD: Dict[str, Dict[str, float]] = {
    "gpt-5-mini": {
        "input_per_1m_tokens": 0.25,
        "output_per_1m_tokens": 2.00,
    },
    "gemini-2.5-flash": {
        "input_per_1m_tokens": 0.30,
        "output_per_1m_tokens": 2.50,
    },
    "claude-haiku-4-5": {
        "input_per_1m_tokens": 1.00,
        "output_per_1m_tokens": 5.00,
    },
}

# Conservative fallback for providers/flows that return parsed Pydantic models and
# discard provider usage metadata. Real provider metadata is preferred whenever it
# is visible to this instrumentation layer.
APPROX_CHARS_PER_TOKEN = 4


@dataclass
class LLMCallRecord:
    workflow_id: Optional[str]
    stage: Optional[str]
    call_name: str
    provider: str
    model: str
    start_time_utc: str
    end_time_utc: str
    duration_seconds: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    usage_source: str
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowRunRecord:
    workflow_id: str
    model: str
    provider: str
    start_time_utc: str
    end_time_utc: str
    wall_time_seconds: float
    active_llm_time_seconds: float
    workflow_overhead_seconds: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    llm_call_count: int
    status: str
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class CostPerformanceReporter:
    """Collects per-call and per-workflow LLM cost/performance metrics."""

    def __init__(
        self,
        model_name: str,
        provider_name: str,
        pricing: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        self.model_name = model_name
        self.provider_name = provider_name
        self.pricing = pricing or MODEL_PRICING_USD
        self.llm_calls: List[LLMCallRecord] = []
        self.workflow_runs: List[WorkflowRunRecord] = []
        self._workflow_stack: List[str] = []
        self._stage_stack: List[Tuple[str, Dict[str, Any]]] = []

    @contextmanager
    def workflow_run(
        self,
        workflow_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Iterator[None]:
        start_perf = time.perf_counter()
        start_utc = _utc_now_iso()
        self._workflow_stack.append(workflow_id)
        error: Optional[str] = None
        try:
            yield
        except Exception as exc:
            error = repr(exc)
            raise
        finally:
            end_perf = time.perf_counter()
            end_utc = _utc_now_iso()
            if self._workflow_stack and self._workflow_stack[-1] == workflow_id:
                self._workflow_stack.pop()
            elif workflow_id in self._workflow_stack:
                self._workflow_stack.remove(workflow_id)

            calls = [call for call in self.llm_calls if call.workflow_id == workflow_id]
            active_llm_time = sum(call.duration_seconds for call in calls)
            wall_time = max(end_perf - start_perf, 0.0)
            overhead = max(wall_time - active_llm_time, 0.0)
            self.workflow_runs.append(
                WorkflowRunRecord(
                    workflow_id=workflow_id,
                    model=self.model_name,
                    provider=self.provider_name,
                    start_time_utc=start_utc,
                    end_time_utc=end_utc,
                    wall_time_seconds=_round_seconds(wall_time),
                    active_llm_time_seconds=_round_seconds(active_llm_time),
                    workflow_overhead_seconds=_round_seconds(overhead),
                    input_tokens=sum(call.input_tokens for call in calls),
                    output_tokens=sum(call.output_tokens for call in calls),
                    total_tokens=sum(call.total_tokens for call in calls),
                    estimated_cost_usd=_round_usd(
                        sum(call.estimated_cost_usd for call in calls)
                    ),
                    llm_call_count=len(calls),
                    status="failed" if error else "success",
                    error=error,
                    metadata=metadata or {},
                )
            )

    @contextmanager
    def stage(self, stage_name: str, **metadata: Any) -> Iterator[None]:
        self._stage_stack.append((stage_name, metadata))
        try:
            yield
        finally:
            if self._stage_stack and self._stage_stack[-1][0] == stage_name:
                self._stage_stack.pop()

    def record_call(
        self,
        call_name: str,
        provider: str,
        model: str,
        start_utc: str,
        end_utc: str,
        duration_seconds: float,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
        usage_source: str,
        error: Optional[str] = None,
    ) -> None:
        stage_name, stage_metadata = (
            self._stage_stack[-1] if self._stage_stack else (None, {})
        )
        cost = estimate_cost_usd(
            model_name=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            pricing=self.pricing,
        )
        self.llm_calls.append(
            LLMCallRecord(
                workflow_id=self._workflow_stack[-1] if self._workflow_stack else None,
                stage=stage_name,
                call_name=call_name,
                provider=provider,
                model=model,
                start_time_utc=start_utc,
                end_time_utc=end_utc,
                duration_seconds=_round_seconds(duration_seconds),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                estimated_cost_usd=cost,
                usage_source=usage_source,
                error=error,
                metadata=stage_metadata,
            )
        )

    def build_report(self) -> Dict[str, Any]:
        workflow_count = len(self.workflow_runs)
        totals = {
            "wall_time_seconds": sum(
                run.wall_time_seconds for run in self.workflow_runs
            ),
            "active_llm_time_seconds": sum(
                run.active_llm_time_seconds for run in self.workflow_runs
            ),
            "workflow_overhead_seconds": sum(
                run.workflow_overhead_seconds for run in self.workflow_runs
            ),
            "input_tokens": sum(run.input_tokens for run in self.workflow_runs),
            "output_tokens": sum(run.output_tokens for run in self.workflow_runs),
            "total_tokens": sum(run.total_tokens for run in self.workflow_runs),
            "estimated_cost_usd": sum(
                run.estimated_cost_usd for run in self.workflow_runs
            ),
        }
        totals = {
            key: (
                _round_usd(value)
                if key == "estimated_cost_usd"
                else _round_seconds(value)
                if key.endswith("seconds")
                else value
            )
            for key, value in totals.items()
        }

        means = {
            key: _safe_mean(value, workflow_count) for key, value in totals.items()
        }
        means = {
            key: (
                _round_usd(value)
                if key == "estimated_cost_usd"
                else _round_seconds(value)
                if key.endswith("seconds")
                else value
            )
            for key, value in means.items()
        }

        pricing = self.pricing.get(
            self.model_name,
            {"input_per_1m_tokens": 0.0, "output_per_1m_tokens": 0.0},
        )
        missing_pricing = self.model_name not in self.pricing

        return {
            "report_type": "llm_cost_performance_report",
            "generated_at_utc": _utc_now_iso(),
            "provider": self.provider_name,
            "model": self.model_name,
            "pricing_usd_per_1m_tokens": pricing,
            "pricing_warning": (
                f"No pricing configured for model '{self.model_name}'; estimated costs are 0."
                if missing_pricing
                else None
            ),
            "token_accounting_note": (
                "Uses provider response metadata when available. If the provider wrapper only sees parsed "
                "objects and not raw provider responses, tokens are estimated with a chars/4 fallback."
            ),
            "workflow_count": workflow_count,
            "llm_call_count": len(self.llm_calls),
            "totals": totals,
            "means_per_workflow": means,
            "workflow_runs": [asdict(run) for run in self.workflow_runs],
            "llm_calls": [asdict(call) for call in self.llm_calls],
        }

    def write_report(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.build_report(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def instrument_provider(provider: Any, reporter: CostPerformanceReporter) -> Any:
    """Return a transparent provider proxy that records LLM-call timings/tokens."""
    return InstrumentedLLMProvider(provider=provider, reporter=reporter)


class InstrumentedLLMProvider:
    """Transparent proxy around provider implementations.

    The existing flow code continues to call the provider normally. Public provider
    methods are wrapped so each actual provider call is timed and token/cost usage
    is recorded centrally.
    """

    _SKIP_CALLABLE_NAMES = {
        "model_dump",
        "model_dump_json",
        "dict",
        "json",
        "copy",
    }

    def __init__(self, provider: Any, reporter: CostPerformanceReporter) -> None:
        object.__setattr__(self, "_provider", provider)
        object.__setattr__(self, "_reporter", reporter)

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._provider, name)
        if not callable(attr) or not self._should_wrap(name):
            return attr

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            return self._call_with_reporting(name, attr, *args, **kwargs)

        return wrapped

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"_provider", "_reporter"}:
            object.__setattr__(self, name, value)
        else:
            setattr(self._provider, name, value)

    def __repr__(self) -> str:
        return repr(self._provider)

    def _should_wrap(self, name: str) -> bool:
        return not name.startswith("_") and name not in self._SKIP_CALLABLE_NAMES

    def _call_with_reporting(
        self,
        call_name: str,
        method: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        provider_name = str(
            getattr(self._provider, "provider_name", type(self._provider).__name__)
        )
        model_name = str(getattr(self._provider, "model", self._reporter.model_name))
        start_perf = time.perf_counter()
        start_utc = _utc_now_iso()
        error: Optional[str] = None
        result: Any = None
        try:
            result = method(*args, **kwargs)
            return result
        except Exception as exc:
            error = repr(exc)
            raise
        finally:
            end_perf = time.perf_counter()
            end_utc = _utc_now_iso()
            duration_seconds = max(end_perf - start_perf, 0.0)
            if error is None:
                usage = extract_token_usage(result)
                if usage is None:
                    input_tokens = estimate_tokens({"args": args, "kwargs": kwargs})
                    output_tokens = estimate_tokens(result)
                    total_tokens = input_tokens + output_tokens
                    usage_source = "estimated_from_call_payload_chars_per_4"
                else:
                    input_tokens, output_tokens, total_tokens = usage
                    usage_source = "provider_response_metadata"
            else:
                input_tokens = estimate_tokens({"args": args, "kwargs": kwargs})
                output_tokens = 0
                total_tokens = input_tokens
                usage_source = "estimated_input_only_after_error"

            self._reporter.record_call(
                call_name=call_name,
                provider=provider_name,
                model=model_name,
                start_utc=start_utc,
                end_utc=end_utc,
                duration_seconds=duration_seconds,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                usage_source=usage_source,
                error=error,
            )


def estimate_cost_usd(
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    pricing: Optional[Dict[str, Dict[str, float]]] = None,
) -> float:
    pricing_table = pricing or MODEL_PRICING_USD
    model_pricing = pricing_table.get(
        model_name,
        {"input_per_1m_tokens": 0.0, "output_per_1m_tokens": 0.0},
    )
    input_rate = float(model_pricing.get("input_per_1m_tokens", 0.0))
    output_rate = float(model_pricing.get("output_per_1m_tokens", 0.0))
    return _round_usd(
        (input_tokens / 1_000_000) * input_rate
        + (output_tokens / 1_000_000) * output_rate
    )


def extract_token_usage(value: Any) -> Optional[Tuple[int, int, int]]:
    """Extract token usage from common OpenAI/Anthropic/Gemini response shapes."""
    candidates = [value]
    for container_name in (
        "usage",
        "usage_metadata",
        "token_usage",
        "response_metadata",
        "llm_output",
        "raw_response",
    ):
        nested = _field(value, container_name)
        if nested is not None:
            candidates.append(nested)
            nested_usage = _field(nested, "usage") or _field(nested, "usage_metadata")
            if nested_usage is not None:
                candidates.append(nested_usage)

    for candidate in candidates:
        input_tokens = _first_int(
            candidate,
            (
                "input_tokens",
                "prompt_tokens",
                "prompt_token_count",
                "input_token_count",
                "total_input_tokens",
            ),
        )
        output_tokens = _first_int(
            candidate,
            (
                "output_tokens",
                "completion_tokens",
                "candidates_token_count",
                "output_token_count",
                "total_output_tokens",
            ),
        )
        total_tokens = _first_int(
            candidate,
            (
                "total_tokens",
                "total_token_count",
                "total_tokens_count",
            ),
        )

        if input_tokens is None and output_tokens is None and total_tokens is None:
            continue

        input_tokens = input_tokens or 0
        output_tokens = output_tokens or 0
        if total_tokens is None or total_tokens < input_tokens + output_tokens:
            total_tokens = input_tokens + output_tokens
        return input_tokens, output_tokens, total_tokens

    return None


def estimate_tokens(value: Any) -> int:
    text = _to_estimation_text(value)
    if not text:
        return 0
    return int(math.ceil(len(text) / APPROX_CHARS_PER_TOKEN))


def _to_estimation_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, type):
        return value.__name__
    if hasattr(value, "model_dump_json"):
        try:
            return value.model_dump_json()
        except Exception:
            pass
    if hasattr(value, "model_dump"):
        try:
            return json.dumps(value.model_dump(), ensure_ascii=False, default=str)
        except Exception:
            pass
    try:
        return json.dumps(value, ensure_ascii=False, default=_json_default)
    except Exception:
        return repr(value)


def _json_default(value: Any) -> str:
    if isinstance(value, type):
        return value.__name__
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return repr(value)


def _field(value: Any, name: str) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _first_int(value: Any, names: Tuple[str, ...]) -> Optional[int]:
    for name in names:
        raw = _field(value, name)
        if raw is None:
            continue
        try:
            return int(raw)
        except (TypeError, ValueError):
            continue
    return None


def _safe_mean(value: float, count: int) -> float:
    if count == 0:
        return 0.0
    return value / count


def _round_seconds(value: float) -> float:
    return round(float(value), 6)


def _round_usd(value: float) -> float:
    return round(float(value), 8)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
