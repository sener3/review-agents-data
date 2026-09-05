from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMProvider(ABC):
    provider_name: str = "unknown"
    model: str = "unknown"

    @property
    def reviewer_label(self) -> str:
        mapping = {
            "openai": "GPT",
            "google": "Gemini",
            "anthropic": "Claude",
        }
        return mapping.get(self.provider_name, self.provider_name.title())

    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[T],
        temperature: float = 0.2,
    ) -> T:
        raw_text = self._call_model(system_prompt, user_prompt, temperature)

        try:
            return self._validate_structured_output(raw_text, response_model)
        except Exception as e:
            repaired = self._repair_and_validate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                raw_text=raw_text,
                response_model=response_model,
                error=e,
            )
            if repaired is not None:
                return repaired
            raise ValueError(
                f"Failed to validate model output for {response_model.__name__}.\n\n"
                f"Raw response:\n{raw_text}\n\n"
                f"Validation/parsing error:\n{e}"
            ) from e

    @abstractmethod
    def _call_model(
        self, system_prompt: str, user_prompt: str, temperature: float
    ) -> str:
        raise NotImplementedError

    def _call_repair_model(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
    ) -> str:
        return self._call_model(system_prompt, user_prompt, temperature)

    def _validate_structured_output(self, raw_text: str, response_model: Type[T]) -> T:
        data = self._safe_json_loads(raw_text)
        data = self._normalize_for_model(data, response_model)
        return response_model.model_validate(data)

    def _repair_and_validate(
        self,
        system_prompt: str,
        user_prompt: str,
        raw_text: str,
        response_model: Type[T],
        error: Exception,
    ) -> T | None:
        repair_prompt = f"""
The previous answer was intended to be JSON for the schema {response_model.__name__}, but it did not validate.
Repair it into a single valid JSON object that matches the schema exactly.
Do not add commentary.
Do not wrap in markdown fences.

Schema:
{json.dumps(response_model.model_json_schema(), indent=2, ensure_ascii=False)}

Original system prompt:
{system_prompt}

Original user prompt:
{user_prompt}

Invalid model output:
{raw_text}

Validation/parsing error:
{str(error)}
""".strip()

        try:
            repaired_text = self._call_repair_model(
                system_prompt="Repair invalid JSON into valid schema-compliant JSON only.",
                user_prompt=repair_prompt,
                temperature=0.0,
            )
            return self._validate_structured_output(repaired_text, response_model)
        except Exception:
            return None

    def _extract_json_block(self, text: str) -> str | None:
        starts = [i for i, ch in enumerate(text) if ch in "{["]
        if not starts:
            return None

        for start in starts:
            stack: list[str] = []
            in_string = False
            escape = False

            for i in range(start, len(text)):
                ch = text[i]

                if in_string:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_string = False
                    continue

                if ch == '"':
                    in_string = True
                elif ch in "{[":
                    stack.append(ch)
                elif ch in "}]":
                    if not stack:
                        break
                    opener = stack.pop()
                    if (opener == "{" and ch != "}") or (opener == "[" and ch != "]"):
                        break
                    if not stack:
                        return text[start : i + 1]

        return None

    def _safe_json_loads(self, text: str) -> Any:
        if not text or not text.strip():
            raise ValueError("Model returned empty response; expected JSON.")

        raw = text.strip()
        raw = re.sub(r"^```json\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"^```\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        candidate = self._extract_json_block(raw)
        if candidate is not None:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                raw = candidate

        repaired = raw.replace("“", '"').replace("”", '"').replace("’", "'")
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
        repaired = "".join(
            ch for ch in repaired if ch in {"\n", "\r", "\t"} or ord(ch) >= 32
        )
        return json.loads(repaired)

    def _normalize_for_model(self, data: Any, response_model: Type[BaseModel]) -> Any:
        if not isinstance(data, dict):
            return data

        name = response_model.__name__

        if name == "PaperClassification":
            data = dict(data)
            data["paper_type"] = self._normalize_paper_type(data.get("paper_type"))
            data["confidence"] = self._normalize_confidence(data.get("confidence"))
            return data
        if name == "SharedAnalysisOutput":
            return self._normalize_shared_analysis_output(data)
        if name == "SpecialistReviewOutput":
            return self._normalize_specialist_review_output(data)
        if name == "ClaimVerificationOutput":
            return self._normalize_claim_verification_output(data)
        if name == "NoveltyAssessmentOutput":
            return self._normalize_novelty_assessment_output(data)
        if name == "MetaReviewOutput":
            return self._normalize_meta_review_output(data)
        if name == "HumanFacingMetaReview":
            return self._normalize_human_facing_meta_review(data)

        return data

    def _normalize_paper_type(self, value: Any) -> str:
        allowed = {
            "empirical",
            "methodological",
            "review",
            "theoretical",
            "hybrid",
            "resource",
            "unknown",
        }
        s = str(value or "unknown").strip().lower()
        if s in allowed:
            return s
        if "method" in s:
            return "methodological"
        if "theor" in s:
            return "theoretical"
        if "empir" in s:
            return "empirical"
        if "review" in s:
            return "review"
        if "resource" in s:
            return "resource"
        return "hybrid" if "hybrid" in s else "unknown"

    def _normalize_confidence(self, value: Any) -> str:
        s = str(value or "medium").strip().lower()
        if s in {"low", "medium", "high"}:
            return s
        if "low" in s:
            return "low"
        if "high" in s:
            return "high"
        return "medium"

    def _normalize_severity(self, value: Any) -> str:
        s = str(value or "moderate").strip().lower()
        if s in {"minor", "moderate", "major"}:
            return s
        if "low" in s or "minor" in s:
            return "minor"
        if "high" in s or "major" in s or "severe" in s:
            return "major"
        return "moderate"

    def _normalize_novelty_level(self, value: Any) -> str:
        s = str(value or "moderate").strip().lower()
        if s in {"low", "moderate", "high"}:
            return s
        if "low" in s:
            return "low"
        if "high" in s:
            return "high"
        return "moderate"

    def _ensure_list_of_strings(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        if isinstance(value, list):
            out: list[str] = []
            for item in value:
                if isinstance(item, str):
                    item = item.strip()
                    if item:
                        out.append(item)
                elif isinstance(item, dict):
                    parts = [
                        str(v).strip()
                        for v in item.values()
                        if v is not None and str(v).strip()
                    ]
                    if parts:
                        out.append(" — ".join(parts))
                elif item is not None:
                    item_str = str(item).strip()
                    if item_str:
                        out.append(item_str)
            return out
        item_str = str(value).strip()
        return [item_str] if item_str else []

    def _coerce_claim_item(self, item: Any, index: int) -> dict:
        if not isinstance(item, dict):
            item = {"claim_text": str(item)}

        claim_text = str(
            item.get("claim_text")
            or item.get("claim")
            or item.get("text")
            or "Claim not clearly specified"
        ).strip()
        claim_label = str(
            item.get("claim_label") or item.get("label") or f"Claim {index}"
        ).strip()
        claim_id = str(item.get("claim_id") or f"C{index}").strip()

        return {
            "claim_id": claim_id,
            "claim_label": claim_label,
            "claim_text": claim_text,
            "section_anchor": str(
                item.get("section_anchor") or item.get("location") or "unspecified"
            ).strip(),
            "evidence": self._ensure_list_of_strings(item.get("evidence", [])),
            "uncertainty_flags": self._ensure_list_of_strings(
                item.get("uncertainty_flags", [])
            ),
        }

    def _coerce_assumption_item(self, item: Any, index: int) -> dict:
        if not isinstance(item, dict):
            item = {"assumption_text": str(item)}

        assumption_type = str(item.get("assumption_type") or "explicit").strip().lower()
        if assumption_type not in {"explicit", "implicit", "operational", "evaluation"}:
            assumption_type = "explicit"

        return {
            "assumption_id": str(item.get("assumption_id") or f"A{index}").strip(),
            "assumption_text": str(
                item.get("assumption_text")
                or item.get("text")
                or "Unspecified assumption"
            ).strip(),
            "assumption_type": assumption_type,
            "section_anchor": str(item.get("section_anchor") or "unspecified").strip(),
            "supporting_evidence": self._ensure_list_of_strings(
                item.get("supporting_evidence", item.get("evidence", []))
            ),
            "risk_if_unmet": str(
                item.get("risk_if_unmet")
                or "Potential validity or interpretation risk if false."
            ).strip(),
        }

    def _normalize_shared_analysis_output(self, data: dict) -> dict:
        out = dict(data)
        out["extracted_claims"] = [
            self._coerce_claim_item(item, i + 1)
            for i, item in enumerate(
                out.get("extracted_claims", out.get("identified_claims", []))
            )
        ]
        out["identified_assumptions"] = [
            self._coerce_assumption_item(item, i + 1)
            for i, item in enumerate(out.get("identified_assumptions", []))
        ]

        for key in [
            "argument_focus",
            "contribution_targets",
            "evaluation_targets",
            "uncertainty_flags",
            "chaining_notes",
        ]:
            out[key] = self._ensure_list_of_strings(out.get(key, []))

        return out

    def _normalize_strengths(self, strengths: Any) -> list[dict]:
        out = []
        for i, item in enumerate(strengths or [], start=1):
            if not isinstance(item, dict):
                item = {"title": f"Strength {i}", "description": str(item)}

            title = str(item.get("title") or f"Strength {i}").strip()
            if title.lower() == "unnamed strength":
                title = f"Strength {i}"

            description = str(
                item.get("description")
                or item.get("impact")
                or item.get("interpretation")
                or "Needs fuller description."
            ).strip()
            if len(description) < 20:
                description = f"{description} This point needs a fuller, evidence-based explanation."

            out.append(
                {
                    "title": title,
                    "description": description,
                    "evidence": self._ensure_list_of_strings(item.get("evidence", [])),
                    "confidence": self._normalize_confidence(
                        item.get("confidence", "medium")
                    ),
                }
            )
        return out

    def _normalize_concerns(self, concerns: Any) -> list[dict]:
        out = []
        for i, item in enumerate(concerns or [], start=1):
            if not isinstance(item, dict):
                item = {"title": f"Concern {i}", "description": str(item)}

            title = str(item.get("title") or f"Concern {i}").strip()
            if title.lower() == "unnamed concern":
                title = f"Concern {i}"

            description = str(
                item.get("description") or "Needs fuller description."
            ).strip()
            if len(description) < 20:
                description = f"{description} This concern needs clearer evidentiary justification."

            actionable = str(
                item.get("actionable_suggestion")
                or "Clarify this issue and support it with a concrete revision step."
            ).strip()
            if len(actionable) < 12:
                actionable = (
                    "Clarify this issue and support it with a concrete revision step."
                )

            out.append(
                {
                    "issue_code": str(item.get("issue_code") or f"M{i}").strip(),
                    "title": title,
                    "description": description,
                    "severity": self._normalize_severity(item.get("severity")),
                    "evidence": self._ensure_list_of_strings(item.get("evidence", [])),
                    "actionable_suggestion": actionable,
                    "novelty_risk": bool(item.get("novelty_risk", False)),
                    "methodological_flaw": bool(item.get("methodological_flaw", False)),
                    "confidence": self._normalize_confidence(
                        item.get("confidence", "medium")
                    ),
                }
            )
        return out

    def _normalize_comparable_works(self, works: Any) -> list[dict]:
        out = []
        for item in works or []:
            if not isinstance(item, dict):
                item = {"citation_label": str(item)}

            out.append(
                {
                    "citation_label": str(
                        item.get("citation_label") or "Unknown work"
                    ).strip(),
                    "relevance": str(
                        item.get("relevance")
                        or "Potentially relevant prior or adjacent work."
                    ).strip(),
                    "similarity_or_difference": str(
                        item.get("similarity_or_difference")
                        or "Similarity/difference not clearly specified."
                    ).strip(),
                    "source_trace": item.get("source_trace"),
                }
            )
        return out

    def _normalize_creativity_safeguard(self, value: Any) -> dict:
        data = value if isinstance(value, dict) else {}
        return {
            "lacks_close_analogs": bool(data.get("lacks_close_analogs", False)),
            "treat_as_innovation_signal": bool(
                data.get("treat_as_innovation_signal", False)
            ),
            "novelty_not_validity_note": str(
                data.get("novelty_not_validity_note")
                or "Novelty and validity should be judged separately."
            ).strip(),
            "counterfactual_conditions_for_success": self._ensure_list_of_strings(
                data.get("counterfactual_conditions_for_success", [])
            ),
        }

    def _normalize_confidence_interval(
        self,
        value: Any,
        default_lower: float = 0.35,
        default_upper: float = 0.75,
    ) -> dict:
        data = value if isinstance(value, dict) else {}

        try:
            lower = float(data.get("lower", default_lower))
        except Exception:
            lower = default_lower

        try:
            upper = float(data.get("upper", default_upper))
        except Exception:
            upper = default_upper

        lower = max(0.0, min(1.0, lower))
        upper = max(lower, min(1.0, upper))

        return {
            "lower": lower,
            "upper": upper,
            "rationale": str(
                data.get("rationale")
                or "Confidence reflects the strength and limits of the available evidence."
            ).strip(),
        }

    def _normalize_novelty_lenses(self, lenses: Any) -> list[dict]:
        out = []
        for item in lenses or []:
            if not isinstance(item, dict):
                item = {"lens": "default", "verdict": str(item), "rationale": str(item)}

            out.append(
                {
                    "lens": str(item.get("lens") or "default").strip(),
                    "verdict": str(item.get("verdict") or "mixed").strip(),
                    "rationale": str(
                        item.get("rationale") or "Lens rationale not clearly specified."
                    ).strip(),
                    "evidence": self._ensure_list_of_strings(item.get("evidence", [])),
                    "uncertainty_flags": self._ensure_list_of_strings(
                        item.get("uncertainty_flags", [])
                    ),
                }
            )
        return out

    def _normalize_specialist_review_output(self, data: dict) -> dict:
        out = dict(data)

        allowed_roles = {
            "structure_analyst",
            "methodology_auditor",
            "novelty_assessor_incremental",
            "novelty_assessor_disruptive",
            "claim_verifier",
        }

        role = str(out.get("cognitive_role") or "structure_analyst").strip()
        if role not in allowed_roles:
            role = "structure_analyst"

        out["reviewer_id"] = str(out.get("reviewer_id") or "reviewer_unknown").strip()
        out["provider"] = str(out.get("provider") or "unknown").strip()
        out["model"] = str(out.get("model") or "unknown").strip()
        out["reviewer_label"] = str(
            out.get("reviewer_label") or self.reviewer_label
        ).strip()
        out["cognitive_role"] = role
        out["epistemic_variant"] = str(
            out.get("epistemic_variant") or "default variant"
        ).strip()
        out["paper_type_assumed"] = self._normalize_paper_type(
            out.get("paper_type_assumed")
        )
        out["prior_stage_inputs"] = self._ensure_list_of_strings(
            out.get("prior_stage_inputs", [])
        )
        out["analytical_summary"] = str(out.get("analytical_summary") or "").strip()
        out["logical_argument_graph"] = self._ensure_list_of_strings(
            out.get("logical_argument_graph", [])
        )
        out["identified_claims"] = [
            self._coerce_claim_item(item, i + 1)
            for i, item in enumerate(out.get("identified_claims", []))
        ]
        out["strengths"] = self._normalize_strengths(out.get("strengths", []))
        out["concerns"] = self._normalize_concerns(out.get("concerns", []))
        out["comparable_works"] = self._normalize_comparable_works(
            out.get("comparable_works", [])
        )
        out["uncertainty_flags"] = self._ensure_list_of_strings(
            out.get("uncertainty_flags", [])
        )
        out["alternative_interpretations"] = self._ensure_list_of_strings(
            out.get("alternative_interpretations", [])
        )
        out["transparency_notes"] = self._ensure_list_of_strings(
            out.get("transparency_notes", [])
        )
        out["creativity_safeguard"] = self._normalize_creativity_safeguard(
            out.get("creativity_safeguard")
        )
        out["novelty_lenses"] = self._normalize_novelty_lenses(
            out.get("novelty_lenses", [])
        )
        out["confidence_interval"] = self._normalize_confidence_interval(
            out.get("confidence_interval")
        )
        return out

    def _normalize_verified_claim(self, item: Any, default_status: str) -> dict:
        if not isinstance(item, dict):
            item = {"claim": str(item)}

        status = str(item.get("status") or default_status).strip().lower()
        allowed = {"supported", "partially_supported", "unsupported_or_overstated"}
        if status not in allowed:
            if "partial" in status:
                status = "partially_supported"
            elif "support" in status:
                status = "supported"
            else:
                status = "unsupported_or_overstated"

        note = item.get("note")
        note = None if note in {None, ""} else str(note).strip()

        return {
            "claim": str(
                item.get("claim") or item.get("claim_text") or "Unspecified claim"
            ).strip(),
            "status": status,
            "evidence": self._ensure_list_of_strings(item.get("evidence", [])),
            "literature_context": self._ensure_list_of_strings(
                item.get("literature_context", [])
            ),
            "note": note,
            "confidence": self._normalize_confidence(item.get("confidence", "medium")),
        }

    def _normalize_claim_verification_output(self, data: dict) -> dict:
        out = dict(data)
        out["supported_claims"] = [
            self._normalize_verified_claim(item, "supported")
            for item in out.get("supported_claims", [])
        ]
        out["partially_supported_claims"] = [
            self._normalize_verified_claim(item, "partially_supported")
            for item in out.get("partially_supported_claims", [])
        ]
        out["unsupported_or_overstated_claims"] = [
            self._normalize_verified_claim(item, "unsupported_or_overstated")
            for item in out.get("unsupported_or_overstated_claims", [])
        ]
        out["missing_evidence"] = self._ensure_list_of_strings(
            out.get("missing_evidence", [])
        )
        out["transparency_notes"] = self._ensure_list_of_strings(
            out.get("transparency_notes", [])
        )
        out["confidence_interval"] = self._normalize_confidence_interval(
            out.get("confidence_interval")
        )
        return out

    def _normalize_novelty_assessment_output(self, data: dict) -> dict:
        out = dict(data)
        out["verdict"] = str(out.get("verdict") or "mixed").strip()
        out["novelty_level"] = self._normalize_novelty_level(out.get("novelty_level"))
        out["main_novelty_claim"] = str(
            out.get("main_novelty_claim") or "Main novelty claim not clearly specified."
        ).strip()
        out["closest_prior_art_risk"] = self._ensure_list_of_strings(
            out.get("closest_prior_art_risk", [])
        )
        out["why_incremental_or_distinct"] = self._ensure_list_of_strings(
            out.get("why_incremental_or_distinct", [])
        )
        out["creativity_safeguard"] = self._normalize_creativity_safeguard(
            out.get("creativity_safeguard")
        )
        out["comparable_works"] = self._normalize_comparable_works(
            out.get("comparable_works", [])
        )
        out["transparency_notes"] = self._ensure_list_of_strings(
            out.get("transparency_notes", [])
        )
        out["confidence_interval"] = self._normalize_confidence_interval(
            out.get("confidence_interval")
        )
        return out

    def _normalize_meta_review_output(self, data: dict) -> dict:
        out = dict(data)
        for key in [
            "consensus_findings",
            "disagreements",
            "uncertainty_hotspots",
            "model_diversity_effects",
            "transparency_report",
            "epistemic_limits",
        ]:
            out[key] = self._ensure_list_of_strings(out.get(key, []))

        out["editor_guidance"] = str(
            out.get("editor_guidance")
            or "Clarify how the specialist evidence should be weighed."
        ).strip()
        out["author_guidance"] = str(
            out.get("author_guidance")
            or "Address the highest-confidence concerns and clarify evidence gaps."
        ).strip()
        return out

    def _ensure_analytical_comment_sections(self, text: str) -> str:
        text = (text or "").strip()
        required_sections = [
            "Summary:",
            "Strengths:",
            "Limitations:",
            "Open Questions:",
            "Transparency:",
        ]
        lower = text.lower()

        if all(section.lower() in lower for section in required_sections):
            return text

        return (
            "Summary:\n"
            + (text or "Analytical synthesis generated from the specialist reviewers.")
            + "\n\n"
            + "Strengths:\n- The review pipeline identified specific strengths, but they should be validated against the cited evidence.\n\n"
            + "Limitations:\n- The current evidence base may be incomplete for some claims.\n- Some findings remain sensitive to corpus coverage and retrieval quality.\n\n"
            + "Open Questions:\n- Which claims most depend on assumptions that are not directly tested?\n- Which revisions would most improve the paper's evidentiary grounding?\n\n"
            + "Transparency:\n- This synthesis preserves disagreement and uncertainty rather than collapsing them into a score."
        ).strip()

    def _normalize_human_facing_meta_review(self, data: dict) -> dict:
        out = dict(data)
        out["is_meta_review"] = True
        out["summary"] = str(
            out.get("summary")
            or out.get("analytical_summary")
            or "Analytical synthesis generated from specialist reviews."
        ).strip()
        out["strengths"] = self._ensure_list_of_strings(out.get("strengths", []))
        out["limitations"] = self._ensure_list_of_strings(out.get("limitations", []))
        out["open_questions"] = self._ensure_list_of_strings(
            out.get("open_questions", [])
        )
        out["consensus_snapshot"] = self._ensure_list_of_strings(
            out.get("consensus_snapshot", [])
        )
        out["disagreement_snapshot"] = self._ensure_list_of_strings(
            out.get("disagreement_snapshot", [])
        )
        out["uncertainty_snapshot"] = self._ensure_list_of_strings(
            out.get("uncertainty_snapshot", [])
        )
        out["source_trace"] = self._ensure_list_of_strings(out.get("source_trace", []))
        out["transparency_notes"] = self._ensure_list_of_strings(
            out.get("transparency_notes", [])
        )
        out["editor_guidance"] = str(
            out.get("editor_guidance")
            or "Use the preserved disagreements and uncertainty hotspots as decision support, not as a substitute for editorial judgment."
        ).strip()
        out["author_guidance"] = str(
            out.get("author_guidance")
            or "Prioritize revisions that tighten evidence support, clarify assumptions, and address the most recurrent concerns."
        ).strip()

        comments = out.get("comments")
        if isinstance(comments, list):
            comments = "\n".join(self._ensure_list_of_strings(comments))

        if not comments:
            strengths_block = (
                "\n".join(f"- {item}" for item in out["strengths"][:5])
                or "- No clear strengths were extracted."
            )
            limitations_block = (
                "\n".join(f"- {item}" for item in out["limitations"][:5])
                or "- No concrete limitations were extracted."
            )
            questions_block = (
                "\n".join(f"- {item}" for item in out["open_questions"][:5])
                or "- Which revisions would most improve the evidentiary grounding?"
            )
            transparency_block = (
                "\n".join(f"- {item}" for item in out["transparency_notes"][:5])
                or "- This synthesis preserves disagreement and uncertainty rather than collapsing them into a score."
            )
            comments = (
                f"Summary:\n{out['summary']}\n\n"
                f"Strengths:\n{strengths_block}\n\n"
                f"Limitations:\n{limitations_block}\n\n"
                f"Open Questions:\n{questions_block}\n\n"
                f"Transparency:\n{transparency_block}"
            )

        out["comments"] = self._ensure_analytical_comment_sections(
            str(comments).strip()
        )
        out["confidence_interval"] = self._normalize_confidence_interval(
            out.get("confidence_interval")
        )
        return out
