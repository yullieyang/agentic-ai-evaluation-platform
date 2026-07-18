"""The agentic (tool-using) QA agent — the counterpart to ``agent.QAAgent``.

Where ``QAAgent`` (single-shot) receives the whole case pre-assembled in one
prompt, ``ToolUsingQAAgent`` receives only a minimal, non-leaking case
identity and must retrieve evidence through ``evidence_tools`` — via a
provider's ``complete_with_tools`` — before producing the same
``AgentFinding``-shaped structured output. Both draw from the identical
per-case evidence set (``evidence_store.build_case_evidence_items``), so the
``experiment_runner``'s single-shot vs. agentic comparison isolates the
evidence-*delivery* mechanism rather than giving one condition different
ground truth than the other.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from .evidence_store import EvidenceStore, build_case_evidence_items
from .evidence_tools import TOOL_SPECS, ToolExecutor
from .llm_providers import BaseProvider, ProviderResponse
from .schemas import AgentFinding, assert_no_ground_truth, json_schema, parse_model
from .utils import get_logger, load_config

LOGGER = get_logger("agentic_agent")

DEFAULT_MAX_TOOL_CALLS = 6


@dataclass
class AgenticResult:
    case_id: str
    finding: Optional[dict[str, Any]]
    schema_valid: bool
    validation_error: Optional[str]
    response: ProviderResponse
    prompt_version: str
    total_retries: int
    tool_calls: int
    tool_call_log: list[dict[str, Any]] = field(default_factory=list)
    mock_unsupported_injected: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


def build_minimal_case_framing(row: dict[str, Any]) -> dict[str, Any]:
    """The safe, non-leaking subset of case identity shown up front in the
    agentic condition — deliberately excludes ``alert_type``/``scenario_type``
    (see ``evidence_store``'s leakage note) and all engineered/deterministic
    evidence; the agent must retrieve that through tools."""
    framing = {
        "case_id": row["case_id"], "model_id": row["model_id"],
        "metric_name": row["metric_name"], "segment": row["segment"],
        "quarter": row["quarter"], "expected_direction": row["expected_direction"],
        "release_version": row.get("release_version"),
    }
    assert_no_ground_truth(framing)
    return framing


class ToolUsingQAAgent:
    """Runs a single QA case through the agentic (tool-use) loop."""

    def __init__(self, provider: BaseProvider, prompts_cfg: dict | None = None,
                 eval_cfg: dict | None = None, max_retries: int = 2,
                 max_tokens: int = 1500, max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS):
        self.provider = provider
        self.prompts_cfg = prompts_cfg or load_config("prompts.yaml")
        self.eval_cfg = eval_cfg or load_config("evaluation.yaml")
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.max_tool_calls = max_tool_calls
        self._schema = json_schema(AgentFinding)

    def run(self, case: dict[str, Any]) -> AgenticResult:
        """``case`` must contain: row, features, checks, chart_summary,
        prompt_version, include_deterministic_evidence, mock_evidence
        (the same machine-summary dict ``agent.build_evidence`` returns —
        used only by the offline mock provider), temperature, model."""
        row = case["row"]
        cid = row["case_id"]
        items = build_case_evidence_items(
            case_id=cid, row=row, features=case["features"], checks=case["checks"],
            chart_summary=case["chart_summary"], eval_cfg=self.eval_cfg,
            include_deterministic=case["include_deterministic_evidence"],
        )
        store = EvidenceStore()
        store.register(cid, items)

        variant = self.prompts_cfg["agentic"]
        framing = build_minimal_case_framing(row)
        system = variant["system"].strip()
        user = (
            variant["user_template"]
            .replace("{case_framing}", json.dumps(framing, indent=2))
            .replace("{output_contract}", self.prompts_cfg["shared_output_contract"])
        )

        prompt_version = case["prompt_version"]
        mock_context_base = {
            "case_id": cid, "prompt_version": prompt_version,
            "profile": case.get("mock_profile", {}),
            "evidence": case["mock_evidence"],
        }

        last: Optional[ProviderResponse] = None
        finding_dict: Optional[dict[str, Any]] = None
        validation_error: Optional[str] = None
        total_retries = 0
        total_tool_calls = 0
        tool_call_log: list[dict[str, Any]] = []
        unsupported_injected = False

        for attempt in range(self.max_retries + 1):
            total_retries = attempt
            executor = ToolExecutor(store, max_calls=self.max_tool_calls)
            ctx = dict(mock_context_base, attempt=attempt)
            resp = self.provider.complete_with_tools(
                system, user, model=case["model"], temperature=case.get("temperature", 0.0),
                max_tokens=self.max_tokens, tool_specs=TOOL_SPECS, executor=executor,
                schema=self._schema, context=ctx,
            )
            last = resp
            total_tool_calls += resp.tool_calls
            tool_call_log.extend(executor.log_as_dicts())
            if resp.parsed is None:
                validation_error = resp.parsing_error or "no parsed output"
                continue
            payload = dict(resp.parsed)
            unsupported_injected = unsupported_injected or bool(payload.pop("_mock_unsupported_injected", False))
            for k in [key for key in payload if key.startswith("_")]:
                payload.pop(k)
            try:
                finding = parse_model(AgentFinding, payload)
                finding_dict = json.loads(finding.json()) if hasattr(finding, "json") else payload
                validation_error = None
                break
            except Exception as exc:  # noqa: BLE001
                validation_error = f"{type(exc).__name__}: {exc}"
                finding_dict = None
                continue

        assert last is not None
        return AgenticResult(
            case_id=cid, finding=finding_dict, schema_valid=finding_dict is not None,
            validation_error=validation_error, response=last, prompt_version=prompt_version,
            total_retries=total_retries, tool_calls=total_tool_calls, tool_call_log=tool_call_log,
            mock_unsupported_injected=unsupported_injected,
            extra={"store": store, "evidence_items": items, "max_tool_calls": self.max_tool_calls},
        )
