"""Explicit evidence-retrieval tools over the local, synthetic evidence store.

These are the tools the agentic (tool-using) condition calls instead of
receiving the whole case pre-assembled in one prompt (that's what the
single-shot condition still does, via ``agent.build_evidence``). Every tool
operates only on the in-memory ``EvidenceStore`` built for the current
experiment run — there is no file, network, or shell access of any kind.

Tool schemas (``TOOL_SPECS``) are hand-written JSON Schema dicts rather than
generated from a Pydantic model, because the installed Pydantic is v1.x and
because the schema needs to match ``anthropic.types.ToolParam.input_schema``
exactly (verified against the installed ``anthropic`` SDK, see
``llm_providers.AnthropicProvider``). The same specs are used to validate
arguments for both the live Claude tool-use loop and the offline mock
tool-selecting agent, so the two conditions exercise one identical interface.

Design note on ``get_validation_rule``: the target design sketch takes an
``alert_type`` argument, but ``alert_type`` is derived 1:1 from
``scenario_type`` for several scenarios and would leak the ground-truth label
if ever surfaced to the agent (see ``evidence_store``'s module docstring and
its leakage test). The validation rule in this dataset is deliberately
universal — the same z-score threshold applies to every case — so this tool
takes ``case_id`` only, for interface consistency with the other tools, and
ignores it functionally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .evidence_store import EvidenceItem, EvidenceStore

MAX_TOOL_CALLS_DEFAULT = 6


class EvidenceToolError(Exception):
    """Raised for invalid tool arguments or an unknown case/evidence_id.

    Always caught by ``ToolExecutor.call`` and turned into a structured error
    result — an agent loop should never crash on a bad tool call, it should
    see the error and can retry, ask for something else, or abstain.
    """


@dataclass
class ToolCallLogEntry:
    tool_name: str
    arguments: dict[str, Any]
    ok: bool
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None


def _require_known_case(store: EvidenceStore, case_id: Any) -> str:
    if not isinstance(case_id, str) or not case_id:
        raise EvidenceToolError("case_id must be a non-empty string")
    if not store.list_for_case(case_id):
        raise EvidenceToolError(f"unknown case_id: {case_id!r}")
    return case_id


def _first_of_kind(store: EvidenceStore, case_id: str, kind: str) -> EvidenceItem:
    items = store.by_kind(case_id, kind)
    if not items:
        raise EvidenceToolError(f"no '{kind}' evidence available for case {case_id!r}")
    return items[0]


# --------------------------------------------------------------------------- #
# Tool implementations
# --------------------------------------------------------------------------- #
def get_case_metrics(store: EvidenceStore, case_id: str) -> dict[str, Any]:
    case_id = _require_known_case(store, case_id)
    item = _first_of_kind(store, case_id, "metric_snapshot")
    return item.to_dict()


def get_historical_baseline(store: EvidenceStore, case_id: str) -> dict[str, Any]:
    case_id = _require_known_case(store, case_id)
    item = _first_of_kind(store, case_id, "historical_baseline")
    return item.to_dict()


def get_release_notes(store: EvidenceStore, case_id: str) -> dict[str, Any]:
    case_id = _require_known_case(store, case_id)
    item = _first_of_kind(store, case_id, "release_note")
    return item.to_dict()


def get_validation_rule(store: EvidenceStore, case_id: str) -> dict[str, Any]:
    """Returns the (universal) validation rule. See module docstring for why
    this does not take an ``alert_type`` argument."""
    case_id = _require_known_case(store, case_id)
    item = _first_of_kind(store, case_id, "validation_rule")
    return item.to_dict()


def get_segment_comparison(store: EvidenceStore, case_id: str) -> dict[str, Any]:
    case_id = _require_known_case(store, case_id)
    item = _first_of_kind(store, case_id, "segment_comparison")
    return item.to_dict()


def get_evidence_item(store: EvidenceStore, case_id: str, evidence_id: str) -> dict[str, Any]:
    case_id = _require_known_case(store, case_id)
    if not isinstance(evidence_id, str) or not evidence_id:
        raise EvidenceToolError("evidence_id must be a non-empty string")
    item = store.get(case_id, evidence_id)
    if item is None:
        raise EvidenceToolError(f"unknown evidence_id {evidence_id!r} for case {case_id!r}")
    return item.to_dict()


def list_available_evidence(store: EvidenceStore, case_id: str) -> dict[str, Any]:
    """Not one of the target's suggested tools, but necessary in practice: an
    agentic run needs some way to discover which evidence_ids/kinds exist for
    a case before fetching them by ID. Returns id/kind/title only — not the
    full evidence content, so this alone can't be used to skip real retrieval."""
    case_id = _require_known_case(store, case_id)
    return {
        "case_id": case_id,
        "items": [
            {"evidence_id": i.evidence_id, "kind": i.kind, "title": i.title}
            for i in store.list_for_case(case_id)
        ],
    }


# --------------------------------------------------------------------------- #
# Tool specs (JSON Schema, matching anthropic.types.ToolParam.input_schema)
# --------------------------------------------------------------------------- #
TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "list_available_evidence",
        "description": "List the evidence_id, kind, and title of every evidence item available "
                        "for a case, without returning the item content. Call this first.",
        "input_schema": {
            "type": "object",
            "properties": {"case_id": {"type": "string"}},
            "required": ["case_id"],
        },
    },
    {
        "name": "get_case_metrics",
        "description": "Get the current-quarter metric statistics (current/previous value, "
                       "percent change, z-score, sample size, missing rate) for a case.",
        "input_schema": {
            "type": "object",
            "properties": {"case_id": {"type": "string"}},
            "required": ["case_id"],
        },
    },
    {
        "name": "get_historical_baseline",
        "description": "Get the historical mean/std and rolling 4-quarter mean/std for a case.",
        "input_schema": {
            "type": "object",
            "properties": {"case_id": {"type": "string"}},
            "required": ["case_id"],
        },
    },
    {
        "name": "get_release_notes",
        "description": "Get release notes for a case's release version, including whether the "
                       "step change is large enough that a metric-definition change should be "
                       "verified (not confirmed) against the release log.",
        "input_schema": {
            "type": "object",
            "properties": {"case_id": {"type": "string"}},
            "required": ["case_id"],
        },
    },
    {
        "name": "get_validation_rule",
        "description": "Get the standard monitoring validation rule (z-score threshold) that "
                       "applies to this case.",
        "input_schema": {
            "type": "object",
            "properties": {"case_id": {"type": "string"}},
            "required": ["case_id"],
        },
    },
    {
        "name": "get_segment_comparison",
        "description": "Get the cross-segment deviation for a case's segment relative to peer "
                       "segments on the same metric.",
        "input_schema": {
            "type": "object",
            "properties": {"case_id": {"type": "string"}},
            "required": ["case_id"],
        },
    },
    {
        "name": "get_evidence_item",
        "description": "Get one specific evidence item by its evidence_id (e.g. a seasonality "
                       "indicator, recovery indicator, or triggered deterministic check).",
        "input_schema": {
            "type": "object",
            "properties": {
                "case_id": {"type": "string"},
                "evidence_id": {"type": "string"},
            },
            "required": ["case_id", "evidence_id"],
        },
    },
]

_DISPATCH: dict[str, Callable[..., dict[str, Any]]] = {
    "list_available_evidence": list_available_evidence,
    "get_case_metrics": get_case_metrics,
    "get_historical_baseline": get_historical_baseline,
    "get_release_notes": get_release_notes,
    "get_validation_rule": get_validation_rule,
    "get_segment_comparison": get_segment_comparison,
    "get_evidence_item": get_evidence_item,
}

TOOL_NAMES = list(_DISPATCH)


class ToolExecutor:
    """Validates and dispatches tool calls against an ``EvidenceStore``, logging
    every call (successful or not) for later audit/artifact generation."""

    def __init__(self, store: EvidenceStore, max_calls: int = MAX_TOOL_CALLS_DEFAULT):
        self.store = store
        self.max_calls = int(max_calls)
        self.call_log: list[ToolCallLogEntry] = []

    @property
    def calls_made(self) -> int:
        return len(self.call_log)

    def limit_reached(self) -> bool:
        return self.calls_made >= self.max_calls

    def call(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute one tool call. Never raises — validation/lookup failures and
        the max-call limit are all returned as a structured error dict, and
        every attempt (successful or not) is appended to ``call_log``."""
        if self.limit_reached():
            entry = ToolCallLogEntry(tool_name, arguments, ok=False,
                                     error=f"tool-call limit ({self.max_calls}) reached")
            self.call_log.append(entry)
            return {"error": entry.error}

        fn = _DISPATCH.get(tool_name)
        if fn is None:
            entry = ToolCallLogEntry(tool_name, arguments, ok=False,
                                     error=f"unknown tool: {tool_name!r}")
            self.call_log.append(entry)
            return {"error": entry.error}

        if not isinstance(arguments, dict):
            entry = ToolCallLogEntry(tool_name, {}, ok=False, error="arguments must be an object")
            self.call_log.append(entry)
            return {"error": entry.error}

        try:
            result = fn(self.store, **arguments)
            self.call_log.append(ToolCallLogEntry(tool_name, arguments, ok=True, result=result))
            return result
        except EvidenceToolError as exc:
            entry = ToolCallLogEntry(tool_name, arguments, ok=False, error=str(exc))
            self.call_log.append(entry)
            return {"error": str(exc)}
        except TypeError as exc:
            # Wrong/missing argument names for the target function signature.
            entry = ToolCallLogEntry(tool_name, arguments, ok=False,
                                     error=f"invalid arguments for {tool_name}: {exc}")
            self.call_log.append(entry)
            return {"error": entry.error}

    def log_as_dicts(self) -> list[dict[str, Any]]:
        return [
            {"tool_name": e.tool_name, "arguments": e.arguments, "ok": e.ok,
             "result": e.result, "error": e.error}
            for e in self.call_log
        ]
