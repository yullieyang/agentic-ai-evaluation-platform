"""Optional second-pass reviewer agent.

The reviewer inspects the first agent's finding alongside the supplied evidence
and deterministic findings, verifies evidence support, flags unsupported claims
and contradictions, and may revise the severity or decision. It returns a strict
``ReviewerOutput``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

from .agent import PROMPT_PROFILES
from .llm_providers import BaseProvider, ProviderResponse
from .schemas import ReviewerOutput, json_schema, parse_model
from .utils import get_logger, load_config

LOGGER = get_logger("reviewer_agent")


@dataclass
class ReviewerResult:
    case_id: str
    review: Optional[dict[str, Any]]
    schema_valid: bool
    validation_error: Optional[str]
    response: ProviderResponse


class ReviewerAgent:
    """Runs a single reviewer pass over a first-agent finding."""

    def __init__(self, provider: BaseProvider, prompts_cfg: dict | None = None,
                 max_retries: int = 2, max_tokens: int = 1200):
        self.provider = provider
        self.prompts_cfg = prompts_cfg or load_config("prompts.yaml")
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self._schema = json_schema(ReviewerOutput)

    def run(self, case: dict[str, Any], agent_finding: dict[str, Any],
            prompt_evidence: dict[str, Any], mock_evidence: dict[str, Any]) -> ReviewerResult:
        cfg = self.prompts_cfg["reviewer"]
        system = cfg["system"].strip()
        user = (
            cfg["user_template"]
            .replace("{evidence_block}", json.dumps(prompt_evidence, indent=2))
            .replace("{agent_output}", json.dumps(agent_finding, indent=2))
        )
        context_base = {
            "mode": "reviewer",
            "case_id": case["row"]["case_id"],
            "prompt_version": case["prompt_version"],
            "profile": PROMPT_PROFILES.get(case["prompt_version"], {}),
            "evidence": mock_evidence,
            "agent_finding": agent_finding,
        }

        last: ProviderResponse | None = None
        review_dict: Optional[dict[str, Any]] = None
        validation_error: Optional[str] = None

        for attempt in range(self.max_retries + 1):
            ctx = dict(context_base, attempt=attempt)
            resp = self.provider.complete(
                system, user, model=case["model"], temperature=case.get("temperature", 0.0),
                max_tokens=self.max_tokens, schema=self._schema, context=ctx,
            )
            last = resp
            if resp.parsed is None:
                validation_error = resp.parsing_error or "no parsed output"
                continue
            payload = {k: v for k, v in resp.parsed.items() if not k.startswith("_")}
            try:
                review = parse_model(ReviewerOutput, payload)
                review_dict = json.loads(review.json()) if hasattr(review, "json") else payload
                validation_error = None
                break
            except Exception as exc:  # noqa: BLE001
                validation_error = f"{type(exc).__name__}: {exc}"
                review_dict = None
                continue

        assert last is not None
        return ReviewerResult(
            case_id=case["row"]["case_id"], review=review_dict,
            schema_valid=review_dict is not None, validation_error=validation_error,
            response=last,
        )
