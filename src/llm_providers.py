"""LLM provider abstraction.

Three provider types are supported behind a single interface:

* ``MockProvider``     — fully offline, deterministic, seeded. Default.
* ``AnthropicProvider``— Claude via the official ``anthropic`` SDK (lazy import).
* ``OpenAIProvider``   — OpenAI via the official ``openai`` SDK (lazy import).

The real providers are imported only when instantiated, so the library and test
suite remain importable without those packages or any API key.

The mock provider does not call any model. It synthesizes a plausible QA finding
from a structured ``context`` dictionary (the same evidence rendered into the
prompt) using documented, seeded rules. It can produce correct and incorrect
decisions, miscalibrated confidence, unsupported claims, abstentions, and
malformed output, and its behaviour responds to a per-prompt "profile" so that
prompt-sensitivity can be studied offline. Mock outputs are simulation, not
measurements of any real model, and are labelled as such throughout.
"""

from __future__ import annotations

import abc
import datetime as _dt
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .utils import estimate_cost, extract_json_block, get_logger

LOGGER = get_logger("llm_providers")


@dataclass
class ProviderResponse:
    """Uniform response object returned by every provider."""

    parsed: Optional[dict[str, Any]]
    raw: str
    provider: str
    model: str
    latency_s: float
    input_tokens: int
    output_tokens: int
    estimated_cost: Optional[float]
    parsing_error: Optional[str]
    retry_count: int
    timestamp: str
    is_mock: bool = False
    tool_calls: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


def _now() -> str:
    return _dt.datetime.utcnow().isoformat() + "Z"


class BaseProvider(abc.ABC):
    """Provider interface. ``context`` is ignored by real providers and used by
    the mock to synthesize offline behaviour."""

    name: str = "base"

    @abc.abstractmethod
    def complete(
        self,
        system: str,
        user: str,
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1500,
        schema: dict | None = None,
        context: dict | None = None,
    ) -> ProviderResponse:
        ...

    def complete_with_tools(
        self,
        system: str,
        user: str,
        *,
        model: str,
        tool_specs: list[dict[str, Any]],
        executor: Any,  # evidence_tools.ToolExecutor
        temperature: float = 0.0,
        max_tokens: int = 1500,
        schema: dict | None = None,
        context: dict | None = None,
    ) -> ProviderResponse:
        """Run a bounded tool-use loop and return a final schema-constrained
        response. Only implemented by providers that support the agentic
        (tool-using) condition; the base implementation raises so an
        unsupported provider fails loudly rather than silently behaving like
        the single-shot path."""
        raise NotImplementedError(
            f"provider '{self.name}' does not implement the agentic tool-use condition"
        )


# --------------------------------------------------------------------------- #
# Mock provider
# --------------------------------------------------------------------------- #
@dataclass
class MockProfile:
    """Behavioural knobs that let the mock respond to prompt configuration."""

    mitigation_skill: float = 0.5   # ability to use mitigation checks to avoid traps
    fn_recovery: float = 0.25       # ability to catch masked (false-negative) shifts
    abstain_bias: float = 0.4       # tendency to abstain under incomplete evidence
    unsupported_rate: float = 0.25  # probability of emitting an unsupported claim
    miscalibration: float = 0.35    # probability of overconfidence when wrong
    malformed_rate: float = 0.04    # probability of malformed JSON on first attempt


class MockProvider(BaseProvider):
    """Deterministic, seeded mock provider. See module docstring."""

    name = "mock"

    def __init__(self, error_rate: float = 0.12, base_latency: float = 0.0):
        self.error_rate = float(error_rate)
        self.base_latency = float(base_latency)

    @staticmethod
    def _seed(context: dict, attempt: int) -> int:
        key = json.dumps(
            {
                "case_id": context.get("case_id"),
                "prompt_version": context.get("prompt_version"),
                "include_det": context.get("include_deterministic_evidence"),
                "reviewer": context.get("reviewer_enabled"),
                "completeness": context.get("evidence_completeness"),
                "attempt": attempt,
            },
            sort_keys=True,
        )
        return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)

    def complete(self, system, user, *, model, temperature=0.0, max_tokens=1500,
                 schema=None, context=None) -> ProviderResponse:
        context = context or {}
        attempt = int(context.get("attempt", 0))
        rng = np.random.default_rng(self._seed(context, attempt))
        profile = MockProfile(**context.get("profile", {}))

        # Simulated latency and token usage (so cost/latency tradeoffs are visible).
        latency = self.base_latency + float(rng.uniform(0.05, 0.25))
        in_tok = int(len(user) / 4) + 200
        out_tok = int(rng.integers(120, 320))

        # Reviewer mode produces a ReviewerOutput-shaped object.
        if context.get("mode") == "reviewer":
            review = self._review(context, rng, profile)
            raw = json.dumps(review)
            return ProviderResponse(
                parsed=review, raw=raw, provider=self.name, model=model, latency_s=latency,
                input_tokens=in_tok, output_tokens=len(raw) // 4, estimated_cost=0.0,
                parsing_error=None, retry_count=attempt, timestamp=_now(), is_mock=True,
            )

        # Malformed-output simulation (first attempt only, so retries can recover).
        if attempt == 0 and rng.random() < profile.malformed_rate:
            return ProviderResponse(
                parsed=None, raw="ANOMALY: yes (not valid json)", provider=self.name,
                model=model, latency_s=latency, input_tokens=in_tok, output_tokens=out_tok,
                estimated_cost=estimate_cost(model, in_tok, out_tok),
                parsing_error="simulated malformed output", retry_count=attempt,
                timestamp=_now(), is_mock=True,
            )

        finding = self._decide(context, rng, profile)
        raw = json.dumps(finding)
        return ProviderResponse(
            parsed=finding, raw=raw, provider=self.name, model=model, latency_s=latency,
            input_tokens=in_tok, output_tokens=len(raw) // 4, estimated_cost=0.0,
            parsing_error=None, retry_count=attempt, timestamp=_now(), is_mock=True,
        )

    def complete_with_tools(self, system, user, *, model, tool_specs, executor,
                            temperature=0.0, max_tokens=1500, schema=None,
                            context=None) -> ProviderResponse:
        """Deterministic offline simulation of the agentic (tool-using) condition.

        This does not fake tool calls — it drives the *real* ``ToolExecutor``
        against the case's real ``EvidenceStore`` with a scripted, seeded
        retrieval policy (list evidence, then fetch metrics/baseline and,
        conditionally, mitigation-relevant items), so ``executor.call_log`` and
        ``executor.calls_made`` reflect genuine dispatch through the same code
        path the live Anthropic tool loop uses.

        The final decision reuses the exact same ``_decide`` logic as the
        single-shot mock path, deliberately: this experiment is designed to
        isolate the *evidence-delivery mechanism* (pre-supplied vs. retrieved)
        from *decision quality*, per the documented single-shot/agentic
        comparison design. It does **not** demonstrate that a live agentic
        Claude run would perform differently than single-shot — only a live
        run could show that; this mock only proves the retrieval/tool-limit/
        citation machinery works end-to-end.
        """
        context = context or {}
        attempt = int(context.get("attempt", 0))
        rng = np.random.default_rng(self._seed(context, attempt))
        profile = MockProfile(**context.get("profile", {}))
        case_id = context.get("case_id")

        latency = self.base_latency + float(rng.uniform(0.08, 0.35))
        in_tok = int(len(user) / 4) + 150
        ev = context.get("evidence", {})

        # Scripted, deterministic retrieval policy.
        listing = executor.call("list_available_evidence", {"case_id": case_id})
        by_kind: dict[str, list[str]] = {}
        for item in listing.get("items", []):
            by_kind.setdefault(item["kind"], []).append(item["evidence_id"])
        executor.call("get_case_metrics", {"case_id": case_id})
        z = abs(float(ev.get("z_score", 0.0)))
        if ev.get("baseline_anomaly") or z >= 2:
            executor.call("get_historical_baseline", {"case_id": case_id})
        if ev.get("mitigation_flags"):
            executor.call("get_release_notes", {"case_id": case_id})
            for eid in by_kind.get("seasonality_indicator", [])[:1]:
                executor.call("get_evidence_item", {"case_id": case_id, "evidence_id": eid})
        if "deterministic" in ev.get("available_sources", []) and ev.get("baseline_anomaly"):
            for eid in by_kind.get("deterministic_check", [])[:1]:
                executor.call("get_evidence_item", {"case_id": case_id, "evidence_id": eid})

        finding = self._decide(context, rng, profile)
        # Reconcile cited evidence_ids against what was actually retrieved this
        # run (the tool-use condition can only cite IDs it fetched).
        fetched_ids = {e.result["evidence_id"] for e in executor.call_log
                       if e.ok and isinstance(e.result, dict) and "evidence_id" in e.result}
        finding["evidence_ids"] = [i for i in finding.get("evidence_ids", []) if i in fetched_ids] \
            or list(fetched_ids)[:1]

        raw = json.dumps(finding)
        return ProviderResponse(
            parsed=finding, raw=raw, provider=self.name, model=model, latency_s=latency,
            input_tokens=in_tok, output_tokens=len(raw) // 4, estimated_cost=0.0,
            parsing_error=None, retry_count=attempt, timestamp=_now(), is_mock=True,
            tool_calls=executor.calls_made,
        )

    def _review(self, ctx: dict, rng: np.random.Generator, prof: MockProfile) -> dict:
        """Simulate a second-pass reviewer that verifies evidence support.

        The reviewer downgrades anomalies that are explained by mitigation checks
        and flags unsupported causal claims by pattern-matching the first agent's
        explanations. This is offline simulation, not a real model.
        """
        finding = ctx.get("agent_finding", {})
        ev = ctx.get("evidence", {})
        mitigations = set(ev.get("mitigation_flags", []))
        revised_anom = bool(finding.get("anomaly_detected", False))
        revised_sev = finding.get("severity", "none")
        revised_conf = float(finding.get("confidence_score", 0.5))
        unsupported: list[str] = []
        missing: list[str] = []
        decision = "approve"

        # Detect unsupported causal claims by surface pattern.
        for claim in finding.get("possible_explanations", []):
            text = str(claim).lower()
            if ("caused by" in text or "because of" in text) and "hypothesis" not in text:
                unsupported.append(str(claim))

        # Downgrade anomalies explained by mitigation checks (skill-gated).
        if revised_anom and mitigations and rng.random() < prof.mitigation_skill:
            revised_anom = False
            revised_sev = "none"
            revised_conf = max(0.2, revised_conf - 0.2)
            decision = "revise"

        if unsupported:
            decision = "reject" if not revised_anom else "revise"
            revised_conf = max(0.2, revised_conf - 0.15)

        requires_review = bool(unsupported) or bool(mitigations) or ev.get("incomplete", False)
        return {
            "review_decision": decision,
            "revised_anomaly_detected": bool(revised_anom),
            "revised_severity": revised_sev if revised_anom else "none",
            "revised_anomaly_type": finding.get("anomaly_type", "none") if revised_anom else "none",
            "unsupported_claims_found": unsupported,
            "missing_evidence": missing,
            "review_summary": (
                "Revised the finding after verifying evidence support."
                if decision != "approve" else "Finding is supported by the supplied evidence."
            ),
            "revised_confidence": round(float(np.clip(revised_conf, 0.05, 0.99)), 4),
            "requires_human_review": bool(requires_review),
        }

    def _decide(self, ctx: dict, rng: np.random.Generator, prof: MockProfile) -> dict:
        ev = ctx.get("evidence", {})
        z = abs(float(ev.get("z_score", 0.0)))
        robust_z = abs(float(ev.get("robust_z_score", z)))
        baseline_anom = bool(ev.get("baseline_anomaly", False))
        baseline_sev = ev.get("baseline_severity", "none")
        mitigations = set(ev.get("mitigation_flags", []))   # e.g. seasonal_warning, recovery
        incomplete = bool(ev.get("incomplete", False))
        missing_rate = float(ev.get("missing_rate", 0.0))
        small_sample = bool(ev.get("small_sample", False))

        decision = baseline_anom
        severity = baseline_sev if baseline_anom else "none"
        abstained = False
        abstention_reason = None
        requires_review = False
        unsupported = False

        # Abstain under incomplete/very-poor evidence with prompt-dependent bias.
        if (incomplete or missing_rate >= 0.12) and rng.random() < prof.abstain_bias:
            abstained = True
            decision = False
            severity = "none"
            requires_review = True
            abstention_reason = "Evidence is insufficient or unreliable to decide."

        # Use mitigation checks to avoid false-positive traps (skill-gated).
        if not abstained and decision and mitigations and rng.random() < prof.mitigation_skill:
            decision = False
            severity = "none"
            requires_review = True

        # Catch masked (false-negative) shifts (weaker, skill-gated).
        if not abstained and not decision and small_sample and robust_z >= 1.3:
            if rng.random() < prof.fn_recovery:
                decision = True
                severity = "low"

        # Random error: flip the decision with the configured base error rate.
        if not abstained and rng.random() < self.error_rate:
            decision = not decision
            severity = ("low" if z < 3 else "medium") if decision else "none"

        # Confidence model: higher with stronger, agreeing signal; miscalibrated
        # (overconfident) on some flipped/erroneous cases.
        agreement = 1.0 if decision == baseline_anom else 0.0
        base_conf = 0.45 + 0.1 * min(z, 4.0) / 4.0 + 0.2 * agreement
        if rng.random() < prof.miscalibration and agreement == 0.0:
            base_conf = min(0.95, base_conf + 0.35)  # overconfident when wrong
        confidence = float(np.clip(base_conf + rng.normal(0, 0.05), 0.05, 0.99))

        # Unsupported claim injection (prompt-dependent).
        possible = []
        claim_risk = "low"
        if not abstained and decision and rng.random() < prof.unsupported_rate:
            possible.append("Likely caused by an upstream data-pipeline change.")  # not in evidence
            unsupported = True
            claim_risk = "high"
        elif decision:
            possible.append("Investigate input-data distribution shift (hypothesis).")

        evidence_items = []
        evidence_ids: list[str] = []
        id_map = ev.get("evidence_id_by_kind", {})
        if decision or z >= 2:
            evidence_items.append({
                "metric_name": ev.get("metric_name", "metric"),
                "observed_value": float(ev.get("z_score", 0.0)),
                "reference_value": 0.0,
                "comparison": "above" if ev.get("z_score", 0.0) > 0 else "below",
                "source": "feature",
                "relevance": "Standardized deviation from historical mean.",
            })
            evidence_ids.extend(id_map.get("metric_snapshot", [])[:1])
        if "deterministic" in ev.get("available_sources", []) and baseline_anom:
            evidence_items.append({
                "metric_name": "deterministic_check",
                "observed_value": 1.0,
                "reference_value": None,
                "comparison": "triggered",
                "source": "deterministic_check",
                "relevance": "Rule baseline flagged a magnitude exceedance.",
            })
            evidence_ids.extend(id_map.get("deterministic_check", [])[:1])
        if not abstained and mitigations:
            evidence_ids.extend(id_map.get("seasonality_indicator", [])[:1])
            evidence_ids.extend(id_map.get("recovery_indicator", [])[:1])

        sufficiency = "insufficient" if abstained else (
            "limited" if (small_sample or missing_rate >= 0.05) else "adequate")
        if not abstained and (decision != baseline_anom or mitigations or small_sample):
            requires_review = True

        return {
            "anomaly_detected": bool(decision),
            "severity": severity,
            "anomaly_type": (ev.get("metric_name", "metric") + "_shift") if decision else "none",
            "summary": (
                "Potential anomaly indicated by the supplied statistics."
                if decision else
                ("Insufficient evidence to decide; recommend human review."
                 if abstained else "No anomaly indicated relative to history.")
            ),
            "observations": [
                f"z-score magnitude {z:.2f}",
                f"missing_rate {missing_rate:.3f}",
            ],
            "supporting_evidence": evidence_items,
            "evidence_ids": evidence_ids,
            "affected_metrics": [ev.get("metric_name", "metric")] if decision else [],
            "possible_explanations": possible,
            "recommended_follow_up": (
                ["Review segment-level inputs for the latest quarter."] if decision
                else (["Confirm data completeness before re-evaluating."] if abstained else [])
            ),
            "confidence_score": round(confidence, 4),
            "evidence_sufficiency": sufficiency,
            "unsupported_claim_risk": claim_risk,
            "requires_human_review": bool(requires_review),
            "abstained": bool(abstained),
            "abstention_reason": abstention_reason,
            "_mock_unsupported_injected": unsupported,
        }


# --------------------------------------------------------------------------- #
# Real providers (lazy imports; never required for tests or mock mode)
# --------------------------------------------------------------------------- #
class AnthropicProvider(BaseProvider):
    name = "anthropic"

    def __init__(self) -> None:
        import anthropic  # noqa: F401  (lazy; raises if unavailable)

        self._anthropic = anthropic
        self._client = anthropic.Anthropic()

    def complete(self, system, user, *, model, temperature=0.0, max_tokens=1500,
                 schema=None, context=None) -> ProviderResponse:
        import time

        kwargs: dict[str, Any] = dict(
            model=model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": user}],
        )
        if schema is not None:
            kwargs["output_config"] = {"format": {"type": "json_schema", "schema": schema}}
        start = time.time()
        try:
            resp = self._client.messages.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            # Distinguishable from a model-quality/parsing failure: this is the
            # provider/network/auth layer, not the model's output.
            return ProviderResponse(
                parsed=None, raw="", provider=self.name, model=model,
                latency_s=time.time() - start, input_tokens=0, output_tokens=0,
                estimated_cost=None, parsing_error=f"provider error: {type(exc).__name__}: {exc}",
                retry_count=int((context or {}).get("attempt", 0)), timestamp=_now(),
            )
        latency = time.time() - start
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        in_tok = getattr(resp.usage, "input_tokens", 0)
        out_tok = getattr(resp.usage, "output_tokens", 0)
        parsed, err = _safe_parse(text)
        return ProviderResponse(
            parsed=parsed, raw=text, provider=self.name, model=model, latency_s=latency,
            input_tokens=in_tok, output_tokens=out_tok,
            estimated_cost=estimate_cost(model, in_tok, out_tok), parsing_error=err,
            retry_count=int((context or {}).get("attempt", 0)), timestamp=_now(),
        )

    def complete_with_tools(self, system, user, *, model, tool_specs, executor,
                            temperature=0.0, max_tokens=1500, schema=None,
                            context=None) -> ProviderResponse:
        """Bounded Claude tool-use loop against the local ``ToolExecutor``.

        Two-phase pattern, verified against the installed ``anthropic`` SDK
        (0.117.0): (1) loop with ``tools=`` until the model stops requesting
        tool calls or the executor's call limit is reached; (2) one final
        call with ``tools`` omitted and ``output_config`` set, so the last
        turn is forced into the same JSON-schema-constrained shape the
        single-shot path already uses. Combining live ``tools=`` with
        ``output_config`` in the same call is not a pattern this codebase
        relies on or has verified, so the schema is only enforced on that
        final, tool-free turn.

        IMPORTANT: this method has not been executed against the live API in
        this environment (no ANTHROPIC_API_KEY was available) — see the
        mock-vs-live disclosure in the README and dashboard. It is written
        and unit-tested against a stubbed client that exercises this exact
        control flow (tool_use -> tool_result -> ... -> final schema call);
        that is the strongest verification possible without a live key.
        """
        import time

        messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
        start = time.time()
        total_in_tok = 0
        total_out_tok = 0

        def _provider_error(exc: Exception) -> ProviderResponse:
            return ProviderResponse(
                parsed=None, raw="", provider=self.name, model=model,
                latency_s=time.time() - start, input_tokens=total_in_tok, output_tokens=total_out_tok,
                estimated_cost=None, parsing_error=f"provider error: {type(exc).__name__}: {exc}",
                retry_count=int((context or {}).get("attempt", 0)), timestamp=_now(),
                tool_calls=executor.calls_made,
            )

        while not executor.limit_reached():
            try:
                resp = self._client.messages.create(
                    model=model, max_tokens=max_tokens, system=system,
                    messages=messages, tools=tool_specs,
                )
            except Exception as exc:  # noqa: BLE001
                return _provider_error(exc)
            total_in_tok += getattr(resp.usage, "input_tokens", 0)
            total_out_tok += getattr(resp.usage, "output_tokens", 0)
            tool_use_blocks = [b for b in resp.content if getattr(b, "type", "") == "tool_use"]
            if not tool_use_blocks:
                break
            messages.append({"role": "assistant", "content": [_block_to_param(b) for b in resp.content]})
            tool_results = []
            for block in tool_use_blocks:
                if executor.limit_reached():
                    tool_results.append({"type": "tool_result", "tool_use_id": block.id,
                                         "content": json.dumps({"error": "tool-call limit reached"}),
                                         "is_error": True})
                    continue
                result = executor.call(block.name, block.input or {})
                tool_results.append({"type": "tool_result", "tool_use_id": block.id,
                                     "content": json.dumps(result),
                                     "is_error": "error" in result})
            messages.append({"role": "user", "content": tool_results})

        # Final, tool-free call forced into the AgentFinding schema.
        messages.append({"role": "user", "content": "Return your final structured finding now."})
        final_kwargs: dict[str, Any] = dict(model=model, max_tokens=max_tokens, system=system,
                                            messages=messages)
        if schema is not None:
            final_kwargs["output_config"] = {"format": {"type": "json_schema", "schema": schema}}
        try:
            final_resp = self._client.messages.create(**final_kwargs)
        except Exception as exc:  # noqa: BLE001
            return _provider_error(exc)
        latency = time.time() - start
        total_in_tok += getattr(final_resp.usage, "input_tokens", 0)
        total_out_tok += getattr(final_resp.usage, "output_tokens", 0)
        text = "".join(b.text for b in final_resp.content if getattr(b, "type", "") == "text")
        parsed, err = _safe_parse(text)
        return ProviderResponse(
            parsed=parsed, raw=text, provider=self.name, model=model, latency_s=latency,
            input_tokens=total_in_tok, output_tokens=total_out_tok,
            estimated_cost=estimate_cost(model, total_in_tok, total_out_tok), parsing_error=err,
            retry_count=int((context or {}).get("attempt", 0)), timestamp=_now(),
            tool_calls=executor.calls_made,
        )


def _block_to_param(block: Any) -> dict[str, Any]:
    """Convert an Anthropic response content block back into the plain-dict
    shape needed to resubmit it as message content in the next turn."""
    block_type = getattr(block, "type", "")
    if block_type == "text":
        return {"type": "text", "text": block.text}
    if block_type == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    # Fall back to whatever the SDK gives us for any other block type.
    return block.model_dump() if hasattr(block, "model_dump") else dict(block)


class OpenAIProvider(BaseProvider):
    name = "openai"

    def __init__(self) -> None:
        import openai  # noqa: F401

        self._client = openai.OpenAI()

    def complete(self, system, user, *, model, temperature=0.0, max_tokens=1500,
                 schema=None, context=None) -> ProviderResponse:
        import time

        start = time.time()
        resp = self._client.chat.completions.create(
            model=model, temperature=temperature, max_tokens=max_tokens,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        latency = time.time() - start
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        in_tok = getattr(usage, "prompt_tokens", 0)
        out_tok = getattr(usage, "completion_tokens", 0)
        parsed, err = _safe_parse(text)
        return ProviderResponse(
            parsed=parsed, raw=text, provider=self.name, model=model, latency_s=latency,
            input_tokens=in_tok, output_tokens=out_tok,
            estimated_cost=estimate_cost(model, in_tok, out_tok), parsing_error=err,
            retry_count=int((context or {}).get("attempt", 0)), timestamp=_now(),
        )


def _safe_parse(text: str) -> tuple[Optional[dict], Optional[str]]:
    try:
        return json.loads(extract_json_block(text)), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def get_provider(name: str, **kwargs: Any) -> BaseProvider:
    """Factory for providers by name."""
    name = name.lower()
    if name == "mock":
        return MockProvider(**kwargs)
    if name == "anthropic":
        return AnthropicProvider()
    if name == "openai":
        return OpenAIProvider()
    raise ValueError(f"unknown provider: {name}")
