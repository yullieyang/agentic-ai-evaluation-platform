"""SQLite-backed storage for human-in-the-loop review decisions.

Stores analyst decisions (approve/reject/revise/uncertain), corrections, and
notes, and computes acceptance/rejection/revision and disagreement rates,
including agent-rule, agent-human, and reviewer-agent-human disagreement.
"""

from __future__ import annotations

import datetime as _dt
import sqlite3
from pathlib import Path
from typing import Any, Optional

from .utils import OUTPUT_DIR

SCHEMA = """
CREATE TABLE IF NOT EXISTS human_reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id TEXT NOT NULL,
    experiment_id TEXT NOT NULL,
    reviewer_decision TEXT NOT NULL,
    corrected_anomaly_detected INTEGER,
    corrected_severity TEXT,
    corrected_anomaly_type TEXT,
    reviewer_notes TEXT,
    reviewer_confidence REAL,
    unsupported_claim_flag INTEGER DEFAULT 0,
    agent_anomaly INTEGER,
    rule_anomaly INTEGER,
    timestamp TEXT NOT NULL
);
"""

VALID_DECISIONS = {"approve", "reject", "revise", "uncertain"}


class ReviewStore:
    """A thin SQLite wrapper for human review decisions."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else (OUTPUT_DIR / "human_reviews.sqlite")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def add_review(self, review: dict[str, Any]) -> int:
        decision = review.get("reviewer_decision")
        if decision not in VALID_DECISIONS:
            raise ValueError(f"invalid reviewer_decision: {decision}")
        row = {
            "case_id": review["case_id"],
            "experiment_id": review["experiment_id"],
            "reviewer_decision": decision,
            "corrected_anomaly_detected": _as_int(review.get("corrected_anomaly_detected")),
            "corrected_severity": review.get("corrected_severity"),
            "corrected_anomaly_type": review.get("corrected_anomaly_type"),
            "reviewer_notes": review.get("reviewer_notes"),
            "reviewer_confidence": review.get("reviewer_confidence"),
            "unsupported_claim_flag": _as_int(review.get("unsupported_claim_flag", 0)),
            "agent_anomaly": _as_int(review.get("agent_anomaly")),
            "rule_anomaly": _as_int(review.get("rule_anomaly")),
            "timestamp": review.get("timestamp", _dt.datetime.utcnow().isoformat() + "Z"),
        }
        cols = ", ".join(row)
        placeholders = ", ".join(["?"] * len(row))
        cur = self._conn.execute(
            f"INSERT INTO human_reviews ({cols}) VALUES ({placeholders})", tuple(row.values())
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def get_reviews(self, experiment_id: Optional[str] = None) -> list[dict[str, Any]]:
        if experiment_id:
            cur = self._conn.execute(
                "SELECT * FROM human_reviews WHERE experiment_id = ?", (experiment_id,)
            )
        else:
            cur = self._conn.execute("SELECT * FROM human_reviews")
        return [dict(r) for r in cur.fetchall()]

    def disagreement_summary(self, experiment_id: Optional[str] = None) -> dict[str, Any]:
        rows = self.get_reviews(experiment_id)
        n = len(rows)
        if n == 0:
            return {"n": 0}

        def rate(pred) -> float:
            return sum(1 for r in rows if pred(r)) / n

        return {
            "n": n,
            "acceptance_rate": rate(lambda r: r["reviewer_decision"] == "approve"),
            "rejection_rate": rate(lambda r: r["reviewer_decision"] == "reject"),
            "revision_rate": rate(lambda r: r["reviewer_decision"] == "revise"),
            "uncertain_rate": rate(lambda r: r["reviewer_decision"] == "uncertain"),
            "agent_human_disagreement": rate(
                lambda r: r["corrected_anomaly_detected"] is not None
                and r["agent_anomaly"] is not None
                and r["corrected_anomaly_detected"] != r["agent_anomaly"]
            ),
            "rule_human_disagreement": rate(
                lambda r: r["corrected_anomaly_detected"] is not None
                and r["rule_anomaly"] is not None
                and r["corrected_anomaly_detected"] != r["rule_anomaly"]
            ),
            "unsupported_claim_flag_rate": rate(lambda r: bool(r["unsupported_claim_flag"])),
        }

    def close(self) -> None:
        self._conn.close()


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    return int(bool(value)) if isinstance(value, bool) else int(value)
