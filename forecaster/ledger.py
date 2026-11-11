"""Append-only, hash-chained JSONL ledger.

Each line is one record in canonical JSON. A record stores the hash of the record
before it (``prev_hash``) and its own ``hash`` = sha256 of its canonical JSON
including ``prev_hash`` and excluding ``hash``. Editing, inserting, reordering or
removing any earlier record breaks the chain, and ``verify()`` reports it.

What the chain alone cannot show is truncation of the newest records; the git
history of ``data/ledger.jsonl`` covers that, and also proves when each line was added.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from forecaster.models import Forecast, ensure_utc

GENESIS_HASH = "0" * 64

Record = dict[str, Any]


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str, allow_nan=False)


def record_hash(record: Record) -> str:
    """Hash of a record (which must already hold ``prev_hash``), ignoring ``hash``."""
    body = {k: v for k, v in record.items() if k != "hash"}
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


class Ledger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    # ---------- reading ----------

    def records(self) -> list[Record]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as fh:
            return [json.loads(line) for line in fh.read().splitlines()]

    def last_hash(self) -> str:
        records = self.records()
        return records[-1]["hash"] if records else GENESIS_HASH

    def has_market(self, venue: str, market_id: str) -> bool:
        """v1 forecasts each market once, so callers skip markets already here."""
        return any(
            r["type"] == "forecast"
            and r["question"]["venue"] == venue
            and r["question"]["market_id"] == market_id
            for r in self.records()
        )

    # ---------- writing (append only; there is no update or delete) ----------

    def append(self, record: Record) -> Record:
        """Chain ``record`` onto the ledger and return it with its hashes."""
        if "hash" in record or "prev_hash" in record:
            raise ValueError("record must not already contain 'hash' or 'prev_hash'")
        # Round-trip through canonical JSON first so the hash is computed over
        # exactly the values a later reader will parse back from the file.
        chained: Record = json.loads(canonical_json(record))
        chained["prev_hash"] = self.last_hash()
        chained["hash"] = record_hash(chained)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(canonical_json(chained) + "\n")
        return chained

    def append_forecast(self, forecast: Forecast) -> Record:
        body = forecast.to_dict()
        ts = body.pop("created_at")
        return self.append({"type": "forecast", "ts": ts, **body})

    def append_resolution(
        self, venue: str, market_id: str, outcome: int, ts: datetime
    ) -> Record:
        if outcome not in (0, 1):
            raise ValueError(f"outcome must be 0 or 1, got {outcome!r}")
        return self.append(
            {
                "type": "resolution",
                "venue": venue,
                "market_id": market_id,
                "outcome": outcome,
                "ts": ensure_utc(ts).isoformat(),
            }
        )

    # ---------- integrity ----------

    def verify(self) -> bool:
        """Recompute the whole chain. A missing or empty ledger is trivially valid."""
        if not self.path.exists():
            return True
        raw = self.path.read_bytes()
        if not raw:
            return True
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return False
        if not text.endswith("\n"):
            return False
        prev = GENESIS_HASH
        for line in text[:-1].split("\n"):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                return False
            if not isinstance(record, dict):
                return False
            # The line must be byte-for-byte canonical, so no edit of any kind
            # (even one that parses to the same values) goes unnoticed.
            try:
                if canonical_json(record) != line:
                    return False
            except ValueError:
                return False
            if record.get("prev_hash") != prev or record.get("hash") != record_hash(record):
                return False
            prev = record["hash"]
        return True

    # ---------- scoring view ----------

    def joined(self) -> list[Record]:
        """Latest forecast per (venue, market_id), joined to its resolution.

        Each row is the forecast record plus ``outcome`` and ``resolved_at``
        (both None while the market is unresolved), in order of first forecast.
        """
        forecasts: dict[tuple[str, str], Record] = {}
        resolutions: dict[tuple[str, str], Record] = {}
        for r in self.records():
            if r["type"] == "forecast":
                forecasts[(r["question"]["venue"], r["question"]["market_id"])] = r
            elif r["type"] == "resolution":
                resolutions[(r["venue"], r["market_id"])] = r
        rows: list[Record] = []
        for key, forecast in forecasts.items():
            resolution = resolutions.get(key)
            rows.append(
                {
                    **forecast,
                    "outcome": resolution["outcome"] if resolution else None,
                    "resolved_at": resolution["ts"] if resolution else None,
                }
            )
        return rows
