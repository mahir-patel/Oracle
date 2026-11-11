import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from forecaster.ledger import GENESIS_HASH, Ledger, canonical_json, record_hash
from forecaster.models import Forecast, Question, Sample

T0 = datetime(2026, 8, 1, tzinfo=UTC)


def make_forecast(market_id: str = "m1", final: float = 0.35, venue: str = "sim") -> Forecast:
    q = Question(
        venue=venue,
        market_id=market_id,
        title=f"Question {market_id}?",
        rules="Resolves YES if it happens.",
        close_time=datetime(2026, 9, 1, tzinfo=UTC),
        yes_bid=0.40,
        yes_ask=0.44,
    )
    return Forecast(
        question=q,
        samples=[Sample("outside_view", 0.3, "base rate", 0.25), Sample("inside_view", 0.4, "news")],
        raw_prob=0.34,
        calibrated_prob=0.33,
        final_prob=final,
        market_prob=q.mid,
        created_at=T0,
        model="fake",
        disagreement=0.31,
        latency_s=1.5,
    )


@pytest.fixture
def ledger(tmp_path: Path) -> Ledger:
    return Ledger(tmp_path / "data" / "ledger.jsonl")


def filled(ledger: Ledger) -> Ledger:
    ledger.append_forecast(make_forecast("m1"))
    ledger.append_forecast(make_forecast("m2"))
    ledger.append_resolution("sim", "m1", 1, datetime(2026, 9, 2, tzinfo=UTC))
    return ledger


# ---------- append ----------


def test_empty_ledger(ledger: Ledger) -> None:
    assert ledger.records() == []
    assert ledger.last_hash() == GENESIS_HASH
    assert ledger.verify()
    assert ledger.joined() == []


def test_append_chains_hashes(ledger: Ledger) -> None:
    first = ledger.append_forecast(make_forecast("m1"))
    second = ledger.append_forecast(make_forecast("m2"))
    assert first["prev_hash"] == GENESIS_HASH
    assert second["prev_hash"] == first["hash"]
    assert len(first["hash"]) == 64
    assert first["hash"] == record_hash(first)
    assert ledger.records() == [first, second]


def test_forecast_record_fields(ledger: Ledger) -> None:
    rec = ledger.append_forecast(make_forecast())
    assert rec["type"] == "forecast"
    assert rec["ts"] == "2026-08-01T00:00:00+00:00"
    for key in (
        "question", "samples", "raw_prob", "calibrated_prob", "final_prob", "market_prob",
        "disagreement", "model", "cost_usd", "bet", "latency_s",
    ):  # fmt: skip
        assert key in rec
    assert "created_at" not in rec
    assert rec["question"]["close_time"] == "2026-09-01T00:00:00+00:00"


def test_resolution_record(ledger: Ledger) -> None:
    rec = ledger.append_resolution("sim", "m1", 0, datetime(2026, 9, 2, 12, 0))
    assert rec == {
        "type": "resolution",
        "venue": "sim",
        "market_id": "m1",
        "outcome": 0,
        "ts": "2026-09-02T12:00:00+00:00",
        "prev_hash": GENESIS_HASH,
        "hash": rec["hash"],
    }
    with pytest.raises(ValueError):
        ledger.append_resolution("sim", "m1", 2, T0)


def test_lines_are_canonical_json(ledger: Ledger) -> None:
    filled(ledger)
    for line in ledger.path.read_text().splitlines():
        assert line == canonical_json(json.loads(line))


def test_append_rejects_prehashed_records(ledger: Ledger) -> None:
    with pytest.raises(ValueError):
        ledger.append({"type": "resolution", "hash": "x"})


def test_append_rejects_nan(ledger: Ledger) -> None:
    with pytest.raises(ValueError):
        ledger.append({"type": "forecast", "raw_prob": float("nan")})
    assert ledger.records() == []


def test_reopening_continues_the_chain(ledger: Ledger) -> None:
    filled(ledger)
    reopened = Ledger(ledger.path)
    rec = reopened.append_forecast(make_forecast("m3"))
    assert rec["prev_hash"] == ledger.records()[-2]["hash"]
    assert reopened.verify()


# ---------- verify / tamper detection ----------


def test_verify_passes_on_untouched_ledger(ledger: Ledger) -> None:
    assert filled(ledger).verify()


def test_tamper_one_byte_anywhere_is_detected(ledger: Ledger) -> None:
    """Change each single byte of the file in turn; every change must be caught."""
    filled(ledger)
    original = ledger.path.read_bytes()
    for i, byte in enumerate(original):
        tampered = bytearray(original)
        tampered[i] = ord("7") if byte != ord("7") else ord("8")
        ledger.path.write_bytes(bytes(tampered))
        assert not ledger.verify(), f"edit at byte {i} went undetected"
    ledger.path.write_bytes(original)
    assert ledger.verify()


def test_tamper_edited_probability_is_detected(ledger: Ledger) -> None:
    filled(ledger)
    text = ledger.path.read_text()
    assert '"final_prob":0.35' in text
    ledger.path.write_text(text.replace('"final_prob":0.35', '"final_prob":0.95', 1))
    assert not ledger.verify()


def test_tamper_rehashed_record_still_breaks_the_chain(ledger: Ledger) -> None:
    """Recomputing the edited record's own hash is not enough: the next record's
    prev_hash no longer matches."""
    filled(ledger)
    records = ledger.records()
    records[0]["final_prob"] = 0.95
    records[0]["hash"] = record_hash(records[0])
    ledger.path.write_text("".join(canonical_json(r) + "\n" for r in records))
    assert not ledger.verify()


def test_tamper_deleted_reordered_or_inserted_lines_are_detected(ledger: Ledger) -> None:
    filled(ledger)
    lines = ledger.path.read_text().splitlines(keepends=True)

    ledger.path.write_text("".join(lines[1:]))  # first record deleted
    assert not ledger.verify()
    ledger.path.write_text("".join([lines[0], lines[2]]))  # middle record deleted
    assert not ledger.verify()
    ledger.path.write_text("".join([lines[1], lines[0], lines[2]]))  # reordered
    assert not ledger.verify()
    ledger.path.write_text("".join([lines[0], lines[0], lines[1], lines[2]]))  # duplicated
    assert not ledger.verify()
    ledger.path.write_text("".join(lines) + "\n")  # stray blank line
    assert not ledger.verify()


def test_tamper_reformatted_line_is_detected(ledger: Ledger) -> None:
    """Same values, different bytes (pretty-printed) is still an edit."""
    filled(ledger)
    lines = ledger.path.read_text().splitlines()
    lines[0] = json.dumps(json.loads(lines[0]), sort_keys=True)
    ledger.path.write_text("\n".join(lines) + "\n")
    assert not ledger.verify()


# ---------- joined ----------


def test_joined_attaches_resolutions(ledger: Ledger) -> None:
    rows = filled(ledger).joined()
    assert [r["question"]["market_id"] for r in rows] == ["m1", "m2"]
    assert rows[0]["outcome"] == 1
    assert rows[0]["resolved_at"] == "2026-09-02T00:00:00+00:00"
    assert rows[1]["outcome"] is None
    assert rows[1]["resolved_at"] is None
    assert rows[0]["final_prob"] == 0.35


def test_joined_uses_latest_forecast_per_market(ledger: Ledger) -> None:
    ledger.append_forecast(make_forecast("m1", final=0.35))
    ledger.append_forecast(make_forecast("m2", final=0.50))
    ledger.append_forecast(make_forecast("m1", final=0.60))
    rows = ledger.joined()
    assert len(rows) == 2
    assert rows[0]["question"]["market_id"] == "m1"
    assert rows[0]["final_prob"] == 0.60


def test_joined_keys_on_venue_and_market_id(ledger: Ledger) -> None:
    ledger.append_forecast(make_forecast("same-id", venue="kalshi"))
    ledger.append_forecast(make_forecast("same-id", venue="polymarket"))
    ledger.append_resolution("polymarket", "same-id", 0, T0)
    rows = ledger.joined()
    assert [(r["question"]["venue"], r["outcome"]) for r in rows] == [
        ("kalshi", None),
        ("polymarket", 0),
    ]


def test_has_market(ledger: Ledger) -> None:
    filled(ledger)
    assert ledger.has_market("sim", "m1")
    assert not ledger.has_market("kalshi", "m1")
    assert not ledger.has_market("sim", "m9")
