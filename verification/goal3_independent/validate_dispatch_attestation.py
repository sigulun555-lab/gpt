#!/usr/bin/env python3
"""Fail-closed binding of a human-reviewed official GOAL3 fixture snapshot to the fetched prediction payload."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

OFFICIAL_URL = (
    "https://sp.toto-dream.com/dcs/subos/screen/ss01/sssl014/"
    "PGSSSL01401InittotoSP.form?commodityId=02&gameAssortment=1"
)
FIXTURE_KEYS = {"number", "date", "home", "away"}
ATTESTATION_KEYS = {
    "officialUrl", "marketType", "round", "deadline", "fixtures", "reviewedAt"
}
SNAPSHOT_IDENTITY_KEYS = ("officialUrl", "marketType", "round", "deadline", "fixtures")


def snapshot_identity(value: dict[str, Any]) -> dict[str, Any]:
    """Return only official snapshot fields; review/validation timestamps are provenance, not identity."""
    return {key: value[key] for key in SNAPSHOT_IDENTITY_KEYS}




class AttestationError(ValueError):
    pass


def parse_time(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise AttestationError(f"{label} must be a non-empty ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AttestationError(f"{label} is not valid ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise AttestationError(f"{label} must include a timezone")
    return parsed.astimezone(timezone.utc)


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def normalized_attestation(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != ATTESTATION_KEYS:
        raise AttestationError(
            "attestation must contain exactly officialUrl, marketType, round, "
            "deadline, fixtures, reviewedAt"
        )
    if value["officialUrl"] != OFFICIAL_URL:
        raise AttestationError("attested officialUrl does not match the pinned official GOAL3 URL")
    if value["marketType"] != "totoGOAL3":
        raise AttestationError("attested marketType must be totoGOAL3")
    if isinstance(value["round"], bool) or not isinstance(value["round"], int) or value["round"] <= 0:
        raise AttestationError("attested round must be a positive integer")
    parse_time(value["deadline"], "attested deadline")
    parse_time(value["reviewedAt"], "reviewedAt")
    fixtures = value["fixtures"]
    if not isinstance(fixtures, list) or len(fixtures) != 3:
        raise AttestationError("attested fixtures must contain exactly three matches")
    normalized = []
    for index, fixture in enumerate(fixtures, start=1):
        if not isinstance(fixture, dict) or set(fixture) != FIXTURE_KEYS:
            raise AttestationError(f"fixture {index} must contain exactly number/date/home/away")
        if isinstance(fixture["number"], bool) or not isinstance(fixture["number"], int) or fixture["number"] != index:
            raise AttestationError("attested fixture numbers must be integer values 1, 2, 3 in order")
        if not all(isinstance(fixture[k], str) and fixture[k].strip() for k in ("date", "home", "away")):
            raise AttestationError(f"fixture {index} date/home/away must be non-empty strings")
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", fixture["date"]):
            raise AttestationError(f"fixture {index} date must be YYYY-MM-DD")
        try:
            date.fromisoformat(fixture["date"])
        except ValueError as exc:
            raise AttestationError(f"fixture {index} date is not a real calendar date") from exc
        if fixture["home"] == fixture["away"]:
            raise AttestationError(f"fixture {index} home and away must differ")
        normalized.append({
            "number": index,
            "date": fixture["date"],
            "home": fixture["home"],
            "away": fixture["away"],
        })
    return {
        "officialUrl": value["officialUrl"],
        "marketType": value["marketType"],
        "round": value["round"],
        "deadline": value["deadline"],
        "fixtures": normalized,
    }


def observed_snapshot(payload: Any, now: datetime) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise AttestationError("predictions payload must be a JSON object")
    if payload.get("status") != "predictions_generated" or payload.get("predictionsGenerated") is not True:
        raise AttestationError("pipeline did not confirm a live predictions_generated payload")
    if payload.get("totoTargetsConfirmed") is not True:
        raise AttestationError("pipeline did not confirm official toto targets")
    source = payload.get("totoSource")
    if not isinstance(source, dict):
        raise AttestationError("totoSource metadata is missing")
    if source.get("url") != OFFICIAL_URL or source.get("marketType") != "totoGOAL3":
        raise AttestationError("fetched target source is not the pinned official GOAL3 page")
    if source.get("round") != payload.get("round") or source.get("deadline") != payload.get("deadline"):
        raise AttestationError("payload and totoSource round/deadline disagree")
    if source.get("fixtureCount") != 3 or source.get("teamTargetCount") != 6:
        raise AttestationError("official GOAL3 source must report 3 fixtures and 6 team targets")
    if isinstance(payload.get("round"), bool) or not isinstance(payload.get("round"), int):
        raise AttestationError("fetched round must be an integer")
    parse_time(payload.get("deadline"), "fetched deadline")
    fetched_at = parse_time(source.get("lastFetchedAt"), "totoSource.lastFetchedAt")
    generated_at = parse_time(payload.get("generatedAt"), "generatedAt")
    if fetched_at > now + timedelta(minutes=1) or generated_at > now + timedelta(minutes=1):
        raise AttestationError("pipeline timestamps are in the future")
    if now - fetched_at > timedelta(minutes=10) or now - generated_at > timedelta(minutes=10):
        raise AttestationError("official fixture snapshot is older than 10 minutes")
    rows = payload.get("predictions")
    if not isinstance(rows, list) or len(rows) != 3:
        raise AttestationError("predictions must contain exactly three GOAL3 fixtures")
    fixtures = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise AttestationError(f"prediction fixture {index} is not an object")
        if (isinstance(row.get("number"), bool) or not isinstance(row.get("number"), int)
            or row.get("number") != index or row.get("marketType") != "totoGOAL3"):
            raise AttestationError(f"prediction fixture {index} numbering/marketType mismatch")
        if row.get("totoRound") != payload["round"] or row.get("totoTargetConfirmed") is not True:
            raise AttestationError(f"prediction fixture {index} round/confirmation mismatch")
        if row.get("sourceUrl") != OFFICIAL_URL:
            raise AttestationError(f"prediction fixture {index} source URL mismatch")
        fixtures.append({
            "number": index,
            "date": row.get("date"),
            "home": row.get("home"),
            "away": row.get("away"),
        })
    observed = {
        "officialUrl": source["url"],
        "marketType": source["marketType"],
        "round": payload["round"],
        "deadline": payload["deadline"],
        "fixtures": fixtures,
    }
    # Apply the same strict shape and value checks used for the human attestation.
    return normalized_attestation({**observed, "reviewedAt": datetime.now(timezone.utc).isoformat()})


def validate(attested_raw: Any, payload: Any, now: datetime | None = None) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None or now.utcoffset() is None:
        raise AttestationError("validation clock must be timezone-aware")
    attested = normalized_attestation(attested_raw)
    reviewed_at = parse_time(attested_raw["reviewedAt"], "reviewedAt")
    now_utc = now.astimezone(timezone.utc)
    if reviewed_at > now_utc + timedelta(minutes=1):
        raise AttestationError("reviewedAt is more than one minute in the future")
    if now_utc - reviewed_at > timedelta(minutes=10):
        raise AttestationError("official fixture review is older than 10 minutes; review and dispatch again")
    observed = observed_snapshot(payload, now_utc)
    if parse_time(observed["deadline"], "fetched deadline") <= now_utc:
        raise AttestationError("fetched official deadline has already passed")
    attested_identity = snapshot_identity(attested)
    observed_identity = snapshot_identity(observed)
    if canonical(attested_identity) != canonical(observed_identity):
        raise AttestationError("human-reviewed fixture snapshot does not exactly match fetched official snapshot")
    # Fingerprint only the official round/deadline/fixtures. reviewedAt and
    # validatedAt are provenance timestamps and must not alter snapshot identity.
    digest = hashlib.sha256(canonical(observed_identity)).hexdigest()
    source_fetched_at = payload["totoSource"]["lastFetchedAt"]
    return {
        "schemaVersion": 1,
        "experiment": "GOAL3_PROSPECTIVE_SHADOW_K6_VS_K10",
        "reviewedAt": attested_raw["reviewedAt"],
        "validatedAt": now_utc.isoformat().replace("+00:00", "Z"),
        "officialUrl": OFFICIAL_URL,
        "attestedSnapshot": attested,
        "observedSnapshot": observed,
        "snapshotFingerprintSha256": digest,
        "snapshotFetchedAt": source_fetched_at,
        "match": True,
    }


def self_test() -> None:
    now = datetime(2026, 11, 6, 1, 0, tzinfo=timezone.utc)
    # Deliberately differ from validation time: a valid human review need not
    # have the exact same timestamp as the runner's observed-snapshot record.
    reviewed = (now - timedelta(minutes=2)).isoformat().replace("+00:00", "Z")
    fixtures = [
        {"number": 1, "date": "2026-11-07", "home": "清水", "away": "東京Ｖ"},
        {"number": 2, "date": "2026-11-07", "home": "福岡", "away": "長崎"},
        {"number": 3, "date": "2026-11-07", "home": "柏", "away": "川崎Ｆ"},
    ]
    attested = {
        "officialUrl": OFFICIAL_URL, "marketType": "totoGOAL3", "round": 1670,
        "deadline": "2026-11-07T12:50:00+09:00", "fixtures": fixtures, "reviewedAt": reviewed,
    }
    payload = {
        "status": "predictions_generated", "predictionsGenerated": True,
        "totoTargetsConfirmed": True, "round": 1670,
        "deadline": "2026-11-07T12:50:00+09:00", "generatedAt": reviewed,
        "totoSource": {
            "url": OFFICIAL_URL, "marketType": "totoGOAL3", "round": 1670,
            "deadline": "2026-11-07T12:50:00+09:00", "fixtureCount": 3,
            "teamTargetCount": 6, "lastFetchedAt": reviewed,
        },
        "predictions": [
            {"number": f["number"], "marketType": "totoGOAL3", "date": f["date"],
             "home": f["home"], "away": f["away"], "sourceUrl": OFFICIAL_URL,
             "totoRound": 1670, "totoTargetConfirmed": True}
            for f in fixtures
        ],
    }
    result = validate(attested, payload, now)
    assert result["match"] is True and len(result["snapshotFingerprintSha256"]) == 64
    assert result["reviewedAt"] != result["validatedAt"], (
        "human review time must remain distinct from runner validation time"
    )
    assert "reviewedAt" not in result["observedSnapshot"], (
        "observed snapshot identity must not synthesize a human review timestamp"
    )
    expected_fingerprint = hashlib.sha256(canonical(snapshot_identity(result["observedSnapshot"]))).hexdigest()
    assert result["snapshotFingerprintSha256"] == expected_fingerprint
    mismatch = json.loads(json.dumps(payload, ensure_ascii=False))
    mismatch["predictions"][1]["home"] = "別チーム"
    try:
        validate(attested, mismatch, now)
    except AttestationError:
        pass
    else:
        raise AssertionError("fixture mismatch must fail closed")
    stale = dict(attested, reviewedAt=(now - timedelta(minutes=11)).isoformat())
    try:
        validate(stale, payload, now)
    except AttestationError:
        pass
    else:
        raise AssertionError("stale human review must fail closed")
    bad_url = dict(attested, officialUrl="https://example.com/")
    try:
        validate(bad_url, payload, now)
    except AttestationError:
        pass
    else:
        raise AssertionError("non-official URL must fail closed")
    print("PASS GOAL3 dispatch attestation self-test: exact match, mismatch, stale review, URL pin")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="data/predictions.json")
    parser.add_argument("--out", default="artifacts/fixture-attestation.json")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    raw = os.environ.get("ATTESTED_FIXTURE_SNAPSHOT", "")
    if not raw:
        raise AttestationError("ATTESTED_FIXTURE_SNAPSHOT workflow input is required")
    attested = json.loads(raw)
    payload = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    record = validate(attested, payload)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "match": record["match"],
        "round": record["observedSnapshot"]["round"],
        "fingerprintSha256": record["snapshotFingerprintSha256"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AttestationError, json.JSONDecodeError, OSError) as exc:
        print(f"DISPATCH_ATTESTATION_FAILED: {exc}", file=sys.stderr)
        raise SystemExit(2)
