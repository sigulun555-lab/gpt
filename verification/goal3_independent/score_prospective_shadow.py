#!/usr/bin/env python3
"""Research-only scorer for immutable GOAL3 k=6/k=10 shadow artifacts."""
from __future__ import annotations
import argparse, hashlib, json, math, platform, re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
import numpy as np

PROTOCOL = "1.4"
EXPECTED_SHADOW_BRANCH = "research/goal3-prospective-shadow-20261104-v2"
FIT_GRADIENT_MAX_ABS_TOL = 1e-6
EXPERIMENT = "GOAL3_PROSPECTIVE_SHADOW_K6_VS_K10"
KS = ("6", "10")
LEAGUES = ("J1", "J2", "J3")
FLOOR = 1e-12
REPS = 10_000
SEED = 20261105
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
ARTIFACT_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40,64}$")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    raw = Path(path).read_text(encoding="utf-8")
    doc = json.loads(raw)
    if not isinstance(doc, dict):
        raise ValueError(f"expected JSON object: {path}")
    return doc


def ts(value, name):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} missing")
    try:
        d = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as e:
        raise ValueError(f"{name} invalid") from e
    if d.tzinfo is None:
        raise ValueError(f"{name} must include timezone")
    return d.astimezone(timezone.utc)


def pvec(value, label):
    p = np.asarray(value, dtype=float)
    if p.shape != (4,) or not np.isfinite(p).all():
        raise ValueError(f"{label}: expected four finite probabilities")
    if np.any(p < 0) or np.any(p > 1) or abs(float(p.sum()) - 1) > 1e-9:
        raise ValueError(f"{label}: probability vector outside simplex")
    return p


def forecast_errors(f, round_no):
    e = []
    if f.get("experiment") != EXPERIMENT: e.append("wrong_experiment")
    if (not isinstance(f.get("round"), int) or isinstance(f.get("round"), bool)
        or f.get("round") != round_no): e.append("forecast_round_mismatch")
    if f.get("status") != "forecast_created_before_deadline": e.append("forecast_status_invalid")
    if f.get("productionImpact") != "none": e.append("production_impact_not_none")
    ev = f.get("evaluation")
    if not isinstance(ev, dict): ev = {}
    if ev.get("protocolVersion") != PROTOCOL: e.append("protocol_version_not_1.4")
    optimizer_meta = ev.get("optimizerDiagnostics")
    if (not isinstance(optimizer_meta, dict)
            or optimizer_meta.get("maxAbsGradientTolerance") != FIT_GRADIENT_MAX_ABS_TOL
            or optimizer_meta.get("successFlagAloneIsInsufficient") is not True):
        e.append("optimizer_diagnostic_protocol_missing")
    try:
        snap = ts(f.get("snapshotAvailableBy"), "snapshotAvailableBy")
    except ValueError:
        snap = None; e.append("missing_or_invalid_snapshotAvailableBy")
    try:
        asof = ts(f.get("forecastAsOf"), "forecastAsOf")
    except ValueError:
        asof = None; e.append("missing_or_invalid_forecastAsOf")
    try:
        recorded = ts(f.get("recordedAt"), "recordedAt")
    except ValueError:
        recorded = None; e.append("missing_or_invalid_recordedAt")
    try:
        deadline = ts(f.get("deadline"), "deadline")
    except ValueError:
        deadline = None; e.append("missing_or_invalid_deadline")
    if snap and asof and snap >= asof: e.append("snapshot_not_before_forecast")
    if asof and deadline and asof >= deadline: e.append("forecast_not_before_deadline")
    if asof and recorded and recorded <= asof: e.append("recorded_not_after_forecast")
    if recorded and deadline and recorded >= deadline: e.append("recorded_not_before_deadline")

    workflow = f.get("workflowRun")
    if not isinstance(workflow, dict):
        e.append("workflow_run_metadata_missing")
    else:
        for key in ("id", "attempt"):
            value = workflow.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                e.append(f"workflow_run_{key}_invalid")
        event = workflow.get("event")
        if event == "workflow_dispatch":
            pass
        elif event == "push":
            if workflow.get("headBranch") != EXPECTED_SHADOW_BRANCH:
                e.append("workflow_run_push_branch_not_research")
            message = workflow.get("headCommitMessage")
            if not isinstance(message, str) or "[goal3-shadow]" not in message:
                e.append("workflow_run_push_marker_missing")
        else:
            e.append("workflow_run_event_not_allowed")
        if not isinstance(workflow.get("headSha"), str) or not GIT_SHA_RE.fullmatch(workflow["headSha"]):
            e.append("workflow_run_head_sha_invalid")

    inp = f.get("inputs")
    if not isinstance(inp, dict): inp = {}
    for key in ("historySha256", "officialPredictionsSha256", "aliasesSha256"):
        if not isinstance(inp.get(key), str) or not SHA_RE.fullmatch(inp[key]):
            e.append(f"invalid_input_hash:{key}")
    if (not isinstance(inp.get("historyEligibleRows"), int)
        or isinstance(inp.get("historyEligibleRows"), bool)
        or inp.get("historyEligibleRows", 0) <= 0):
        e.append("invalid_historyEligibleRows")
    try:
        fetched = ts(inp.get("historyMaxDeclaredFetchedAt"), "historyMaxDeclaredFetchedAt")
        if snap and fetched > snap: e.append("history_fetch_after_snapshot")
    except ValueError:
        e.append("missing_or_invalid_historyMaxDeclaredFetchedAt")
    if snap:
        try:
            if ts(inp.get("snapshotAvailableBy"), "inputs.snapshotAvailableBy") != snap:
                e.append("snapshot_bound_mismatch")
        except ValueError:
            e.append("missing_input_snapshot_bound")
        times = {}
        for key in ("pipelineGeneratedAt", "pipelineDeclaredLastFetchedAt"):
            try:
                times[key] = ts(f.get(key), key)
                if times[key] > snap: e.append(f"{key}_after_snapshot")
            except ValueError:
                e.append(f"missing_or_invalid_{key}")
        if (len(times) == 2 and
            times["pipelineDeclaredLastFetchedAt"] < times["pipelineGeneratedAt"]):
            e.append("pipeline_fetch_before_generatedAt")

    code = f.get("code")
    if not isinstance(code, dict): code = {}
    for key in ("runnerSha256", "modelSha256", "snapshotPipelineSha256"):
        if not isinstance(code.get(key), str) or not SHA_RE.fullmatch(code[key]):
            e.append(f"invalid_code_hash:{key}")
    model = f.get("model")
    if not isinstance(model, dict): model = {}
    model_arms = model.get("arms")
    if (not isinstance(model_arms, list) or len(model_arms) != 2
        or set(map(str, model_arms)) != set(KS)):
        e.append("arms_not_exactly_k6_k10")
    if model.get("trainingWindow") != "2026_only": e.append("training_window_not_2026_only")
    if model.get("otherAdjustments") != "none; no market blend, Dixon-Coles or draw boost":
        e.append("unexpected_model_adjustments")

    fixtures = f.get("fixtures")
    if not isinstance(fixtures, list) or len(fixtures) != 3:
        return sorted(set(e + ["forecast_not_three_fixtures"]))
    numbers, pairs = set(), set()
    for fx in fixtures:
        if not isinstance(fx, dict):
            e.append("malformed_forecast_fixture"); continue
        n = fx.get("number")
        if not isinstance(n, int) or isinstance(n, bool) or n in numbers:
            e.append("invalid_or_duplicate_fixture_number")
        numbers.add(n)
        date, home, away = fx.get("date"), fx.get("home"), fx.get("away")
        if not isinstance(date, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
            e.append("invalid_fixture_date")
        if not isinstance(home, str) or not home or not isinstance(away, str) or not away or home == away:
            e.append("invalid_fixture_teams")
        if (date, home, away) in pairs: e.append("duplicate_fixture_pair")
        pairs.add((date, home, away))
        arms = fx.get("arms") or {}
        if not isinstance(arms, dict) or set(arms) != set(KS):
            e.append(f"fixture_{n}_missing_arm"); continue
        for k in KS:
            arm = arms[k] or {}
            arm_fit = arm.get("fit")
            if not isinstance(arm_fit, dict) or arm_fit.get("success") is not True:
                e.append(f"fixture_{n}_k{k}_fit_failed")
            for side in ("home", "away"):
                try: pvec(arm.get(f"{side}GoalProbabilities"), f"{n}/{k}/{side}")
                except (ValueError, TypeError): e.append(f"fixture_{n}_k{k}_{side}_probability_invalid")

    if numbers != {1, 2, 3}:
        e.append("fixture_numbers_not_exactly_1_2_3")

    fixture_dates = {
        fx["number"]: fx.get("date")
        for fx in fixtures
        if isinstance(fx, dict)
        and isinstance(fx.get("number"), int)
        and not isinstance(fx.get("number"), bool)
    }
    diags = f.get("fitDiagnostics")
    expected = {(n, k) for n in (1, 2, 3) for k in (6, 10)}
    actual = set()
    if not isinstance(diags, list) or len(diags) != 6:
        e.append("fit_diagnostics_not_six")
    else:
        for d in diags:
            if not isinstance(d, dict): e.append("malformed_fit_diagnostic"); continue
            fixture_no, k = d.get("fixtureNumber"), d.get("k")
            if (not isinstance(fixture_no, int) or isinstance(fixture_no, bool)
                or not isinstance(k, int) or isinstance(k, bool)):
                e.append("invalid_fit_diagnostic_key")
                continue
            key = (fixture_no, k)
            if key in actual: e.append("duplicate_fit_diagnostic")
            actual.add(key)
            if d.get("fitSuccess") is not True: e.append(f"fit_diagnostic_failed:{key}")
            if d.get("optimizerStatus") != 0:
                e.append(f"fit_optimizer_status_not_zero:{key}")
            max_gradient = d.get("maxAbsGradient")
            if (not isinstance(max_gradient, (int, float)) or isinstance(max_gradient, bool)
                    or not math.isfinite(max_gradient) or max_gradient < 0):
                e.append(f"fit_max_gradient_missing_or_invalid:{key}")
            elif max_gradient > FIT_GRADIENT_MAX_ABS_TOL:
                e.append(f"fit_gradient_exceeds_tolerance:{key}")
            gradient_l2 = d.get("gradientL2")
            if (not isinstance(gradient_l2, (int, float)) or isinstance(gradient_l2, bool)
                    or not math.isfinite(gradient_l2) or gradient_l2 < 0):
                e.append(f"fit_gradient_l2_missing_or_invalid:{key}")
            training_rows = d.get("trainingRows")
            if (not isinstance(training_rows, int) or isinstance(training_rows, bool)
                or training_rows <= 0):
                e.append(f"invalid_training_rows:{key}")
            if d.get("trainingCutoffExclusive") != fixture_dates.get(fixture_no):
                e.append(f"training_cutoff_mismatch:{key}")
        if actual != expected: e.append("fit_diagnostic_coverage_incomplete")
    return sorted(set(e))


def input_snapshot_errors(f, forecast_path):
    """Verify the byte-identical input files are present beside the forecast."""
    errors = []
    snapshot = f.get("inputSnapshots")
    if not isinstance(snapshot, dict):
        return ["input_snapshot_missing"]
    if snapshot.get("format") != "byte-for-byte copies of the raw UTF-8 input files":
        errors.append("input_snapshot_format_invalid")
    files = snapshot.get("files")
    if not isinstance(files, dict):
        return sorted(set(errors + ["input_snapshot_files_missing"]))
    expected = {
        "history": ("historySha256", "history.json"),
        "officialPredictions": ("officialPredictionsSha256", "official_predictions.json"),
        "aliases": ("aliasesSha256", "team_aliases.json"),
    }
    if set(files) != set(expected):
        errors.append("input_snapshot_file_set_invalid")
    inputs = f.get("inputs") if isinstance(f.get("inputs"), dict) else {}
    base = forecast_path.parent.resolve()
    expected_dir = f"{forecast_path.stem}-inputs"
    for label, (hash_key, filename) in expected.items():
        item = files.get(label)
        if not isinstance(item, dict):
            errors.append(f"input_snapshot_entry_missing:{label}")
            continue
        value = item.get("path")
        rel = Path(value) if isinstance(value, str) else None
        expected_rel = Path(expected_dir) / filename
        if rel is None or rel.is_absolute() or rel != expected_rel:
            errors.append(f"input_snapshot_path_invalid:{label}")
            continue
        resolved = (base / rel).resolve()
        if base not in resolved.parents:
            errors.append(f"input_snapshot_path_escapes:{label}")
            continue
        if not resolved.is_file():
            errors.append(f"input_snapshot_file_missing:{label}")
            continue
        actual = sha(resolved)
        if item.get("sha256") != actual or inputs.get(hash_key) != actual:
            errors.append(f"input_snapshot_sha_mismatch:{label}")
        size = item.get("sizeBytes")
        if not isinstance(size, int) or isinstance(size, bool) or size != resolved.stat().st_size:
            errors.append(f"input_snapshot_size_mismatch:{label}")
    return sorted(set(errors))


def artifact_evidence_errors(entry, round_no, forecast, forecast_sha):
    """Validate reviewer-transcribed GitHub run/artifact metadata."""
    errors = []
    evidence = entry.get("artifactEvidence")
    if not isinstance(evidence, dict):
        return ["artifact_evidence_missing"]
    workflow = forecast.get("workflowRun") if isinstance(forecast.get("workflowRun"), dict) else {}
    run_id, attempt = workflow.get("id"), workflow.get("attempt")
    if evidence.get("workflowRunId") != run_id:
        errors.append("artifact_evidence_workflow_run_id_mismatch")
    if evidence.get("workflowRunAttempt") != attempt:
        errors.append("artifact_evidence_workflow_run_attempt_mismatch")
    event = workflow.get("event")
    if evidence.get("workflowRunEvent") != event:
        errors.append("artifact_evidence_event_mismatch")
    if event == "workflow_dispatch":
        pass
    elif event == "push":
        if evidence.get("workflowRunHeadBranch") != workflow.get("headBranch"):
            errors.append("artifact_evidence_head_branch_mismatch")
        if evidence.get("workflowRunHeadCommitMessage") != workflow.get("headCommitMessage"):
            errors.append("artifact_evidence_head_commit_message_mismatch")
        if workflow.get("headBranch") != EXPECTED_SHADOW_BRANCH:
            errors.append("artifact_evidence_push_branch_not_research")
        message = workflow.get("headCommitMessage")
        if not isinstance(message, str) or "[goal3-shadow]" not in message:
            errors.append("artifact_evidence_push_marker_missing")
    else:
        errors.append("artifact_evidence_event_not_allowed")
    if evidence.get("workflowRunConclusion") != "success":
        errors.append("artifact_evidence_workflow_run_not_success")
    head_sha = evidence.get("workflowRunHeadSha")
    if not isinstance(head_sha, str) or not GIT_SHA_RE.fullmatch(head_sha) or head_sha != workflow.get("headSha"):
        errors.append("artifact_evidence_head_sha_mismatch")
    if not isinstance(run_id, int) or isinstance(run_id, bool) or run_id <= 0:
        errors.append("forecast_workflow_run_id_invalid")
    if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt <= 0:
        errors.append("forecast_workflow_run_attempt_invalid")
    artifact_id = evidence.get("artifactId")
    if not isinstance(artifact_id, int) or isinstance(artifact_id, bool) or artifact_id <= 0:
        errors.append("artifact_evidence_artifact_id_invalid")
    expected_name = (
        f"goal3-shadow-round-{round_no}-run-{run_id}-attempt-{attempt}"
        if isinstance(run_id, int) and isinstance(attempt, int) else None
    )
    if evidence.get("artifactName") != expected_name:
        errors.append("artifact_evidence_name_mismatch")
    if not isinstance(evidence.get("artifactDigest"), str) or not ARTIFACT_DIGEST_RE.fullmatch(evidence["artifactDigest"]):
        errors.append("artifact_evidence_digest_invalid")
    if evidence.get("forecastSha256") != forecast_sha:
        errors.append("artifact_evidence_forecast_sha_mismatch")
    if run_id and evidence.get("workflowRunUrl") != f"https://github.com/sigulun555-lab/gpt/actions/runs/{run_id}":
        errors.append("artifact_evidence_workflow_url_mismatch")
    if artifact_id and evidence.get("artifactApiUrl") != f"https://api.github.com/repos/sigulun555-lab/gpt/actions/artifacts/{artifact_id}":
        errors.append("artifact_evidence_api_url_mismatch")
    try:
        run_created = ts(evidence.get("workflowRunCreatedAt"), "workflowRunCreatedAt")
        deadline = ts(forecast.get("deadline"), "deadline")
        snapshot = ts(forecast.get("snapshotAvailableBy"), "snapshotAvailableBy")
        recorded = ts(forecast.get("recordedAt"), "recordedAt")
        if run_created >= deadline:
            errors.append("workflow_run_not_created_before_deadline")
        if run_created > snapshot:
            errors.append("workflow_run_after_input_snapshot")
    except ValueError:
        run_created = None
        deadline = None
        recorded = None
        errors.append("workflow_run_created_at_invalid")
    try:
        artifact_created = ts(evidence.get("artifactCreatedAt"), "artifactCreatedAt")
        if deadline and artifact_created >= deadline:
            errors.append("artifact_not_created_before_deadline")
        if run_created and artifact_created < run_created:
            errors.append("artifact_created_before_workflow_run")
        # GitHub artifact created_at is second-granularity; compare at that precision.
        if recorded and artifact_created < recorded.replace(microsecond=0):
            errors.append("artifact_created_before_forecast_record")
    except ValueError:
        errors.append("artifact_created_at_invalid")
    return sorted(set(errors))


def is_official_result_url(value):
    """Accept HTTPS URLs on official J.League result domains only."""
    try:
        parsed = urlparse(value)
        host = (parsed.hostname or "").lower().rstrip(".")
        if (parsed.scheme.lower() != "https" or not host
                or parsed.username is not None or parsed.password is not None):
            return False
        if parsed.port not in (None, 443):
            return False
        # A bare official homepage is not a result locator. Require at least
        # one non-empty path segment; reviewers must still verify page content.
        if not any(segment for segment in parsed.path.split("/")):
            return False
        return any(host == domain or host.endswith("." + domain)
                   for domain in ("jleague.jp", "j-league.or.jp"))
    except (TypeError, ValueError):
        return False


def result_errors(res, round_no, forecast):
    e = []
    if (not isinstance(res.get("round"), int) or isinstance(res.get("round"), bool)
        or res.get("round") != round_no): e.append("result_round_mismatch")
    if res.get("resultBasis") != "full_time_90_plus_stoppage":
        e.append("result_basis_not_90min_full_time")
    rfx = res.get("fixtures")
    ffx = forecast.get("fixtures")
    if not isinstance(rfx, list) or len(rfx) != 3:
        return e + ["result_not_three_fixtures"]
    expected = {x.get("number"): x for x in ffx if isinstance(x, dict)}
    seen = set()
    for x in rfx:
        if not isinstance(x, dict): e.append("malformed_result_fixture"); continue
        n = x.get("number")
        if not isinstance(n, int) or isinstance(n, bool) or n in seen:
            e.append("invalid_or_duplicate_result_fixture_number"); continue
        seen.add(n)
        fx = expected.get(n)
        if fx is None: e.append(f"unexpected_result_fixture:{n}"); continue
        for key in ("date", "home", "away"):
            if x.get(key) != fx.get(key): e.append(f"fixture_{n}_{key}_mismatch")
        if x.get("status") != "final": e.append(f"fixture_{n}_not_final")
        for key in ("fullTimeHomeGoals", "fullTimeAwayGoals"):
            v = x.get(key)
            if not isinstance(v, int) or isinstance(v, bool) or v < 0:
                e.append(f"fixture_{n}_invalid_{key}")
        if x.get("competition") not in LEAGUES: e.append(f"fixture_{n}_competition_invalid")
        if x.get("sourceType") != "official": e.append(f"fixture_{n}_source_not_official")
        source_url = x.get("sourceUrl")
        if not isinstance(source_url, str) or not source_url:
            e.append(f"fixture_{n}_source_url_invalid")
        elif not is_official_result_url(source_url):
            e.append(f"fixture_{n}_source_url_not_official_domain")
        try: ts(x.get("retrievedAt"), f"fixture_{n}_retrievedAt")
        except ValueError: e.append(f"fixture_{n}_retrievedAt_invalid")
        if x.get("publishedAt") is not None:
            try:
                if ts(x["publishedAt"], f"fixture_{n}_publishedAt") > ts(x["retrievedAt"], f"fixture_{n}_retrievedAt"):
                    e.append(f"fixture_{n}_published_after_retrieval")
            except ValueError: e.append(f"fixture_{n}_publishedAt_invalid")
    if seen != set(expected): e.append("result_fixture_coverage_incomplete")
    return sorted(set(e))


def failure_counts(f):
    failures, missing = {"6": 0, "10": 0}, {"6": 0, "10": 0}
    fixtures = f.get("fixtures") if isinstance(f.get("fixtures"), list) else []
    diags = {}
    for d in f.get("fitDiagnostics", []) if isinstance(f.get("fitDiagnostics"), list) else []:
        if isinstance(d, dict): diags[(d.get("fixtureNumber"), d.get("k"))] = d
    seen = set()
    for fx in fixtures:
        if not isinstance(fx, dict):
            for k in KS: missing[k] += 2
            continue
        n = fx.get("number"); seen.add(n)
        arms = fx.get("arms") if isinstance(fx.get("arms"), dict) else {}
        for k in KS:
            arm = arms.get(k)
            diag = diags.get((n, int(k)))
            if not isinstance(arm, dict):
                missing[k] += 2; continue
            arm_fit = arm.get("fit")
            ok = (isinstance(arm_fit, dict) and arm_fit.get("success") is True
                  and diag and diag.get("fitSuccess") is True)
            if not ok:
                failures[k] += 1; missing[k] += 2; continue
            for side in ("home", "away"):
                try: pvec(arm.get(f"{side}GoalProbabilities"), "probability")
                except (ValueError, TypeError): missing[k] += 1
    for n in (1, 2, 3):
        if n not in seen:
            for k in KS: missing[k] += 2
    return failures, missing


def score_one(entry, base):
    n = entry.get("round")
    if not isinstance(n, int) or isinstance(n, bool): raise ValueError("manifest round must be integer")
    paths = []
    for key in ("forecastPath", "resultsPath"):
        value = entry.get(key)
        p = Path(value) if isinstance(value, str) else None
        if p is None or p.is_absolute(): raise ValueError("manifest paths must be relative")
        resolved = (base / p).resolve()
        if base.resolve() not in resolved.parents: raise ValueError("manifest path escapes base directory")
        if not resolved.is_file(): raise FileNotFoundError(resolved)
        paths.append(resolved)
    fp, rp = paths
    fsha, rsha = sha(fp), sha(rp)
    if fsha != entry.get("forecastSha256"): raise ValueError(f"round {n}: forecast SHA-256 mismatch")
    if rsha != entry.get("resultsSha256"): raise ValueError(f"round {n}: results SHA-256 mismatch")
    f, res = read_json(fp), read_json(rp)
    failures, missing = failure_counts(f)
    snapshot_errors = input_snapshot_errors(f, fp)
    artifact_errors = artifact_evidence_errors(entry, n, f, fsha)
    errors = forecast_errors(f, n) + snapshot_errors + artifact_errors + result_errors(res, n, f)
    audit = {
        "round": n, "forecastSha256": fsha, "resultsSha256": rsha,
        "forecastTiming": {k: f.get(k) for k in ("snapshotAvailableBy", "forecastAsOf", "deadline")},
        "inputHashes": f.get("inputs", {}), "codeHashes": f.get("code", {}),
        "inputSnapshotFiles": (f.get("inputSnapshots", {}).get("files", {})
                               if isinstance(f.get("inputSnapshots"), dict) else {}),
        "artifactEvidence": entry.get("artifactEvidence"),
        "resultBasis": res.get("resultBasis"),
        "resultProvenance": [
            {k: x.get(k) for k in ("number", "competition", "sourceType", "sourceUrl", "publishedAt", "retrievedAt")}
            for x in (res.get("fixtures") if isinstance(res.get("fixtures"), list) else [])
            if isinstance(x, dict)
        ],
        "fitFailures": failures, "missingForecastTargets": missing,
        "exclusionReasons": sorted(set(errors)), "included": False,
    }
    if errors: return None, audit
    by_number = {x["number"]: x for x in res["fixtures"]}
    targets = []
    for fx in sorted(f["fixtures"], key=lambda x: x["number"]):
        rx = by_number[fx["number"]]
        for side, pk, rk in (
            ("home", "homeGoalProbabilities", "fullTimeHomeGoals"),
            ("away", "awayGoalProbabilities", "fullTimeAwayGoals"),
        ):
            y = min(rx[rk], 3)
            onehot = np.eye(4)[y]
            scores = {}
            for k in KS:
                p = pvec(fx["arms"][k][pk], f"round {n}/{k}/{side}")
                scores[k] = {
                    "brier": float(np.square(p - onehot).sum()),
                    "logloss": float(-math.log(max(float(p[y]), FLOOR))),
                }
            targets.append({
                "round": n, "fixtureNumber": fx["number"], "date": fx["date"],
                "competition": rx["competition"], "side": side,
                "brierDelta": scores["6"]["brier"] - scores["10"]["brier"],
                "loglossDelta": scores["6"]["logloss"] - scores["10"]["logloss"],
                "brier6": scores["6"]["brier"], "brier10": scores["10"]["brier"],
                "logloss6": scores["6"]["logloss"], "logloss10": scores["10"]["logloss"],
            })
    if len(targets) != 6: raise AssertionError("complete round must contain six paired targets")
    audit["included"] = True
    audit["includedMatches"] = 3
    audit["includedTargets"] = 6
    audit["pairedScoreMeans"] = {
        m: {k: float(np.mean([t[f"{m}{suffix}"] for t in targets]))
            for k, suffix in (("k6", "6"), ("k10", "10"), ("k6MinusK10", "Delta"))}
        for m in ("brier", "logloss")
    }
    return {"round": n, "targets": targets}, audit


def draws(n, reps, rng):
    if n < 1: raise ValueError("no bootstrap clusters")
    return rng.integers(0, n, size=(reps, n))


def boot_mean(records, metric, sampled):
    sums = np.asarray([sum(t[f"{metric}Delta"] for t in r["targets"]) for r in records])
    counts = np.asarray([len(r["targets"]) for r in records])
    return sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)


def score_manifest(manifest_path, replicates=REPS, seed=SEED):
    mp = Path(manifest_path).resolve()
    manifest = read_json(mp)
    if manifest.get("schemaVersion") != 1 or manifest.get("protocolVersion") != PROTOCOL:
        raise ValueError("manifest must declare schemaVersion=1 and protocolVersion=1.3")
    entries = manifest.get("rounds")
    if not isinstance(entries, list) or not entries: raise ValueError("manifest rounds missing")
    if any(not isinstance(x, dict) for x in entries): raise ValueError("manifest round entries must be objects")
    if len({x.get("round") for x in entries}) != len(entries): raise ValueError("duplicate manifest round")
    if replicates < 1000 or seed < 0: raise ValueError("invalid bootstrap settings")
    included, audits = [], []
    for entry in entries:
        record, audit = score_one(entry, mp.parent)
        audits.append(audit)
        if record is not None: included.append(record)
    included.sort(key=lambda x: x["round"])
    if len({x["round"] for x in included}) != len(included): raise ValueError("duplicate included round")
    fit = {k: sum(a["fitFailures"][k] for a in audits) for k in KS}
    missing = {k: sum(a["missingForecastTargets"][k] for a in audits) for k in KS}
    out = {
        "schemaVersion": 1, "protocolVersion": PROTOCOL, "experiment": EXPERIMENT,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "manifestSha256": sha(mp), "scoringScriptSha256": sha(Path(__file__)),
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "bootstrap": {
            "primaryCluster": "official GOAL3 round",
            "resampling": "paired complete-round bootstrap with replacement",
            "replicates": replicates, "rng": "NumPy PCG64", "seed": seed,
            "interval": "percentile; NumPy linear quantile",
            "sameRoundDrawsForBothMetrics": True,
            "sameDateDrawsForBothMetrics": True,
            "leagueSensitivity": "round-cluster bootstrap among rounds containing that league",
            "dateSensitivity": "date-cluster bootstrap retaining all matches and both team-sides on each date when multiple dates occur",
        },
        "cohort": {
            "manifestRounds": len(entries), "includedRounds": len(included),
            "includedMatches": 3 * len(included), "includedTeamSideTargets": 6 * len(included),
            "excludedRounds": len(audits) - len(included),
        },
        "fitFailures": fit, "missingForecastTargets": missing,
        "roundAudit": audits,
        "excludedRounds": [{"round": a["round"], "reasons": a["exclusionReasons"]}
                           for a in audits if a["exclusionReasons"]],
        "overall": {}, "leagueNonInferiority": {}, "sensitivity": {},
    }
    if included:
        rng = np.random.Generator(np.random.PCG64(seed))
        overall_draws = draws(len(included), replicates, rng)
        for metric in ("brier", "logloss"):
            point = float(np.mean([t[f"{metric}Delta"] for r in included for t in r["targets"]]))
            boot = boot_mean(included, metric, overall_draws)
            lo, hi = np.quantile(boot, [0.025, 0.975], method="linear")
            out["overall"][metric] = {
                "meanK6MinusK10": point, "twoSided95PercentileCI": [float(lo), float(hi)],
                "upper95BelowZero": bool(hi < 0),
            }
        for league in LEAGUES:
            subset = []
            for r in included:
                ts_ = [t for t in r["targets"] if t["competition"] == league]
                if ts_: subset.append({"round": r["round"], "targets": ts_})
            sampled = draws(len(subset), replicates, rng) if subset else None
            out["leagueNonInferiority"][league] = {}
            for metric, margin in (("brier", 0.01), ("logloss", 0.02)):
                vals = [t for r in subset for t in r["targets"]]
                if subset:
                    point = float(np.mean([t[f"{metric}Delta"] for t in vals]))
                    boot = boot_mean(subset, metric, sampled)
                    upper = float(np.quantile(boot, 0.90, method="linear"))
                else:
                    point, upper = None, None
                out["leagueNonInferiority"][league][metric] = {
                    "roundsWithLeagueTargets": len(subset), "teamSideTargets": len(vals),
                    "meanK6MinusK10": point, "oneSided90UpperBound": upper, "margin": margin,
                    "passes": bool(upper is not None and len(subset) >= 20 and upper < margin),
                }
        by_date = {}
        for r in included:
            for target in r["targets"]:
                by_date.setdefault(target["date"], []).append(target)
        date_records = [{"date": d, "targets": by_date[d]} for d in sorted(by_date)]
        if len(date_records) > 1:
            date_rng = np.random.Generator(np.random.PCG64(seed))
            date_draws = draws(len(date_records), replicates, date_rng)
            out["sensitivity"]["dateCluster"] = {
                "status": "computed", "dates": len(date_records), "replicates": replicates,
                "rng": "NumPy PCG64", "seed": seed,
                "interval": "two-sided 95% percentile; NumPy linear quantile", "results": {},
            }
            for metric in ("brier", "logloss"):
                point = float(np.mean([t[f"{metric}Delta"] for r in date_records for t in r["targets"]]))
                boot = boot_mean(date_records, metric, date_draws)
                lo, hi = np.quantile(boot, [0.025, 0.975], method="linear")
                out["sensitivity"]["dateCluster"]["results"][metric] = {
                    "meanK6MinusK10": point, "twoSided95PercentileCI": [float(lo), float(hi)],
                }
        else:
            out["sensitivity"]["dateCluster"] = {
                "status": "not_required_single_calendar_date", "dates": len(date_records),
            }
    enough = out["cohort"]["includedRounds"] >= 20 and out["cohort"]["includedMatches"] >= 60
    overall_ok = set(out["overall"]) == {"brier", "logloss"} and all(
        x["upper95BelowZero"] for x in out["overall"].values())
    league_ok = set(out["leagueNonInferiority"]) == set(LEAGUES) and all(
        out["leagueNonInferiority"][l][m]["passes"] for l in LEAGUES for m in ("brier", "logloss"))
    fit_ok = fit["6"] <= fit["10"] and missing["6"] <= missing["10"]
    included_audits = [a for a in audits if a["included"]]
    provenance_ok = len(included_audits) == len(included) and all(
        not a["exclusionReasons"] for a in included_audits)
    out["decision"] = {
        "interimDescriptiveThresholdMet": out["cohort"]["includedRounds"] >= 10 and out["cohort"]["includedMatches"] >= 30,
        "adoptionReviewEvidenceFloorMet": enough,
        "overallBothUpper95BelowZero": overall_ok,
        "allLeagueNonInferiorityGuardsPass": league_ok,
        "fitFailureAndMissingTargetGuardPass": fit_ok,
        "includedRoundProvenanceValid": provenance_ok,
        "eligibleForHumanAdoptionReview": bool(enough and overall_ok and league_ok and fit_ok and provenance_ok),
        "automaticProductionAdoption": False,
        "status": "REVIEW_CRITERIA_MET_REQUIRES_HUMAN_DECISION" if (
            enough and overall_ok and league_ok and fit_ok and provenance_ok) else "RESEARCH_ONLY_NO_ADOPTION",
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--replicates", type=int, default=REPS)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    if a.replicates != REPS or a.seed != SEED:
        raise ValueError("frozen CLI settings are 10000 replicates and seed 20261105")
    report = score_manifest(a.manifest, a.replicates, a.seed)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2); f.write("\n")
    print(json.dumps({"status": report["decision"]["status"],
                      "includedRounds": report["cohort"]["includedRounds"],
                      "excludedRounds": report["cohort"]["excludedRounds"], "out": str(out)}))


if __name__ == "__main__":
    main()
