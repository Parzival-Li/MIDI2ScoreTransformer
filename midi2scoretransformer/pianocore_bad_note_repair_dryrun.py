"""
Dry-run analysis for conservative PianoCoRe bad-note repair.

This script does not write repaired training chunks. It answers:
  If we remove only performance notes that are unmatched/extra for a local
  score-time bucket, how many currently rejected buckets would pass filters?

Run from MIDI2ScoreTransformer/midi2scoretransformer so imports resolve.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from chunker_pianocore import (
    MultistreamTokenizer,
    _as_numpy_1d,
    _identity_key,
    _is_identity_trusted,
    _load_identity_cache_keys,
    _metadata_int,
    _score_onsets_from_npz,
    _score_onsets_from_xml,
    check_identity_mapping,
    extract_alignment_pairs,
    normalize_index_base,
    parse_streams,
    read_npz,
    resolve_path,
)


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def _build_alignment_maps(pairs: np.ndarray, n_xml: int, n_perf: int):
    matched = pairs[(pairs[:, 0] >= 0) & (pairs[:, 1] >= 0)]
    score_to_perf: dict[int, list[int]] = {}
    perf_to_score: dict[int, list[int]] = {}
    for s, p in matched:
        if 0 <= s < n_xml and 0 <= p < n_perf:
            score_to_perf.setdefault(int(s), []).append(int(p))
            perf_to_score.setdefault(int(p), []).append(int(s))
    return score_to_perf, perf_to_score


def _first_failure(metrics: Dict[str, Any], args) -> str:
    """Mirror the chunker_pianocore alignment_segment filter order."""
    if metrics["n_mxl"] < args.min_mxl_notes:
        return "min_mxl_notes"
    if not metrics["has_perf_matched"]:
        return "no_matched_perf"
    if metrics["n_midi"] < args.min_midi_notes:
        return "min_midi_notes"
    if args.max_midi_notes_per_chunk > 0 and metrics["n_midi"] > args.max_midi_notes_per_chunk:
        return "max_midi_notes_per_chunk"
    if args.max_midi_mxl_ratio > 0 and metrics["midi_mxl_ratio"] > args.max_midi_mxl_ratio:
        return "max_midi_mxl_ratio"
    if metrics["matched_score_ratio"] < args.min_matched_score_ratio:
        return "min_matched_score_ratio"
    if metrics["matched_perf_ratio"] < args.min_matched_perf_ratio:
        return "min_matched_perf_ratio"
    if metrics["monotonic_ratio"] < args.min_monotonic_ratio:
        return "min_monotonic_ratio"
    return "ok"


def _monotonic_ratio(values: list[int]) -> float:
    if len(values) <= 1:
        return 1.0
    return float(np.mean(np.diff(np.asarray(values)) >= 0))


def _chord_aware_monotonic_ratio(
    mxl_chunk: list[int],
    score_to_perf: dict[int, list[int]],
    score_onsets: Optional[np.ndarray],
    onset_eps: float,
) -> tuple[float, int, int]:
    """Ignore ordering differences inside notes that share one score onset."""
    if score_onsets is None:
        return 1.0, 0, 0

    intervals: list[tuple[int, int]] = []
    current_onset: Optional[float] = None
    current_perf: list[int] = []
    chord_group_count = 0
    chord_note_count = 0

    def flush_group() -> None:
        nonlocal current_onset, current_perf, chord_group_count, chord_note_count
        if not current_perf:
            current_onset = None
            return
        intervals.append((min(current_perf), max(current_perf)))
        if len(current_perf) > 1:
            chord_group_count += 1
            chord_note_count += len(current_perf)
        current_onset = None
        current_perf = []

    for s in mxl_chunk:
        ps = score_to_perf.get(s, [])
        if not ps:
            continue
        if s < 0 or s >= len(score_onsets) or not np.isfinite(score_onsets[s]):
            flush_group()
            intervals.append((min(ps), max(ps)))
            continue

        onset = float(score_onsets[s])
        if current_onset is not None and abs(onset - current_onset) <= onset_eps:
            current_perf.extend(int(p) for p in ps)
        else:
            flush_group()
            current_onset = onset
            current_perf = [int(p) for p in ps]
    flush_group()

    if len(intervals) <= 1:
        return 1.0, chord_group_count, chord_note_count
    ok = [intervals[i][0] >= intervals[i - 1][1] for i in range(1, len(intervals))]
    return float(np.mean(ok)), chord_group_count, chord_note_count


def _bucket_metrics(
    mxl_chunk: list[int],
    midi_chunk: list[int],
    score_to_perf: dict[int, list[int]],
    perf_to_score: dict[int, list[int]],
    score_onsets: Optional[np.ndarray] = None,
    chord_aware_monotonic: bool = False,
    chord_onset_eps: float = 1e-6,
) -> Dict[str, Any]:
    mxl_set = set(mxl_chunk)
    matched_score = [s for s in mxl_chunk if s in score_to_perf]
    matched_perf_in_chunk = [
        p for p in midi_chunk if any(s in mxl_set for s in perf_to_score.get(p, []))
    ]

    perf_for_score = []
    for s in mxl_chunk:
        ps = score_to_perf.get(s, [])
        if ps:
            perf_for_score.append(min(ps))
    raw_monotonic_ratio = _monotonic_ratio(perf_for_score)

    chord_aware_ratio = raw_monotonic_ratio
    chord_group_count = 0
    chord_note_count = 0
    if chord_aware_monotonic and score_onsets is not None:
        chord_aware_ratio, chord_group_count, chord_note_count = _chord_aware_monotonic_ratio(
            mxl_chunk, score_to_perf, score_onsets, chord_onset_eps
        )

    monotonic_ratio = chord_aware_ratio if chord_aware_monotonic else raw_monotonic_ratio

    return {
        "n_mxl": len(mxl_chunk),
        "n_midi": len(midi_chunk),
        "has_perf_matched": len(matched_perf_in_chunk) > 0,
        "matched_score_ratio": _safe_div(len(matched_score), len(mxl_chunk)),
        "matched_perf_ratio": _safe_div(len(matched_perf_in_chunk), len(midi_chunk)),
        "midi_mxl_ratio": _safe_div(len(midi_chunk), max(len(mxl_chunk), 1)),
        "monotonic_ratio": monotonic_ratio,
        "raw_monotonic_ratio": raw_monotonic_ratio,
        "chord_aware_monotonic_ratio": chord_aware_ratio,
        "chord_group_count": chord_group_count,
        "chord_note_count": chord_note_count,
        "matched_perf_in_chunk": matched_perf_in_chunk,
    }


def analyze_repair_for_piece(
    n_xml: int,
    n_perf: int,
    pairs: np.ndarray,
    score_onsets: np.ndarray,
    args,
) -> Dict[str, Any]:
    score_to_perf, perf_to_score = _build_alignment_maps(pairs, n_xml, n_perf)

    width = float(args.beat_quarter_len) * int(args.beats_per_chunk)
    finite_onsets = score_onsets[np.isfinite(score_onsets)]
    if len(finite_onsets) == 0:
        raise ValueError("No finite score onset values")

    start_bin = int(np.floor(float(finite_onsets.min()) / width))
    end_bin = int(np.floor(float(finite_onsets.max()) / width))

    summary = Counter()
    reason_counts = Counter()
    post_chord_reason_counts = Counter()
    rescued_reason_counts = Counter()
    chord_rescued_reason_counts = Counter()
    extra_rescued_reason_counts = Counter()
    candidate_reason_counts = Counter()
    deleted_notes_rescued = 0
    deleted_notes_candidates = 0
    repaired_good_notes = 0
    detail_rows: list[dict[str, Any]] = []
    repair_relevant_reasons = {
        "max_midi_notes_per_chunk",
        "max_midi_mxl_ratio",
        "min_matched_perf_ratio",
    }

    for b in range(start_bin, end_bin + 1):
        t0 = b * width
        t1 = t0 + width
        if b == end_bin:
            mxl_idx = np.where((score_onsets >= t0) & (score_onsets <= t1))[0]
        else:
            mxl_idx = np.where((score_onsets >= t0) & (score_onsets < t1))[0]
        mxl_chunk = [int(i) for i in mxl_idx if 0 <= int(i) < n_xml]
        if len(mxl_chunk) == 0:
            continue

        summary["nonempty_score_time_bins"] += 1
        perf_matched = sorted({p for s in mxl_chunk for p in score_to_perf.get(s, [])})
        if perf_matched:
            p0 = max(min(perf_matched) - args.context_perf_notes, 0)
            p1 = min(max(perf_matched) + args.context_perf_notes + 1, n_perf)
            midi_chunk = list(range(p0, p1))
        else:
            p0 = -1
            p1 = -1
            midi_chunk = []

        original_metrics = _bucket_metrics(mxl_chunk, midi_chunk, score_to_perf, perf_to_score)
        original_reason = _first_failure(original_metrics, args)
        reason_counts[original_reason] += 1
        if original_reason == "ok":
            summary["original_ok_bins"] += 1
            continue

        summary["original_bad_bins"] += 1
        if original_reason in repair_relevant_reasons:
            summary["repair_relevant_bad_bins"] += 1

        active_metrics = original_metrics
        active_reason = original_reason
        post_chord_reason = original_reason
        if args.chord_aware_monotonic:
            summary["chord_checked_bad_bins"] += 1
            chord_metrics = _bucket_metrics(
                mxl_chunk,
                midi_chunk,
                score_to_perf,
                perf_to_score,
                score_onsets=score_onsets,
                chord_aware_monotonic=True,
                chord_onset_eps=args.chord_onset_eps,
            )
            post_chord_reason = _first_failure(chord_metrics, args)
            post_chord_reason_counts[post_chord_reason] += 1
            if post_chord_reason == "ok":
                summary["chord_rescued_bins"] += 1
                summary["rescued_bins"] += 1
                rescued_reason_counts[original_reason] += 1
                chord_rescued_reason_counts[original_reason] += 1
                if args.write_bin_details:
                    detail_rows.append({
                        "score_bin": int(b),
                        "score_time_start": float(t0),
                        "score_time_end": float(t1),
                        "original_reason": original_reason,
                        "post_chord_reason": post_chord_reason,
                        "repaired_reason": post_chord_reason,
                        "repair_attempted": True,
                        "repair_type": "chord_monotonic",
                        "repairable": True,
                        "deleted_notes": 0,
                        "original_n_mxl": len(mxl_chunk),
                        "original_n_midi": len(midi_chunk),
                        "raw_monotonic_ratio": original_metrics["raw_monotonic_ratio"],
                        "chord_aware_monotonic_ratio": chord_metrics["chord_aware_monotonic_ratio"],
                        "chord_group_count": chord_metrics["chord_group_count"],
                        "chord_note_count": chord_metrics["chord_note_count"],
                    })
                continue
            summary["post_chord_bad_bins"] += 1
            active_metrics = chord_metrics
            active_reason = post_chord_reason

        if not midi_chunk:
            if args.write_bin_details:
                detail_rows.append({
                    "score_bin": int(b),
                    "original_reason": original_reason,
                    "post_chord_reason": post_chord_reason,
                    "repair_attempted": False,
                    "repairable": False,
                    "deleted_notes": 0,
                    "note": "no performance span",
                })
            continue

        matched_perf = sorted(set(active_metrics["matched_perf_in_chunk"]))
        deleted_notes = len(midi_chunk) - len(matched_perf)
        repair_attempted = deleted_notes > 0
        if not repair_attempted:
            if args.write_bin_details:
                detail_rows.append({
                    "score_bin": int(b),
                    "original_reason": original_reason,
                    "post_chord_reason": post_chord_reason,
                    "repair_attempted": False,
                    "repairable": False,
                    "deleted_notes": 0,
                    "original_n_midi": len(midi_chunk),
                    "original_n_mxl": len(mxl_chunk),
                })
            continue

        summary["repair_candidate_bins"] += 1
        if active_reason in repair_relevant_reasons:
            summary["repair_relevant_candidate_bins"] += 1
        candidate_reason_counts[active_reason] += 1
        deleted_notes_candidates += deleted_notes

        repaired_midi_chunk = matched_perf
        repaired_metrics = _bucket_metrics(
            mxl_chunk,
            repaired_midi_chunk,
            score_to_perf,
            perf_to_score,
            score_onsets=score_onsets,
            chord_aware_monotonic=args.chord_aware_monotonic,
            chord_onset_eps=args.chord_onset_eps,
        )
        repaired_reason = _first_failure(repaired_metrics, args)
        repaired_ok = repaired_reason == "ok"
        if repaired_ok:
            summary["rescued_bins"] += 1
            summary["extra_rescued_bins"] += 1
            if active_reason in repair_relevant_reasons:
                summary["repair_relevant_rescued_bins"] += 1
            rescued_reason_counts[original_reason] += 1
            extra_rescued_reason_counts[original_reason] += 1
            deleted_notes_rescued += deleted_notes
            repaired_good_notes += len(repaired_midi_chunk)

        if args.write_bin_details:
            detail_rows.append({
                "score_bin": int(b),
                "score_time_start": float(t0),
                "score_time_end": float(t1),
                "original_reason": original_reason,
                "post_chord_reason": post_chord_reason,
                "repaired_reason": repaired_reason,
                "repair_attempted": True,
                "repair_type": "delete_extra_perf",
                "repairable": bool(repaired_ok),
                "deleted_notes": int(deleted_notes),
                "delete_ratio": _safe_div(deleted_notes, len(midi_chunk)),
                "original_n_mxl": len(mxl_chunk),
                "original_n_midi": len(midi_chunk),
                "repaired_n_midi": len(repaired_midi_chunk),
                "original_matched_score_ratio": original_metrics["matched_score_ratio"],
                "original_matched_perf_ratio": original_metrics["matched_perf_ratio"],
                "repaired_matched_perf_ratio": repaired_metrics["matched_perf_ratio"],
                "original_midi_mxl_ratio": original_metrics["midi_mxl_ratio"],
                "repaired_midi_mxl_ratio": repaired_metrics["midi_mxl_ratio"],
                "monotonic_ratio": original_metrics["monotonic_ratio"],
                "raw_monotonic_ratio": original_metrics["raw_monotonic_ratio"],
                "chord_aware_monotonic_ratio": repaired_metrics["chord_aware_monotonic_ratio"],
                "chord_group_count": repaired_metrics["chord_group_count"],
                "chord_note_count": repaired_metrics["chord_note_count"],
            })

    return {
        "summary": dict(summary),
        "reason_counts": dict(reason_counts),
        "post_chord_reason_counts": dict(post_chord_reason_counts),
        "candidate_reason_counts": dict(candidate_reason_counts),
        "rescued_reason_counts": dict(rescued_reason_counts),
        "chord_rescued_reason_counts": dict(chord_rescued_reason_counts),
        "extra_rescued_reason_counts": dict(extra_rescued_reason_counts),
        "deleted_notes_candidates": int(deleted_notes_candidates),
        "deleted_notes_rescued": int(deleted_notes_rescued),
        "repaired_good_notes": int(repaired_good_notes),
        "details": detail_rows,
    }


def handle_row(row_dict: Dict[str, Any], args) -> Dict[str, Any]:
    row = pd.Series(row_dict)
    out = {
        "id": row.get("id", ""),
        "performance_midi_path": row.get("performance_midi_path", ""),
        "score_xml_path": row.get("score_xml_path", ""),
    }
    try:
        perf_midi_path = resolve_path(row["performance_midi_path"], args.pianocore_root)
        score_xml_path = resolve_path(row["score_xml_path"], args.pianocore_root)
        score_midi_path = resolve_path(row["score_midi_path"], args.pianocore_root)
        raw_alignment_path = str(row["raw_alignment_path"])

        identity_trusted = _is_identity_trusted(row, args)
        required_paths = [perf_midi_path, score_xml_path]
        if not identity_trusted:
            required_paths.append(score_midi_path)
        for p in required_paths:
            if not os.path.exists(p):
                raise FileNotFoundError(p)

        npz = read_npz(raw_alignment_path, args.pianocore_root, args.raw_alignments_zip)
        pairs_raw = extract_alignment_pairs(npz)

        score_xml = None
        if identity_trusted:
            n_xml = _metadata_int(row, "score_note_count")
            n_perf = _metadata_int(row, "performance_note_count")
            if args.pseudo_beat_source == "xml_absolute_onset" or n_xml is None:
                score_xml = MultistreamTokenizer.parse_mxl(score_xml_path)
                n_xml = int(len(score_xml["pitch"]))
            if n_perf is None:
                perf = MultistreamTokenizer.parse_midi(perf_midi_path)
                n_perf = int(len(perf["pitch"]))
            pairs = normalize_index_base(pairs_raw, n_xml, n_perf)
        else:
            perf, score_midi, score_xml = parse_streams(perf_midi_path, score_midi_path, score_xml_path)
            check = check_identity_mapping(perf, score_midi, score_xml, pairs_raw)
            if args.require_score_identity and not (check["score_len_equal"] and check["score_pitch_equal"]):
                raise RuntimeError(f"score MIDI != parse_mxl identity check failed: {check}")
            if not (check["perf_idx_in_bounds"] and check["score_idx_in_bounds"]):
                raise RuntimeError(f"alignment index out of bounds: {check}")
            n_xml = int(check["n_xml"])
            n_perf = int(check["n_perf"])
            pairs = check.pop("pairs_zero_based")

        if args.pseudo_beat_source == "xml_absolute_onset":
            if score_xml is None:
                score_xml = MultistreamTokenizer.parse_mxl(score_xml_path)
            score_onsets = _score_onsets_from_xml(score_xml)
        elif args.pseudo_beat_source == "npz_score_times":
            score_onsets = _score_onsets_from_npz(npz, n_xml)
        else:
            raise ValueError(f"Invalid pseudo_beat_source={args.pseudo_beat_source}")

        result = analyze_repair_for_piece(n_xml, n_perf, pairs, score_onsets, args)
        summary = result["summary"]

        out.update({
            "status": "ok",
            "error": "",
            "n_xml": int(n_xml),
            "n_perf": int(n_perf),
            "identity_reused": bool(identity_trusted),
            "nonempty_score_time_bins": summary.get("nonempty_score_time_bins", 0),
            "original_ok_bins": summary.get("original_ok_bins", 0),
            "original_bad_bins": summary.get("original_bad_bins", 0),
            "repair_relevant_bad_bins": summary.get("repair_relevant_bad_bins", 0),
            "chord_checked_bad_bins": summary.get("chord_checked_bad_bins", 0),
            "chord_rescued_bins": summary.get("chord_rescued_bins", 0),
            "post_chord_bad_bins": summary.get("post_chord_bad_bins", 0),
            "repair_candidate_bins": summary.get("repair_candidate_bins", 0),
            "repair_relevant_candidate_bins": summary.get("repair_relevant_candidate_bins", 0),
            "rescued_bins": summary.get("rescued_bins", 0),
            "extra_rescued_bins": summary.get("extra_rescued_bins", 0),
            "repair_relevant_rescued_bins": summary.get("repair_relevant_rescued_bins", 0),
            "deleted_notes_candidates": result["deleted_notes_candidates"],
            "deleted_notes_rescued": result["deleted_notes_rescued"],
            "repaired_good_notes": result["repaired_good_notes"],
            "reason_counts": json.dumps(result["reason_counts"], sort_keys=True),
            "post_chord_reason_counts": json.dumps(result["post_chord_reason_counts"], sort_keys=True),
            "candidate_reason_counts": json.dumps(result["candidate_reason_counts"], sort_keys=True),
            "rescued_reason_counts": json.dumps(result["rescued_reason_counts"], sort_keys=True),
            "chord_rescued_reason_counts": json.dumps(result["chord_rescued_reason_counts"], sort_keys=True),
            "extra_rescued_reason_counts": json.dumps(result["extra_rescued_reason_counts"], sort_keys=True),
        })
        if args.write_bin_details:
            out["_details"] = result["details"]
        return out
    except Exception as e:
        out.update({
            "status": "failed",
            "error": f"{type(e).__name__}: {str(e)}",
        })
        return out


def _sum_json_counter(rows: list[dict[str, Any]], column: str) -> Counter:
    total = Counter()
    for row in rows:
        value = row.get(column)
        if not value:
            continue
        if isinstance(value, str):
            value = json.loads(value)
        total.update({k: int(v) for k, v in value.items()})
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--pianocore-root", default="/mnt/ssd/hbli/datasets/pianocore/")
    ap.add_argument("--raw-alignments-zip", default="")
    ap.add_argument(
        "--out-dir",
        default="",
        help="Compatibility fallback used by identity-cache chunk path resolution.",
    )
    ap.add_argument("--out-summary", required=True)
    ap.add_argument("--out-rows", required=True)
    ap.add_argument("--out-bin-details", default="")
    ap.add_argument("--write-bin-details", action="store_true")
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--manifest-status-filter", default="ok")
    ap.add_argument("--split", default="", help="Optional split filter, e.g. train/test.")
    ap.add_argument("--pseudo-beat-source", choices=["xml_absolute_onset", "npz_score_times"], default="xml_absolute_onset")
    ap.add_argument("--beat-quarter-len", type=float, default=1.0)
    ap.add_argument("--beats-per-chunk", type=int, default=1)
    ap.add_argument("--min-mxl-notes", type=int, default=1)
    ap.add_argument("--min-midi-notes", type=int, default=1)
    ap.add_argument("--min-matched-score-ratio", type=float, default=0.7)
    ap.add_argument("--min-matched-perf-ratio", type=float, default=0.3)
    ap.add_argument("--context-perf-notes", type=int, default=0)
    ap.add_argument("--max-midi-mxl-ratio", type=float, default=4.0)
    ap.add_argument("--max-midi-notes-per-chunk", type=int, default=128)
    ap.add_argument("--min-monotonic-ratio", type=float, default=0.01)
    ap.add_argument(
        "--chord-aware-monotonic",
        action="store_true",
        help="Treat notes with the same XML absolute_onset as one chord when checking monotonicity.",
    )
    ap.add_argument(
        "--chord-onset-eps",
        type=float,
        default=1e-6,
        help="Tolerance in quarterLength units for grouping XML notes into one chord onset.",
    )
    ap.add_argument("--reuse-identity-cache", action="store_true")
    ap.add_argument("--identity-cache-manifest", default="")
    ap.add_argument("--identity-cache-chunk-suffix", default="")
    ap.add_argument("--identity-cache-chunk-location", choices=["next_to_midi", "out_dir"], default="next_to_midi")
    ap.add_argument("--identity-cache-out-dir", default="")
    ap.add_argument("--require-score-identity", action="store_true", default=True)
    ap.add_argument("--no-require-score-identity", dest="require_score_identity", action="store_false")
    args = ap.parse_args()

    if args.write_bin_details and not args.out_bin_details:
        raise ValueError("--write-bin-details requires --out-bin-details")

    args.identity_cache_keys = _load_identity_cache_keys(args.identity_cache_manifest)

    df = pd.read_csv(args.manifest, low_memory=False)
    required_cols = {"performance_midi_path", "score_xml_path", "score_midi_path", "raw_alignment_path"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"manifest missing columns: {missing}")
    if args.manifest_status_filter:
        if "chunk_status" not in df.columns:
            raise ValueError("--manifest-status-filter requires a chunk_status column")
        df = df[df["chunk_status"].astype(str).eq(args.manifest_status_filter)].copy()
    if args.split:
        split_col = "pm2s_split" if "pm2s_split" in df.columns else "split"
        if split_col not in df.columns:
            raise ValueError(f"--split requires a split column, but none was found")
        df = df[df[split_col].astype(str).eq(args.split)].copy()
    if args.max_rows and args.max_rows > 0:
        df = df.head(args.max_rows).copy()

    rows = df.to_dict(orient="records")
    results = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(handle_row)(row, args) for row in tqdm(rows, total=len(rows))
    )

    detail_rows = []
    if args.write_bin_details:
        for row in results:
            details = row.pop("_details", [])
            row_id = row.get("id", "")
            for detail in details:
                detail["id"] = row_id
                detail_rows.append(detail)

    out_rows = Path(args.out_rows)
    out_rows.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_rows, index=False)

    if args.write_bin_details:
        out_details = Path(args.out_bin_details)
        out_details.parent.mkdir(parents=True, exist_ok=True)
        with open(out_details, "w", encoding="utf-8") as f:
            for detail in detail_rows:
                f.write(json.dumps(detail, ensure_ascii=False) + "\n")

    ok_rows = [r for r in results if r.get("status") == "ok"]
    failed_rows = [r for r in results if r.get("status") != "ok"]

    totals = Counter()
    for row in ok_rows:
        for key in [
            "nonempty_score_time_bins", "original_ok_bins", "original_bad_bins",
            "repair_relevant_bad_bins", "chord_checked_bad_bins",
            "chord_rescued_bins", "post_chord_bad_bins", "repair_candidate_bins",
            "repair_relevant_candidate_bins", "rescued_bins", "extra_rescued_bins",
            "repair_relevant_rescued_bins", "deleted_notes_candidates",
            "deleted_notes_rescued", "repaired_good_notes",
        ]:
            totals[key] += int(row.get(key, 0) or 0)

    summary = {
        "manifest": args.manifest,
        "split": args.split,
        "rows_in_scope": len(rows),
        "ok_rows": len(ok_rows),
        "failed_rows": len(failed_rows),
        "totals": dict(totals),
        "rates": {
            "original_bad_ratio": _safe_div(totals["original_bad_bins"], totals["nonempty_score_time_bins"]),
            "repair_candidate_ratio_among_bad": _safe_div(totals["repair_candidate_bins"], totals["original_bad_bins"]),
            "repair_relevant_bad_ratio_among_bad": _safe_div(totals["repair_relevant_bad_bins"], totals["original_bad_bins"]),
            "repair_relevant_candidate_ratio_among_bad": _safe_div(
                totals["repair_relevant_candidate_bins"], totals["original_bad_bins"]
            ),
            "chord_rescued_ratio_among_bad": _safe_div(totals["chord_rescued_bins"], totals["original_bad_bins"]),
            "post_chord_bad_ratio_among_bad": _safe_div(totals["post_chord_bad_bins"], totals["original_bad_bins"]),
            "rescued_ratio_among_bad": _safe_div(totals["rescued_bins"], totals["original_bad_bins"]),
            "rescued_ratio_among_candidates": _safe_div(totals["extra_rescued_bins"], totals["repair_candidate_bins"]),
            "extra_rescued_ratio_among_candidates": _safe_div(
                totals["extra_rescued_bins"], totals["repair_candidate_bins"]
            ),
            "extra_rescued_ratio_among_post_chord_bad": _safe_div(
                totals["extra_rescued_bins"], totals["post_chord_bad_bins"]
            ),
            "repair_relevant_rescued_ratio_among_relevant_candidates": _safe_div(
                totals["repair_relevant_rescued_bins"], totals["repair_relevant_candidate_bins"]
            ),
            "avg_deleted_notes_per_candidate": _safe_div(totals["deleted_notes_candidates"], totals["repair_candidate_bins"]),
            "avg_deleted_notes_per_rescued": _safe_div(totals["deleted_notes_rescued"], totals["extra_rescued_bins"]),
            "delete_ratio_within_rescued_repaired_chunks": _safe_div(
                totals["deleted_notes_rescued"],
                totals["deleted_notes_rescued"] + totals["repaired_good_notes"],
            ),
        },
        "reason_counts": dict(_sum_json_counter(ok_rows, "reason_counts")),
        "post_chord_reason_counts": dict(_sum_json_counter(ok_rows, "post_chord_reason_counts")),
        "candidate_reason_counts": dict(_sum_json_counter(ok_rows, "candidate_reason_counts")),
        "rescued_reason_counts": dict(_sum_json_counter(ok_rows, "rescued_reason_counts")),
        "chord_rescued_reason_counts": dict(_sum_json_counter(ok_rows, "chord_rescued_reason_counts")),
        "extra_rescued_reason_counts": dict(_sum_json_counter(ok_rows, "extra_rescued_reason_counts")),
        "filter_params": {
            "beat_quarter_len": args.beat_quarter_len,
            "beats_per_chunk": args.beats_per_chunk,
            "min_mxl_notes": args.min_mxl_notes,
            "min_midi_notes": args.min_midi_notes,
            "min_matched_score_ratio": args.min_matched_score_ratio,
            "min_matched_perf_ratio": args.min_matched_perf_ratio,
            "context_perf_notes": args.context_perf_notes,
            "max_midi_mxl_ratio": args.max_midi_mxl_ratio,
            "max_midi_notes_per_chunk": args.max_midi_notes_per_chunk,
            "min_monotonic_ratio": args.min_monotonic_ratio,
            "chord_aware_monotonic": args.chord_aware_monotonic,
            "chord_onset_eps": args.chord_onset_eps,
        },
    }

    out_summary = Path(args.out_summary)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
