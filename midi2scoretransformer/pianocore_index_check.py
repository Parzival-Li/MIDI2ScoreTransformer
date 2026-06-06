"""
Preflight index-space validation for PianoCoRe -> MIDI2ScoreTransformer.

This script checks whether PianoCoRe's score MIDI note sequence can be treated as the
same index space as MultistreamTokenizer.parse_mxl(score_xml_path), and whether raw
alignment indices are in bounds for MultistreamTokenizer.parse_midi(performance_midi_path).

Use this before full chunk generation.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from tokenizer_v2 import MultistreamTokenizer
from chunker_pianocore import read_npz, extract_alignment_pairs, normalize_index_base, resolve_path


def as_np(x):
    return np.asarray(x.cpu() if hasattr(x, "cpu") else x)


def lcp_equal(a: np.ndarray, b: np.ndarray) -> int:
    n = min(len(a), len(b))
    if n == 0:
        return 0
    eq = a[:n] == b[:n]
    bad = np.where(~eq)[0]
    return int(bad[0]) if len(bad) else int(n)


def handle_row(row: Dict[str, Any], args) -> Dict[str, Any]:
    out = dict(row)
    try:
        perf_midi_path = resolve_path(row["performance_midi_path"], args.pianocore_root)
        score_midi_path = resolve_path(row["score_midi_path"], args.pianocore_root)
        score_xml_path = resolve_path(row["score_xml_path"], args.pianocore_root)
        raw_alignment_path = str(row["raw_alignment_path"])

        for p in [perf_midi_path, score_midi_path, score_xml_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(p)

        perf = MultistreamTokenizer.parse_midi(perf_midi_path)
        score_midi = MultistreamTokenizer.parse_midi(score_midi_path)
        score_xml = MultistreamTokenizer.parse_mxl(score_xml_path)

        perf_pitch = as_np(perf["pitch"]).astype(int)
        score_midi_pitch = as_np(score_midi["pitch"]).astype(int)
        score_xml_pitch = as_np(score_xml["pitch"]).astype(int)

        npz = read_npz(raw_alignment_path, args.pianocore_root, args.raw_alignments_zip)
        pairs = extract_alignment_pairs(npz)
        pairs = normalize_index_base(pairs, len(score_midi_pitch), len(perf_pitch))
        matched = pairs[(pairs[:, 0] >= 0) & (pairs[:, 1] >= 0)]

        out.update({
            "status": "ok",
            "error": "",
            "n_perf": len(perf_pitch),
            "n_score_midi": len(score_midi_pitch),
            "n_score_xml": len(score_xml_pitch),
            "score_len_equal": len(score_midi_pitch) == len(score_xml_pitch),
            "score_pitch_equal": bool(np.array_equal(score_midi_pitch, score_xml_pitch)),
            "score_pitch_lcp": lcp_equal(score_midi_pitch, score_xml_pitch),
            "score_pitch_lcp_ratio": lcp_equal(score_midi_pitch, score_xml_pitch) / max(min(len(score_midi_pitch), len(score_xml_pitch)), 1),
            "n_alignment_pairs": len(pairs),
            "n_matched_pairs": len(matched),
            "max_score_idx": int(matched[:, 0].max()) if len(matched) else -1,
            "max_perf_idx": int(matched[:, 1].max()) if len(matched) else -1,
            "score_idx_in_bounds": bool(len(matched) == 0 or matched[:, 0].max() < len(score_midi_pitch)),
            "perf_idx_in_bounds": bool(len(matched) == 0 or matched[:, 1].max() < len(perf_pitch)),
            "npz_keys": ",".join(npz.keys()),
        })
        return out
    except Exception as e:
        out.update({
            "status": "failed",
            "error": f"{type(e).__name__}: {str(e)}",
        })
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--pianocore-root", default="/mnt/ssd/hbli/datasets/pianocore/")
    ap.add_argument("--raw-alignments-zip", default="")
    ap.add_argument("--out-csv", default="/mnt/ssd/hbli/datasets/pianocore/pianocore_index_check.csv")
    ap.add_argument("--max-rows", type=int, default=500)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-jobs", type=int, default=8)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest, low_memory=False)
    required = {"performance_midi_path", "score_midi_path", "score_xml_path", "raw_alignment_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"manifest missing columns: {missing}")

    if args.max_rows and args.max_rows > 0 and len(df) > args.max_rows:
        if args.sample:
            df = df.sample(args.max_rows, random_state=args.seed).reset_index(drop=True)
        else:
            df = df.head(args.max_rows).copy()

    rows = df.to_dict("records")
    results = Parallel(n_jobs=min(args.n_jobs, len(rows)), verbose=10)(
        delayed(handle_row)(row, args) for row in rows
    )
    out_df = pd.DataFrame(results)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print("saved:", args.out_csv)
    print(out_df["status"].value_counts(dropna=False))
    if "score_pitch_equal" in out_df:
        print("score identity pass rate:", out_df["score_pitch_equal"].mean())
    if "perf_idx_in_bounds" in out_df:
        print("perf index in bounds rate:", out_df["perf_idx_in_bounds"].mean())


if __name__ == "__main__":
    main()
