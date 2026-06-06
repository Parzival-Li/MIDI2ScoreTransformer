"""
Build MIDI2ScoreTransformer-compatible chunk jsons for PianoCoRe paired data.

Core idea:
  PianoCoRe raw_alignment gives score-MIDI-note-index <-> performance-MIDI-note-index.
  MIDI2ScoreTransformer Dataset needs chunk json indices into:
    - MultistreamTokenizer.parse_midi(performance_midi_path)
    - MultistreamTokenizer.parse_mxl(score_xml_path)

This first version is conservative:
  1) require score MIDI parsed by PM2S to match MusicXML parsed by PM2S in length and pitch order;
  2) assume score_midi_idx == parse_mxl_idx and perf_idx == parse_midi_idx;
  3) generate note-count chunks on the target side and use raw_alignment to choose the performance range.

Rows failing the checks are skipped and logged.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

# Adjust this import if your repo uses a different package layout.
from tokenizer_v2 import MultistreamTokenizer


def _path_variants(path_like: str) -> list[str]:
    """
    Generate likely relative path variants for PianoCoRe files.

    Your extracted raw MIDI/XML files live under:
        <pianocore_root>/PianoCoRe/raw/...

    However metadata.csv may store paths without this prefix, e.g.:
        Chopin_Frédéric/Ballade_No.3_in_A_flat_major,_Op.47/Aria_189590_0.mid

    This helper keeps the scripts robust to all common forms:
        <rel>
        raw/<rel>
        PianoCoRe/<rel>
        PianoCoRe/raw/<rel>
    """
    s = str(path_like).replace("\\", "/").strip().lstrip("./")
    if not s or s.lower() in {"nan", "none", "null"}:
        return []

    variants: list[str] = []

    def add(x: str):
        x = x.replace("\\", "/").strip().lstrip("./")
        if x and x not in variants:
            variants.append(x)

    add(s)

    # Remove known leading prefixes, then rebuild common alternatives.
    base = s
    for prefix in ("PianoCoRe/raw/", "PianoCoRe/", "raw/"):
        if base.startswith(prefix):
            base = base[len(prefix):]
            break

    add(base)
    add(f"raw/{base}")
    add(f"PianoCoRe/{base}")
    add(f"PianoCoRe/raw/{base}")

    return variants


def resolve_path(path_like: str, root: str) -> str:
    """
    Resolve metadata path to an existing file when possible.

    Priority:
      1) absolute path if it exists
      2) <root>/<variant>
      3) fallback to <root>/PianoCoRe/raw/<base>, so FileNotFoundError shows
         the path that should exist under the current extracted layout.
    """
    raw = str(path_like)
    p = Path(raw)
    if p.is_absolute():
        if p.exists():
            return str(p)
        return str(p)

    root_p = Path(root)
    for v in _path_variants(raw):
        cand = root_p / v
        if cand.exists():
            return str(cand)

    variants = _path_variants(raw)
    if variants:
        base = variants[0]
        for prefix in ("PianoCoRe/raw/", "PianoCoRe/", "raw/"):
            if base.startswith(prefix):
                base = base[len(prefix):]
                break
        return str(root_p / "PianoCoRe" / "raw" / base)

    return str(root_p / raw)


def _candidate_zip_names(path_like: str) -> list[str]:
    """
    Candidate member names for raw alignment npz inside the zip.
    This mirrors _path_variants because the zip may include PianoCoRe/raw/
    or may store paths without a leading prefix.
    """
    return _path_variants(path_like)


def read_npz(path_like: str, root: str, raw_alignments_zip: str = "") -> Dict[str, Any]:
    """Read an npz either from extracted files or directly from PianoCoRe raw-alignments zip."""
    path = Path(resolve_path(path_like, root))
    if path.exists():
        with np.load(path, allow_pickle=True) as z:
            return {k: z[k] for k in z.files}

    if raw_alignments_zip:
        with zipfile.ZipFile(raw_alignments_zip, "r") as zf:
            for cand in _candidate_zip_names(path_like):
                try:
                    data = zf.read(cand)
                    with np.load(io.BytesIO(data), allow_pickle=True) as z:
                        return {k: z[k] for k in z.files}
                except KeyError:
                    continue

    raise FileNotFoundError(f"Cannot find raw alignment npz: {path_like}")


def _as_int_array(x: Any) -> Optional[np.ndarray]:
    arr = np.asarray(x)
    if arr.dtype.kind not in "iu":
        return None
    return arr.astype(np.int64, copy=False)


def extract_alignment_pairs(npz: Dict[str, Any]) -> np.ndarray:
    """
    Return int64 array of shape [N, 2]: [score_idx, perf_idx].

    This function is intentionally permissive because alignment npz key names can vary.
    It supports:
      - arrays named like alignment/matches/matched_indices with shape [N,2]
      - separate score/performance index arrays
      - structured arrays with fields containing score/performance and idx/index
    """
    # 1) Common Nx2 arrays.
    preferred = [
        "alignment", "alignments", "matched", "matches", "matched_indices",
        "note_alignment", "alignment_idx", "alignment_indices",
    ]
    for key in preferred + list(npz.keys()):
        if key not in npz:
            continue
        arr = _as_int_array(npz[key])
        if arr is not None and arr.ndim == 2 and arr.shape[1] >= 2:
            pair = arr[:, :2]
            if np.nanmax(pair) >= 0:
                return pair.astype(np.int64)

    # 2) Separate score/performance index arrays.
    score_keys = [k for k in npz.keys() if "score" in k.lower() and ("idx" in k.lower() or "index" in k.lower())]
    perf_keys = [k for k in npz.keys() if ("perf" in k.lower() or "performance" in k.lower()) and ("idx" in k.lower() or "index" in k.lower())]
    for sk in score_keys:
        for pk in perf_keys:
            s = _as_int_array(npz[sk])
            p = _as_int_array(npz[pk])
            if s is not None and p is not None and s.ndim == 1 and p.ndim == 1 and len(s) == len(p):
                return np.stack([s, p], axis=1).astype(np.int64)

    # 3) Structured arrays.
    for key, arr in npz.items():
        arr = np.asarray(arr)
        if arr.dtype.names:
            names = list(arr.dtype.names)
            score_name = None
            perf_name = None
            for n in names:
                nl = n.lower()
                if score_name is None and "score" in nl and ("idx" in nl or "index" in nl):
                    score_name = n
                if perf_name is None and ("perf" in nl or "performance" in nl) and ("idx" in nl or "index" in nl):
                    perf_name = n
            if score_name and perf_name:
                return np.stack([arr[score_name], arr[perf_name]], axis=1).astype(np.int64)

    raise ValueError(f"Cannot infer alignment pairs from npz keys={list(npz.keys())}")


def normalize_index_base(pairs: np.ndarray, n_score: int, n_perf: int) -> np.ndarray:
    """If alignment looks one-based, convert to zero-based. Keep -1 as unmatched."""
    out = pairs.copy()
    score_nonneg = out[:, 0][out[:, 0] >= 0]
    perf_nonneg = out[:, 1][out[:, 1] >= 0]
    if len(score_nonneg) and score_nonneg.min() >= 1 and score_nonneg.max() == n_score:
        out[:, 0] = np.where(out[:, 0] >= 0, out[:, 0] - 1, out[:, 0])
    if len(perf_nonneg) and perf_nonneg.min() >= 1 and perf_nonneg.max() == n_perf:
        out[:, 1] = np.where(out[:, 1] >= 0, out[:, 1] - 1, out[:, 1])
    return out


def parse_streams(perf_midi_path: str, score_midi_path: str, score_xml_path: str):
    perf = MultistreamTokenizer.parse_midi(perf_midi_path)
    score_midi = MultistreamTokenizer.parse_midi(score_midi_path)
    score_xml = MultistreamTokenizer.parse_mxl(score_xml_path)
    return perf, score_midi, score_xml


def check_identity_mapping(perf, score_midi, score_xml, pairs: np.ndarray) -> Dict[str, Any]:
    n_perf = int(len(perf["pitch"]))
    n_score_midi = int(len(score_midi["pitch"]))
    n_xml = int(len(score_xml["pitch"]))

    pairs = normalize_index_base(pairs, n_score_midi, n_perf)
    matched = pairs[(pairs[:, 0] >= 0) & (pairs[:, 1] >= 0)]

    max_score_idx = int(matched[:, 0].max()) if len(matched) else -1
    max_perf_idx = int(matched[:, 1].max()) if len(matched) else -1

    score_len_equal = n_score_midi == n_xml
    if score_len_equal:
        score_pitch_equal = bool(np.array_equal(
            np.asarray(score_midi["pitch"].cpu() if hasattr(score_midi["pitch"], "cpu") else score_midi["pitch"]).astype(int),
            np.asarray(score_xml["pitch"].cpu() if hasattr(score_xml["pitch"], "cpu") else score_xml["pitch"]).astype(int),
        ))
    else:
        score_pitch_equal = False

    return {
        "n_perf": n_perf,
        "n_score_midi": n_score_midi,
        "n_xml": n_xml,
        "n_pairs": int(len(pairs)),
        "n_matched": int(len(matched)),
        "max_score_idx": max_score_idx,
        "max_perf_idx": max_perf_idx,
        "score_len_equal": score_len_equal,
        "score_pitch_equal": score_pitch_equal,
        "perf_idx_in_bounds": max_perf_idx < n_perf,
        "score_idx_in_bounds": max_score_idx < n_score_midi,
        "pairs_zero_based": pairs,
    }


def make_note_count_chunks(
    n_xml: int,
    n_perf: int,
    pairs: np.ndarray,
    note_chunk_size: int = 512,
    min_mxl_notes: int = 16,
    min_midi_notes: int = 16,
    min_matched_score_ratio: float = 0.50,
    min_matched_perf_ratio: float = 0.30,
    context_perf_notes: int = 0,
) -> Dict[str, Any]:
    """Generate PM2S chunk json from identity-mapped PianoCoRe alignment pairs."""
    pairs = pairs.astype(np.int64, copy=False)
    matched = pairs[(pairs[:, 0] >= 0) & (pairs[:, 1] >= 0)]

    score_to_perf: dict[int, list[int]] = {}
    perf_to_score: dict[int, list[int]] = {}
    for s, p in matched:
        if 0 <= s < n_xml and 0 <= p < n_perf:
            score_to_perf.setdefault(int(s), []).append(int(p))
            perf_to_score.setdefault(int(p), []).append(int(s))

    midi_chunks: list[list[int]] = []
    mxl_chunks: list[list[int]] = []
    stats: list[dict[str, Any]] = []

    for start in range(0, n_xml, note_chunk_size):
        end = min(start + note_chunk_size, n_xml)
        mxl_chunk = list(range(start, end))
        if len(mxl_chunk) < min_mxl_notes:
            continue

        perf_matched = sorted({p for s in mxl_chunk for p in score_to_perf.get(s, [])})
        if not perf_matched:
            continue

        p0 = max(min(perf_matched) - context_perf_notes, 0)
        p1 = min(max(perf_matched) + context_perf_notes + 1, n_perf)
        midi_chunk = list(range(p0, p1))
        if len(midi_chunk) < min_midi_notes:
            continue

        matched_score_ratio = len([s for s in mxl_chunk if s in score_to_perf]) / max(len(mxl_chunk), 1)
        matched_perf_in_chunk = [p for p in midi_chunk if any(start <= s < end for s in perf_to_score.get(p, []))]
        matched_perf_ratio = len(matched_perf_in_chunk) / max(len(midi_chunk), 1)

        if matched_score_ratio < min_matched_score_ratio or matched_perf_ratio < min_matched_perf_ratio:
            continue

        midi_chunks.append(midi_chunk)
        mxl_chunks.append(mxl_chunk)
        stats.append({
            "mxl_start": start,
            "mxl_end": end,
            "midi_start": p0,
            "midi_end": p1,
            "n_mxl": len(mxl_chunk),
            "n_midi": len(midi_chunk),
            "matched_score_ratio": matched_score_ratio,
            "matched_perf_ratio": matched_perf_ratio,
        })

    return {
        "midi": midi_chunks,
        "mxl": mxl_chunks,
        "swapped": False,
        "source": "pianocore_raw_alignment",
        "chunk_unit": "target_note_count",
        "note_chunk_size": int(note_chunk_size),
        "stats": stats,
    }


def default_chunk_path(
    row: pd.Series,
    out_dir: str,
    root: str,
    chunk_location: str = "next_to_midi",
) -> str:
    """
    Decide where to save PianoCoRe chunk json.

    chunk_location:
      - "next_to_midi": ASAP-like behavior. Save next to the performance MIDI:
            /.../Aria_772751_0.mid
            /.../Aria_772751_0_pianocore_chunks.json
      - "out_dir": centralized behavior. Save under --out-dir.

    The returned path is absolute because resolve_path() resolves the performance MIDI
    against <pianocore_root>/PianoCoRe/raw when needed.
    """
    perf_path = resolve_path(row["performance_midi_path"], root)
    perf_stem_path = Path(perf_path).with_suffix("")

    if chunk_location == "next_to_midi":
        return str(perf_stem_path) + "_pianocore_chunks.json"

    if chunk_location == "out_dir":
        stem = Path(perf_path).stem
        row_id = str(row.get("id", row.name)).replace("/", "_")
        return str(Path(out_dir) / f"{row_id}__{stem}_pianocore_chunks.json")

    raise ValueError(f"Invalid chunk_location={chunk_location}")


def handle_row(row_dict: Dict[str, Any], args) -> Dict[str, Any]:
    row = pd.Series(row_dict)
    out = dict(row_dict)
    row_id = str(row.get("id", row.name))
    try:
        perf_midi_path = resolve_path(row["performance_midi_path"], args.pianocore_root)
        score_xml_path = resolve_path(row["score_xml_path"], args.pianocore_root)
        score_midi_path = resolve_path(row["score_midi_path"], args.pianocore_root)
        raw_alignment_path = str(row["raw_alignment_path"])

        for p in [perf_midi_path, score_xml_path, score_midi_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(p)

        perf, score_midi, score_xml = parse_streams(perf_midi_path, score_midi_path, score_xml_path)
        npz = read_npz(raw_alignment_path, args.pianocore_root, args.raw_alignments_zip)
        pairs = extract_alignment_pairs(npz)
        check = check_identity_mapping(perf, score_midi, score_xml, pairs)
        pairs = check.pop("pairs_zero_based")

        if args.require_score_identity and not (check["score_len_equal"] and check["score_pitch_equal"]):
            raise RuntimeError(f"score MIDI != parse_mxl identity check failed: {check}")
        if not (check["perf_idx_in_bounds"] and check["score_idx_in_bounds"]):
            raise RuntimeError(f"alignment index out of bounds: {check}")

        chunks = make_note_count_chunks(
            n_xml=check["n_xml"],
            n_perf=check["n_perf"],
            pairs=pairs,
            note_chunk_size=args.note_chunk_size,
            min_mxl_notes=args.min_mxl_notes,
            min_midi_notes=args.min_midi_notes,
            min_matched_score_ratio=args.min_matched_score_ratio,
            min_matched_perf_ratio=args.min_matched_perf_ratio,
            context_perf_notes=args.context_perf_notes,
        )
        if len(chunks["midi"]) == 0:
            raise RuntimeError("no chunks passed chunk-level filters")

        chunk_path = default_chunk_path(
            row,
            args.out_dir,
            args.pianocore_root,
            args.chunk_location,
        )
        os.makedirs(os.path.dirname(chunk_path), exist_ok=True)
        with open(chunk_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f)

        out.update(check)
        out.update({
            "chunk_path": chunk_path,
            "chunk_status": "ok",
            "chunk_error": "",
            "n_chunks": len(chunks["midi"]),
        })
        return out

    except Exception as e:
        out.update({
            "chunk_path": "",
            "chunk_status": "failed",
            "chunk_error": f"{type(e).__name__}: {str(e)}",
            "n_chunks": 0,
        })
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--pianocore-root", default="/mnt/ssd/hbli/datasets/pianocore/")
    ap.add_argument("--raw-alignments-zip", default="")
    ap.add_argument("--out-dir", default="/mnt/ssd/hbli/datasets/pianocore/pm2s_chunks")
    ap.add_argument(
        "--chunk-location",
        choices=["next_to_midi", "out_dir"],
        default="next_to_midi",
    )
    ap.add_argument("--out-manifest", default="/mnt/ssd/hbli/datasets/pianocore/pm2s_pianocore_manifest_with_chunks.csv")
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--note-chunk-size", type=int, default=512)
    ap.add_argument("--min-mxl-notes", type=int, default=16)
    ap.add_argument("--min-midi-notes", type=int, default=16)
    ap.add_argument("--min-matched-score-ratio", type=float, default=0.50)
    ap.add_argument("--min-matched-perf-ratio", type=float, default=0.30)
    ap.add_argument("--context-perf-notes", type=int, default=0)
    ap.add_argument("--require-score-identity", action="store_true", default=True)
    ap.add_argument("--no-require-score-identity", dest="require_score_identity", action="store_false")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.manifest, low_memory=False)
    required_cols = {"performance_midi_path", "score_xml_path", "score_midi_path", "raw_alignment_path"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"manifest missing columns: {missing}")
    if args.max_rows and args.max_rows > 0:
        df = df.head(args.max_rows).copy()

    rows = df.to_dict("records")
    results = Parallel(n_jobs=min(args.n_jobs, len(rows)), verbose=10)(
        delayed(handle_row)(row, args) for row in rows
    )
    out_df = pd.DataFrame(results)
    out_df.to_csv(args.out_manifest, index=False)
    print("saved:", args.out_manifest)
    print(out_df["chunk_status"].value_counts(dropna=False))
    ok_df = out_df[out_df["chunk_status"].eq("ok")]
    if len(ok_df):
        print("ok rows:", len(ok_df), "chunks:", int(ok_df["n_chunks"].sum()))


if __name__ == "__main__":
    main()
