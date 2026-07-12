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


def _as_numpy_1d(x: Any, dtype=float) -> np.ndarray:
    arr = np.asarray(x.cpu() if hasattr(x, "cpu") else x)
    return arr.reshape(-1).astype(dtype, copy=False)


def _metadata_int(row: pd.Series, col: str) -> Optional[int]:
    if col not in row or pd.isna(row[col]):
        return None
    try:
        return int(round(float(row[col])))
    except Exception:
        return None


def _identity_key(row: pd.Series) -> str:
    row_id = row.get("id", "")
    if row_id is not None and not pd.isna(row_id) and str(row_id):
        return str(row_id)
    return str(row.get("performance_midi_path", ""))


def _load_identity_cache_keys(path: str) -> set[str]:
    if not path:
        return set()
    df = pd.read_csv(path, low_memory=False)
    if "chunk_status" in df.columns:
        df = df[df["chunk_status"].astype(str).eq("ok")]
    elif "chunk_path" in df.columns:
        df = df[df["chunk_path"].notna()]
    keys = set()
    for _, row in df.iterrows():
        keys.add(_identity_key(row))
        if "performance_midi_path" in row and not pd.isna(row["performance_midi_path"]):
            keys.add(str(row["performance_midi_path"]))
    return keys


def _trusted_by_existing_chunk(row: pd.Series, args) -> bool:
    if not args.identity_cache_chunk_suffix:
        return False
    chunk_path = default_chunk_path(
        row,
        args.identity_cache_out_dir or args.out_dir,
        args.pianocore_root,
        args.identity_cache_chunk_location,
        args.identity_cache_chunk_suffix,
    )
    return os.path.exists(chunk_path)


def _is_identity_trusted(row: pd.Series, args) -> bool:
    if not args.reuse_identity_cache:
        return False
    keys = getattr(args, "identity_cache_keys", set())
    if keys and (_identity_key(row) in keys or str(row.get("performance_midi_path", "")) in keys):
        return True
    return _trusted_by_existing_chunk(row, args)


def _score_onsets_from_xml(score_xml: Dict[str, Any]) -> np.ndarray:
    if "absolute_onset" not in score_xml:
        raise ValueError("parse_mxl output has no absolute_onset; tokenizer_v2 is required")
    return _as_numpy_1d(score_xml["absolute_onset"], dtype=float)


def _score_onsets_from_npz(npz: Dict[str, Any], n_xml: int) -> np.ndarray:
    """
    Build score-index -> score onset from PianoCoRe score_idx/score_times arrays.

    PianoCoRe alignment arrays are alignment-row based, not necessarily score-note
    indexed arrays, so we scatter score_times[:, 0] by score_idx.
    """
    if "score_idx" not in npz or "score_times" not in npz:
        raise ValueError("npz_score_times source requires score_idx and score_times keys")
    score_idx = np.asarray(npz["score_idx"]).reshape(-1).astype(np.int64, copy=False)
    score_times = np.asarray(npz["score_times"])
    if score_times.ndim != 2 or score_times.shape[1] < 1 or len(score_times) != len(score_idx):
        raise ValueError(f"Invalid score_times shape={score_times.shape} for score_idx len={len(score_idx)}")

    out = np.full((n_xml,), np.nan, dtype=float)
    for s, t in zip(score_idx, score_times[:, 0]):
        if 0 <= s < n_xml and np.isnan(out[s]):
            out[s] = float(t)
    if np.isnan(out).any():
        missing = int(np.isnan(out).sum())
        raise ValueError(f"Cannot derive npz score onset for {missing}/{n_xml} score notes")
    return out


def _raw_monotonic_ratio(
    mxl_chunk: list[int],
    score_to_perf: dict[int, list[int]],
) -> float:
    perf_for_score = [
        min(score_to_perf[s])
        for s in mxl_chunk
        if score_to_perf.get(s)
    ]
    if len(perf_for_score) <= 1:
        return 1.0
    return float(np.mean(np.diff(np.asarray(perf_for_score)) >= 0))


def _chord_aware_monotonic_ratio(
    mxl_chunk: list[int],
    score_to_perf: dict[int, list[int]],
    score_onsets: np.ndarray,
    onset_eps: float,
) -> tuple[float, int, int]:
    """Compare aligned performance order only between distinct score onsets."""
    intervals: list[tuple[int, int]] = []
    current_onset: Optional[float] = None
    current_perf: list[int] = []
    multi_note_group_count = 0
    grouped_note_count = 0

    def flush_group() -> None:
        nonlocal current_onset, current_perf
        nonlocal multi_note_group_count, grouped_note_count
        if not current_perf:
            current_onset = None
            return
        intervals.append((min(current_perf), max(current_perf)))
        if len(current_perf) > 1:
            multi_note_group_count += 1
            grouped_note_count += len(current_perf)
        current_onset = None
        current_perf = []

    for score_idx in mxl_chunk:
        perf_indices = score_to_perf.get(score_idx, [])
        if not perf_indices:
            continue
        onset = float(score_onsets[score_idx])
        if not np.isfinite(onset):
            flush_group()
            intervals.append((min(perf_indices), max(perf_indices)))
            continue
        if current_onset is not None and abs(onset - current_onset) <= onset_eps:
            current_perf.extend(int(p) for p in perf_indices)
        else:
            flush_group()
            current_onset = onset
            current_perf = [int(p) for p in perf_indices]
    flush_group()

    if len(intervals) <= 1:
        return 1.0, multi_note_group_count, grouped_note_count
    transitions = [
        intervals[i][0] >= intervals[i - 1][1]
        for i in range(1, len(intervals))
    ]
    return float(np.mean(transitions)), multi_note_group_count, grouped_note_count


def make_score_time_chunks(
    n_xml: int,
    n_perf: int,
    pairs: np.ndarray,
    score_onsets: np.ndarray,
    beat_quarter_len: float = 1.0,
    beats_per_chunk: int = 1,
    min_mxl_notes: int = 1,
    min_midi_notes: int = 1,
    min_matched_score_ratio: float = 0.50,
    min_matched_perf_ratio: float = 0.30,
    context_perf_notes: int = 0,
    max_midi_mxl_ratio: float = 4.0,
    max_midi_notes_per_chunk: int = 128,
    min_monotonic_ratio: float = 0.01,
) -> Dict[str, Any]:
    """
    Generate PM2S chunks from score-side time bins plus PianoCoRe note alignment.

    This is a pseudo-beat fallback for datasets without ASAP-style beat annotations:
    notes are grouped by score onset in quarterLength units, then note alignment is
    used to recover the local performance-MIDI span for each score-time bin.
    """
    if n_xml <= 0 or n_perf <= 0:
        return {"midi": [], "mxl": [], "swapped": False, "source": "pianocore_raw_alignment"}
    if len(score_onsets) != n_xml:
        raise ValueError(f"score_onsets len={len(score_onsets)} does not match n_xml={n_xml}")
    if beat_quarter_len <= 0:
        raise ValueError("--beat-quarter-len must be positive")
    if beats_per_chunk <= 0:
        raise ValueError("--beats-per-chunk must be positive")

    pairs = pairs.astype(np.int64, copy=False)
    matched = pairs[(pairs[:, 0] >= 0) & (pairs[:, 1] >= 0)]

    score_to_perf: dict[int, list[int]] = {}
    perf_to_score: dict[int, list[int]] = {}
    for s, p in matched:
        if 0 <= s < n_xml and 0 <= p < n_perf:
            score_to_perf.setdefault(int(s), []).append(int(p))
            perf_to_score.setdefault(int(p), []).append(int(s))

    width = float(beat_quarter_len) * int(beats_per_chunk)
    finite_onsets = score_onsets[np.isfinite(score_onsets)]
    if len(finite_onsets) == 0:
        raise ValueError("No finite score onset values")

    start_bin = int(np.floor(float(finite_onsets.min()) / width))
    end_bin = int(np.floor(float(finite_onsets.max()) / width))

    midi_chunks: list[list[int]] = []
    mxl_chunks: list[list[int]] = []
    stats: list[dict[str, Any]] = []

    for b in range(start_bin, end_bin + 1):
        t0 = b * width
        t1 = t0 + width
        if b == end_bin:
            mxl_idx = np.where((score_onsets >= t0) & (score_onsets <= t1))[0]
        else:
            mxl_idx = np.where((score_onsets >= t0) & (score_onsets < t1))[0]
        mxl_chunk = [int(i) for i in mxl_idx if 0 <= int(i) < n_xml]
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
        if max_midi_notes_per_chunk > 0 and len(midi_chunk) > max_midi_notes_per_chunk:
            continue
        if max_midi_mxl_ratio > 0 and len(midi_chunk) / max(len(mxl_chunk), 1) > max_midi_mxl_ratio:
            continue

        matched_score_ratio = len([s for s in mxl_chunk if s in score_to_perf]) / max(len(mxl_chunk), 1)
        matched_perf_in_chunk = [p for p in midi_chunk if any(s in mxl_chunk for s in perf_to_score.get(p, []))]
        matched_perf_ratio = len(matched_perf_in_chunk) / max(len(midi_chunk), 1)

        perf_for_score = []
        for s in mxl_chunk:
            ps = score_to_perf.get(s, [])
            if ps:
                perf_for_score.append(min(ps))
        if len(perf_for_score) <= 1:
            monotonic_ratio = 1.0
        else:
            monotonic_ratio = float(np.mean(np.diff(np.asarray(perf_for_score)) >= 0))

        if matched_score_ratio < min_matched_score_ratio or matched_perf_ratio < min_matched_perf_ratio:
            continue
        if monotonic_ratio < min_monotonic_ratio:
            continue

        midi_chunks.append(midi_chunk)
        mxl_chunks.append(mxl_chunk)
        stats.append({
            "score_time_start": float(t0),
            "score_time_end": float(t1),
            "midi_start": p0,
            "midi_end": p1,
            "n_mxl": len(mxl_chunk),
            "n_midi": len(midi_chunk),
            "matched_score_ratio": matched_score_ratio,
            "matched_perf_ratio": matched_perf_ratio,
            "monotonic_ratio": monotonic_ratio,
        })

    return {
        "midi": midi_chunks,
        "mxl": mxl_chunks,
        "swapped": False,
        "source": "pianocore_raw_alignment",
        "chunk_unit": "score_time_pseudo_beat",
        "beat_quarter_len": float(beat_quarter_len),
        "beats_per_chunk": int(beats_per_chunk),
        "stats": stats,
    }


def make_score_time_padding_chunks(
    n_xml: int,
    n_perf: int,
    pairs: np.ndarray,
    score_onsets: np.ndarray,
    beat_quarter_len: float = 1.0,
    beats_per_chunk: int = 1,
    min_mxl_notes: int = 1,
    min_midi_notes: int = 1,
    min_matched_score_ratio: float = 0.50,
    min_matched_perf_ratio: float = 0.30,
    context_perf_notes: int = 0,
    max_midi_mxl_ratio: float = 4.0,
    max_midi_notes_per_chunk: int = 128,
    min_monotonic_ratio: float = 0.01,
    chord_aware_monotonic: bool = False,
    chord_onset_eps: float = 1e-6,
) -> Dict[str, Any]:
    """
    Generate score-time chunks while preserving rejected score-time bins as
    padding-only spans. Good chunks train normally; bad chunks have empty
    midi/mxl lists plus chunk_is_trainable=false and chunk_pad_len>0.
    """
    if n_xml <= 0 or n_perf <= 0:
        return {"midi": [], "mxl": [], "swapped": False, "source": "pianocore_raw_alignment"}
    if len(score_onsets) != n_xml:
        raise ValueError(f"score_onsets len={len(score_onsets)} does not match n_xml={n_xml}")
    if beat_quarter_len <= 0:
        raise ValueError("--beat-quarter-len must be positive")
    if beats_per_chunk <= 0:
        raise ValueError("--beats-per-chunk must be positive")
    if chord_onset_eps < 0:
        raise ValueError("--chord-onset-eps must be non-negative")

    pairs = pairs.astype(np.int64, copy=False)
    matched = pairs[(pairs[:, 0] >= 0) & (pairs[:, 1] >= 0)]

    score_to_perf: dict[int, list[int]] = {}
    perf_to_score: dict[int, list[int]] = {}
    for s, p in matched:
        if 0 <= s < n_xml and 0 <= p < n_perf:
            score_to_perf.setdefault(int(s), []).append(int(p))
            perf_to_score.setdefault(int(p), []).append(int(s))

    width = float(beat_quarter_len) * int(beats_per_chunk)
    finite_onsets = score_onsets[np.isfinite(score_onsets)]
    if len(finite_onsets) == 0:
        raise ValueError("No finite score onset values")

    start_bin = int(np.floor(float(finite_onsets.min()) / width))
    end_bin = int(np.floor(float(finite_onsets.max()) / width))

    midi_chunks: list[list[int]] = []
    mxl_chunks: list[list[int]] = []
    chunk_is_trainable: list[bool] = []
    chunk_pad_len: list[int] = []
    stats: list[dict[str, Any]] = []
    rejection_counts: dict[str, int] = {}

    def reject(reason: str):
        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

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

        reason = "ok"
        perf_matched = sorted({p for s in mxl_chunk for p in score_to_perf.get(s, [])})
        midi_chunk: list[int] = []
        matched_score_ratio = 0.0
        matched_perf_ratio = 0.0
        monotonic_ratio = 1.0
        raw_monotonic_ratio = 1.0
        chord_aware_monotonic_ratio = 1.0
        chord_group_count = 0
        chord_note_count = 0

        if len(mxl_chunk) < min_mxl_notes:
            reason = "min_mxl_notes"
        elif not perf_matched:
            reason = "no_matched_perf"
        else:
            p0 = max(min(perf_matched) - context_perf_notes, 0)
            p1 = min(max(perf_matched) + context_perf_notes + 1, n_perf)
            midi_chunk = list(range(p0, p1))
            if len(midi_chunk) < min_midi_notes:
                reason = "min_midi_notes"
            elif max_midi_notes_per_chunk > 0 and len(midi_chunk) > max_midi_notes_per_chunk:
                reason = "max_midi_notes_per_chunk"
            elif max_midi_mxl_ratio > 0 and len(midi_chunk) / max(len(mxl_chunk), 1) > max_midi_mxl_ratio:
                reason = "max_midi_mxl_ratio"
            else:
                matched_score_ratio = len([s for s in mxl_chunk if s in score_to_perf]) / max(len(mxl_chunk), 1)
                matched_perf_in_chunk = [p for p in midi_chunk if any(s in mxl_chunk for s in perf_to_score.get(p, []))]
                matched_perf_ratio = len(matched_perf_in_chunk) / max(len(midi_chunk), 1)

                raw_monotonic_ratio = _raw_monotonic_ratio(mxl_chunk, score_to_perf)
                chord_aware_monotonic_ratio = raw_monotonic_ratio
                if chord_aware_monotonic:
                    (
                        chord_aware_monotonic_ratio,
                        chord_group_count,
                        chord_note_count,
                    ) = _chord_aware_monotonic_ratio(
                        mxl_chunk,
                        score_to_perf,
                        score_onsets,
                        chord_onset_eps,
                    )
                monotonic_ratio = (
                    chord_aware_monotonic_ratio
                    if chord_aware_monotonic
                    else raw_monotonic_ratio
                )

                if matched_score_ratio < min_matched_score_ratio:
                    reason = "min_matched_score_ratio"
                elif matched_perf_ratio < min_matched_perf_ratio:
                    reason = "min_matched_perf_ratio"
                elif monotonic_ratio < min_monotonic_ratio:
                    reason = "min_monotonic_ratio"

        is_trainable = reason == "ok"
        if not is_trainable:
            reject(reason)
            pad_len = max(len(mxl_chunk), len(midi_chunk), 1)
            midi_chunks.append([])
            mxl_chunks.append([])
        else:
            pad_len = max(len(mxl_chunk), len(midi_chunk), 1)
            midi_chunks.append(midi_chunk)
            mxl_chunks.append(mxl_chunk)

        chunk_is_trainable.append(is_trainable)
        chunk_pad_len.append(int(pad_len))
        stats.append({
            "score_bin": int(b),
            "score_time_start": float(t0),
            "score_time_end": float(t1),
            "midi_start": int(min(midi_chunk)) if midi_chunk else -1,
            "midi_end": int(max(midi_chunk) + 1) if midi_chunk else -1,
            "n_mxl": len(mxl_chunk),
            "n_midi": len(midi_chunk),
            "matched_score_ratio": matched_score_ratio,
            "matched_perf_ratio": matched_perf_ratio,
            "monotonic_ratio": monotonic_ratio,
            "raw_monotonic_ratio": raw_monotonic_ratio,
            "chord_aware_monotonic_ratio": chord_aware_monotonic_ratio,
            "chord_group_count": chord_group_count,
            "chord_note_count": chord_note_count,
            "monotonic_reclassified": bool(
                is_trainable
                and chord_aware_monotonic
                and raw_monotonic_ratio < min_monotonic_ratio
                and chord_aware_monotonic_ratio >= min_monotonic_ratio
            ),
            "chunk_is_trainable": is_trainable,
            "reject_reason": reason,
            "pad_len": int(pad_len),
        })

    if not any(chunk_is_trainable):
        return {
            "midi": [],
            "mxl": [],
            "swapped": False,
            "source": "pianocore_raw_alignment",
            "chunk_unit": "score_time_pseudo_beat_padding",
            "chunk_monotonic_mode": (
                "score_onset_group" if chord_aware_monotonic else "raw_note_order"
            ),
            "chord_onset_eps": float(chord_onset_eps),
            "beat_quarter_len": float(beat_quarter_len),
            "beats_per_chunk": int(beats_per_chunk),
            "stats": stats,
            "chunk_is_trainable": chunk_is_trainable,
            "chunk_pad_len": chunk_pad_len,
            "rejection_counts": rejection_counts,
        }

    return {
        "midi": midi_chunks,
        "mxl": mxl_chunks,
        "swapped": False,
        "source": "pianocore_raw_alignment",
        "chunk_unit": "score_time_pseudo_beat_padding",
        "chunk_monotonic_mode": (
            "score_onset_group" if chord_aware_monotonic else "raw_note_order"
        ),
        "chord_onset_eps": float(chord_onset_eps),
        "beat_quarter_len": float(beat_quarter_len),
        "beats_per_chunk": int(beats_per_chunk),
        "stats": stats,
        "chunk_is_trainable": chunk_is_trainable,
        "chunk_pad_len": chunk_pad_len,
        "rejection_counts": rejection_counts,
        "n_bad_chunks": int(sum(not x for x in chunk_is_trainable)),
        "n_good_chunks": int(sum(chunk_is_trainable)),
        "n_chord_monotonic_rescued": int(
            sum(bool(x.get("monotonic_reclassified")) for x in stats)
        ),
    }


def make_alignment_segment_chunks(
    n_xml: int,
    n_perf: int,
    pairs: np.ndarray,
    score_onsets: np.ndarray,
    beat_quarter_len: float = 1.0,
    beats_per_chunk: int = 1,
    min_segment_mxl_notes: int = 64,
    min_mxl_notes: int = 1,
    min_midi_notes: int = 1,
    min_matched_score_ratio: float = 0.70,
    min_matched_perf_ratio: float = 0.30,
    context_perf_notes: int = 0,
    max_midi_mxl_ratio: float = 4.0,
    max_midi_notes_per_chunk: int = 128,
    min_monotonic_ratio: float = 0.95,
    max_unmatched_score_gap: int = 16,
    max_unmatched_perf_gap: int = 64,
    max_perf_backtrack: int = 8,
) -> Dict[str, Any]:
    """
    Generate pseudo-beat chunks and guard training against skipped bad regions.

    Each chunk is one score-time bin (or a small group of bins) exactly like
    make_score_time_chunks. The extra alignment_segment_id is not a note-window;
    it marks a run of consecutive accepted pseudo-beat chunks. If any non-empty
    score-time bin is rejected by chunk-level filters, or if the performance
    indices jump/backtrack too far, the next accepted chunk starts a new segment.
    Training may concatenate chunks only inside one segment, so it cannot jump
    across a discarded local region.
    """
    if n_xml <= 0 or n_perf <= 0:
        return {"midi": [], "mxl": [], "swapped": False, "source": "pianocore_raw_alignment"}
    if len(score_onsets) != n_xml:
        raise ValueError(f"score_onsets len={len(score_onsets)} does not match n_xml={n_xml}")
    if beat_quarter_len <= 0:
        raise ValueError("--beat-quarter-len must be positive")
    if beats_per_chunk <= 0:
        raise ValueError("--beats-per-chunk must be positive")

    pairs = pairs.astype(np.int64, copy=False)
    matched = pairs[(pairs[:, 0] >= 0) & (pairs[:, 1] >= 0)]

    score_to_perf: dict[int, list[int]] = {}
    perf_to_score: dict[int, list[int]] = {}
    for s, p in matched:
        if 0 <= s < n_xml and 0 <= p < n_perf:
            score_to_perf.setdefault(int(s), []).append(int(p))
            perf_to_score.setdefault(int(p), []).append(int(s))

    width = float(beat_quarter_len) * int(beats_per_chunk)
    finite_onsets = score_onsets[np.isfinite(score_onsets)]
    if len(finite_onsets) == 0:
        raise ValueError("No finite score onset values")

    start_bin = int(np.floor(float(finite_onsets.min()) / width))
    end_bin = int(np.floor(float(finite_onsets.max()) / width))

    accepted: list[dict[str, Any]] = []
    rejected_nonempty_bins: set[int] = set()
    rejection_counts: dict[str, int] = {}

    def reject(bin_id: int, reason: str):
        rejected_nonempty_bins.add(int(bin_id))
        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

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
        if len(mxl_chunk) < min_mxl_notes:
            reject(b, "min_mxl_notes")
            continue

        perf_matched = sorted({p for s in mxl_chunk for p in score_to_perf.get(s, [])})
        if not perf_matched:
            reject(b, "no_matched_perf")
            continue

        p0 = max(min(perf_matched) - context_perf_notes, 0)
        p1 = min(max(perf_matched) + context_perf_notes + 1, n_perf)
        midi_chunk = list(range(p0, p1))
        if len(midi_chunk) < min_midi_notes:
            reject(b, "min_midi_notes")
            continue
        if max_midi_notes_per_chunk > 0 and len(midi_chunk) > max_midi_notes_per_chunk:
            reject(b, "max_midi_notes_per_chunk")
            continue
        if max_midi_mxl_ratio > 0 and len(midi_chunk) / max(len(mxl_chunk), 1) > max_midi_mxl_ratio:
            reject(b, "max_midi_mxl_ratio")
            continue

        matched_score_ratio = len([s for s in mxl_chunk if s in score_to_perf]) / max(len(mxl_chunk), 1)
        matched_perf_in_chunk = [p for p in midi_chunk if any(s in mxl_chunk for s in perf_to_score.get(p, []))]
        matched_perf_ratio = len(matched_perf_in_chunk) / max(len(midi_chunk), 1)

        perf_for_score = []
        for s in mxl_chunk:
            ps = score_to_perf.get(s, [])
            if ps:
                perf_for_score.append(min(ps))
        if len(perf_for_score) <= 1:
            monotonic_ratio = 1.0
        else:
            monotonic_ratio = float(np.mean(np.diff(np.asarray(perf_for_score)) >= 0))

        if matched_score_ratio < min_matched_score_ratio:
            reject(b, "min_matched_score_ratio")
            continue
        if matched_perf_ratio < min_matched_perf_ratio:
            reject(b, "min_matched_perf_ratio")
            continue
        if monotonic_ratio < min_monotonic_ratio:
            reject(b, "min_monotonic_ratio")
            continue

        accepted.append({
            "score_bin": int(b),
            "score_time_start": float(t0),
            "score_time_end": float(t1),
            "midi": midi_chunk,
            "mxl": mxl_chunk,
            "midi_start": int(p0),
            "midi_end": int(p1),
            "mxl_start": int(min(mxl_chunk)),
            "mxl_end": int(max(mxl_chunk) + 1),
            "n_mxl": len(mxl_chunk),
            "n_midi": len(midi_chunk),
            "matched_score_ratio": matched_score_ratio,
            "matched_perf_ratio": matched_perf_ratio,
            "monotonic_ratio": monotonic_ratio,
        })

    if not accepted:
        return {
            "midi": [],
            "mxl": [],
            "swapped": False,
            "source": "pianocore_raw_alignment",
            "chunk_unit": "score_time_pseudo_beat_segment_guard",
            "beat_quarter_len": float(beat_quarter_len),
            "beats_per_chunk": int(beats_per_chunk),
            "stats": [],
            "alignment_segment_id": [],
            "rejection_counts": rejection_counts,
            "n_rejected_score_time_bins": len(rejected_nonempty_bins),
        }

    def has_rejected_gap(prev_bin: int, cur_bin: int) -> bool:
        if cur_bin <= prev_bin + 1:
            return False
        return any(prev_bin < b < cur_bin for b in rejected_nonempty_bins)

    raw_segments: list[list[dict[str, Any]]] = []
    cur_segment: list[dict[str, Any]] = []
    prev: Optional[dict[str, Any]] = None
    for item in accepted:
        should_break = prev is None
        break_reason = "first_chunk" if should_break else ""
        if prev is not None:
            score_note_gap = int(item["mxl_start"] - prev["mxl_end"])
            perf_forward_gap = int(item["midi_start"] - prev["midi_end"]) if item["midi_start"] >= prev["midi_end"] else 0
            perf_backtrack = int(prev["midi_end"] - item["midi_start"]) if item["midi_start"] < prev["midi_end"] else 0

            if has_rejected_gap(prev["score_bin"], item["score_bin"]):
                should_break = True
                break_reason = "rejected_score_time_bin"
            elif max_unmatched_score_gap >= 0 and score_note_gap > max_unmatched_score_gap:
                should_break = True
                break_reason = "score_note_gap"
            elif max_unmatched_perf_gap >= 0 and perf_forward_gap > max_unmatched_perf_gap:
                should_break = True
                break_reason = "perf_note_gap"
            elif max_perf_backtrack >= 0 and perf_backtrack > max_perf_backtrack:
                should_break = True
                break_reason = "perf_backtrack"

        item["segment_break_reason"] = break_reason
        if should_break:
            if cur_segment:
                raw_segments.append(cur_segment)
            cur_segment = [item]
        else:
            cur_segment.append(item)
        prev = item
    if cur_segment:
        raw_segments.append(cur_segment)

    midi_chunks: list[list[int]] = []
    mxl_chunks: list[list[int]] = []
    alignment_segment_ids: list[int] = []
    stats: list[dict[str, Any]] = []

    kept_segment_id = 0
    dropped_short_segments = 0
    for segment in raw_segments:
        segment_n_mxl = sum(int(item["n_mxl"]) for item in segment)
        if segment_n_mxl < min_segment_mxl_notes:
            dropped_short_segments += 1
            continue

        segment_score_start = min(int(item["mxl_start"]) for item in segment)
        segment_score_end = max(int(item["mxl_end"]) for item in segment)
        segment_perf_start = min(int(item["midi_start"]) for item in segment)
        segment_perf_end = max(int(item["midi_end"]) for item in segment)

        for local_chunk_id, item in enumerate(segment):
            midi_chunks.append(item["midi"])
            mxl_chunks.append(item["mxl"])
            alignment_segment_ids.append(kept_segment_id)
            stats.append({
                "alignment_segment_id": kept_segment_id,
                "segment_chunk_id": local_chunk_id,
                "segment_score_start": segment_score_start,
                "segment_score_end": segment_score_end,
                "segment_perf_start": segment_perf_start,
                "segment_perf_end": segment_perf_end,
                "segment_n_chunks": len(segment),
                "segment_n_mxl": segment_n_mxl,
                "score_bin": item["score_bin"],
                "score_time_start": item["score_time_start"],
                "score_time_end": item["score_time_end"],
                "mxl_start": item["mxl_start"],
                "mxl_end": item["mxl_end"],
                "midi_start": item["midi_start"],
                "midi_end": item["midi_end"],
                "n_mxl": item["n_mxl"],
                "n_midi": item["n_midi"],
                "matched_score_ratio": item["matched_score_ratio"],
                "matched_perf_ratio": item["matched_perf_ratio"],
                "monotonic_ratio": item["monotonic_ratio"],
                "segment_break_reason": item["segment_break_reason"],
            })
        kept_segment_id += 1

    return {
        "midi": midi_chunks,
        "mxl": mxl_chunks,
        "swapped": False,
        "source": "pianocore_raw_alignment",
        "chunk_unit": "score_time_pseudo_beat_segment_guard",
        "beat_quarter_len": float(beat_quarter_len),
        "beats_per_chunk": int(beats_per_chunk),
        "min_segment_mxl_notes": int(min_segment_mxl_notes),
        "alignment_segment_id": alignment_segment_ids,
        "stats": stats,
        "rejection_counts": rejection_counts,
        "n_rejected_score_time_bins": len(rejected_nonempty_bins),
        "n_dropped_short_segments": dropped_short_segments,
    }


def default_chunk_path(
    row: pd.Series,
    out_dir: str,
    root: str,
    chunk_location: str = "next_to_midi",
    chunk_suffix: str = "_pianocore_chunks.json",
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
        return str(perf_stem_path) + chunk_suffix

    if chunk_location == "out_dir":
        stem = Path(perf_path).stem
        row_id = str(row.get("id", row.name)).replace("/", "_")
        return str(Path(out_dir) / f"{row_id}__{stem}{chunk_suffix}")

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

            needs_xml_onsets = (
                args.chunk_mode in {"score_time", "score_time_padding", "alignment_segment"}
                and args.pseudo_beat_source == "xml_absolute_onset"
            )
            if needs_xml_onsets or n_xml is None:
                score_xml = MultistreamTokenizer.parse_mxl(score_xml_path)
                n_xml = int(len(score_xml["pitch"]))
            if n_perf is None:
                perf = MultistreamTokenizer.parse_midi(perf_midi_path)
                n_perf = int(len(perf["pitch"]))

            pairs = normalize_index_base(pairs_raw, n_xml, n_perf)
            matched = pairs[(pairs[:, 0] >= 0) & (pairs[:, 1] >= 0)]
            max_score_idx = int(matched[:, 0].max()) if len(matched) else -1
            max_perf_idx = int(matched[:, 1].max()) if len(matched) else -1
            check = {
                "n_perf": n_perf,
                "n_score_midi": n_xml,
                "n_xml": n_xml,
                "n_pairs": int(len(pairs)),
                "n_matched": int(len(matched)),
                "max_score_idx": max_score_idx,
                "max_perf_idx": max_perf_idx,
                "score_len_equal": True,
                "score_pitch_equal": True,
                "perf_idx_in_bounds": max_perf_idx < n_perf,
                "score_idx_in_bounds": max_score_idx < n_xml,
                "identity_reused": True,
            }
        else:
            perf, score_midi, score_xml = parse_streams(perf_midi_path, score_midi_path, score_xml_path)
            check = check_identity_mapping(perf, score_midi, score_xml, pairs_raw)
            pairs = check.pop("pairs_zero_based")
            check["identity_reused"] = False

        if args.require_score_identity and not (check["score_len_equal"] and check["score_pitch_equal"]):
            raise RuntimeError(f"score MIDI != parse_mxl identity check failed: {check}")
        if not (check["perf_idx_in_bounds"] and check["score_idx_in_bounds"]):
            raise RuntimeError(f"alignment index out of bounds: {check}")

        if args.chunk_mode == "note_count":
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
        elif args.chunk_mode == "score_time":
            if args.pseudo_beat_source == "xml_absolute_onset":
                if score_xml is None:
                    score_xml = MultistreamTokenizer.parse_mxl(score_xml_path)
                score_onsets = _score_onsets_from_xml(score_xml)
            elif args.pseudo_beat_source == "npz_score_times":
                score_onsets = _score_onsets_from_npz(npz, check["n_xml"])
            else:
                raise ValueError(f"Invalid pseudo_beat_source={args.pseudo_beat_source}")

            chunks = make_score_time_chunks(
                n_xml=check["n_xml"],
                n_perf=check["n_perf"],
                pairs=pairs,
                score_onsets=score_onsets,
                beat_quarter_len=args.beat_quarter_len,
                beats_per_chunk=args.beats_per_chunk,
                min_mxl_notes=args.min_mxl_notes,
                min_midi_notes=args.min_midi_notes,
                min_matched_score_ratio=args.min_matched_score_ratio,
                min_matched_perf_ratio=args.min_matched_perf_ratio,
                context_perf_notes=args.context_perf_notes,
                max_midi_mxl_ratio=args.max_midi_mxl_ratio,
                max_midi_notes_per_chunk=args.max_midi_notes_per_chunk,
                min_monotonic_ratio=args.min_monotonic_ratio,
            )
        elif args.chunk_mode == "score_time_padding":
            if args.pseudo_beat_source == "xml_absolute_onset":
                if score_xml is None:
                    score_xml = MultistreamTokenizer.parse_mxl(score_xml_path)
                score_onsets = _score_onsets_from_xml(score_xml)
            elif args.pseudo_beat_source == "npz_score_times":
                score_onsets = _score_onsets_from_npz(npz, check["n_xml"])
            else:
                raise ValueError(f"Invalid pseudo_beat_source={args.pseudo_beat_source}")

            chunks = make_score_time_padding_chunks(
                n_xml=check["n_xml"],
                n_perf=check["n_perf"],
                pairs=pairs,
                score_onsets=score_onsets,
                beat_quarter_len=args.beat_quarter_len,
                beats_per_chunk=args.beats_per_chunk,
                min_mxl_notes=args.min_mxl_notes,
                min_midi_notes=args.min_midi_notes,
                min_matched_score_ratio=args.min_matched_score_ratio,
                min_matched_perf_ratio=args.min_matched_perf_ratio,
                context_perf_notes=args.context_perf_notes,
                max_midi_mxl_ratio=args.max_midi_mxl_ratio,
                max_midi_notes_per_chunk=args.max_midi_notes_per_chunk,
                min_monotonic_ratio=args.min_monotonic_ratio,
                chord_aware_monotonic=args.chord_aware_monotonic,
                chord_onset_eps=args.chord_onset_eps,
            )
        elif args.chunk_mode == "alignment_segment":
            if args.pseudo_beat_source == "xml_absolute_onset":
                if score_xml is None:
                    score_xml = MultistreamTokenizer.parse_mxl(score_xml_path)
                score_onsets = _score_onsets_from_xml(score_xml)
            elif args.pseudo_beat_source == "npz_score_times":
                score_onsets = _score_onsets_from_npz(npz, check["n_xml"])
            else:
                raise ValueError(f"Invalid pseudo_beat_source={args.pseudo_beat_source}")

            chunks = make_alignment_segment_chunks(
                n_xml=check["n_xml"],
                n_perf=check["n_perf"],
                pairs=pairs,
                score_onsets=score_onsets,
                beat_quarter_len=args.beat_quarter_len,
                beats_per_chunk=args.beats_per_chunk,
                min_segment_mxl_notes=args.min_segment_mxl_notes,
                min_mxl_notes=args.min_mxl_notes,
                min_midi_notes=args.min_midi_notes,
                min_matched_score_ratio=args.min_matched_score_ratio,
                min_matched_perf_ratio=args.min_matched_perf_ratio,
                context_perf_notes=args.context_perf_notes,
                max_midi_mxl_ratio=args.max_midi_mxl_ratio,
                max_midi_notes_per_chunk=args.max_midi_notes_per_chunk,
                min_monotonic_ratio=args.min_monotonic_ratio,
                max_unmatched_score_gap=args.max_unmatched_score_gap,
                max_unmatched_perf_gap=args.max_unmatched_perf_gap,
                max_perf_backtrack=args.max_perf_backtrack,
            )
        else:
            raise ValueError(f"Invalid chunk_mode={args.chunk_mode}")
        if len(chunks["midi"]) == 0:
            raise RuntimeError("no chunks passed chunk-level filters")

        chunk_path = default_chunk_path(
            row,
            args.out_dir,
            args.pianocore_root,
            args.chunk_location,
            args.chunk_suffix,
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
            "n_alignment_segments": len(set(chunks.get("alignment_segment_id", []))),
            "chunk_mode": args.chunk_mode,
            "chunk_monotonic_mode": chunks.get("chunk_monotonic_mode", "raw_note_order"),
            "chord_onset_eps": chunks.get("chord_onset_eps", ""),
            "n_chord_monotonic_rescued": chunks.get("n_chord_monotonic_rescued", 0),
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
    ap.add_argument(
        "--chunk-suffix",
        default="_pianocore_chunks.json",
        help="Suffix appended to the performance MIDI stem when writing chunk jsons.",
    )
    ap.add_argument("--out-manifest", default="/mnt/ssd/hbli/datasets/pianocore/pm2s_pianocore_manifest_with_chunks.csv")
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument(
        "--manifest-status-filter",
        default="",
        help="If set, keep only input manifest rows whose chunk_status equals this value, e.g. ok.",
    )
    ap.add_argument(
        "--chunk-mode",
        choices=["note_count", "score_time", "score_time_padding", "alignment_segment"],
        default="note_count",
        help=(
            "note_count keeps the old 512-target-note chunks; score_time builds pseudo-beat chunks "
            "from score onsets; score_time_padding keeps rejected non-empty bins as padding-only "
            "chunks; alignment_segment builds pseudo-beat chunks and assigns continuous segment ids "
            "so training cannot concatenate across rejected local regions."
        ),
    )
    ap.add_argument("--note-chunk-size", type=int, default=512)
    ap.add_argument("--beat-quarter-len", type=float, default=1.0)
    ap.add_argument("--beats-per-chunk", type=int, default=1)
    ap.add_argument(
        "--segment-window-notes",
        type=int,
        default=512,
        help="Deprecated compatibility option; ignored by alignment_segment mode.",
    )
    ap.add_argument(
        "--segment-window-overlap",
        type=int,
        default=0,
        help="Deprecated compatibility option; ignored by alignment_segment mode.",
    )
    ap.add_argument(
        "--min-segment-mxl-notes",
        type=int,
        default=0,
        help="Minimum total XML notes for a retained continuous pseudo-beat segment.",
    )
    ap.add_argument(
        "--max-unmatched-score-gap",
        type=int,
        default=16,
        help="Break a continuous segment when adjacent accepted chunks skip more score notes than this.",
    )
    ap.add_argument(
        "--max-unmatched-perf-gap",
        type=int,
        default=64,
        help="Break a continuous segment when adjacent accepted chunks skip more performance notes than this.",
    )
    ap.add_argument(
        "--max-perf-backtrack",
        type=int,
        default=8,
        help="Break a continuous segment when adjacent accepted chunks overlap/backtrack by more notes than this.",
    )
    ap.add_argument(
        "--pseudo-beat-source",
        choices=["xml_absolute_onset", "npz_score_times"],
        default="xml_absolute_onset",
    )
    ap.add_argument("--min-mxl-notes", type=int, default=0)
    ap.add_argument("--min-midi-notes", type=int, default=0)
    ap.add_argument("--min-matched-score-ratio", type=float, default=0.50)
    ap.add_argument("--min-matched-perf-ratio", type=float, default=0.30)
    ap.add_argument("--context-perf-notes", type=int, default=0)
    ap.add_argument("--max-midi-mxl-ratio", type=float, default=0.0)
    ap.add_argument("--max-midi-notes-per-chunk", type=int, default=0)
    ap.add_argument("--min-monotonic-ratio", type=float, default=0.0)
    ap.add_argument(
        "--chord-aware-monotonic",
        action="store_true",
        help=(
            "For score_time_padding, ignore aligned performance-index permutations "
            "inside XML notes sharing one absolute_onset."
        ),
    )
    ap.add_argument(
        "--chord-onset-eps",
        type=float,
        default=1e-6,
        help="QuarterLength tolerance used to form one score-onset group.",
    )
    ap.add_argument(
        "--reuse-identity-cache",
        action="store_true",
        help="Trust previous strict identity results and skip score-MIDI vs parse_mxl rechecking for trusted rows.",
    )
    ap.add_argument(
        "--identity-cache-manifest",
        default="",
        help="Previous chunker manifest. Rows with chunk_status == ok are trusted identity-pass rows.",
    )
    ap.add_argument(
        "--identity-cache-chunk-suffix",
        default="",
        help="Trust rows whose previous chunk json exists with this suffix, e.g. _pianocore_chunks.json.",
    )
    ap.add_argument(
        "--identity-cache-chunk-location",
        choices=["next_to_midi", "out_dir"],
        default="next_to_midi",
    )
    ap.add_argument(
        "--identity-cache-out-dir",
        default="",
        help="Out dir for previous chunk jsons when --identity-cache-chunk-location out_dir is used.",
    )
    ap.add_argument("--require-score-identity", action="store_true", default=True)
    ap.add_argument("--no-require-score-identity", dest="require_score_identity", action="store_false")
    args = ap.parse_args()

    if args.chord_aware_monotonic and args.chunk_mode != "score_time_padding":
        raise ValueError("--chord-aware-monotonic requires --chunk-mode score_time_padding")

    if args.min_mxl_notes <= 0:
        if args.chunk_mode == "note_count":
            args.min_mxl_notes = 16
        elif args.chunk_mode in {"score_time", "score_time_padding", "alignment_segment"}:
            args.min_mxl_notes = 1
        else:
            args.min_mxl_notes = 64
    if args.min_midi_notes <= 0:
        args.min_midi_notes = 16 if args.chunk_mode == "note_count" else 1
    if args.min_segment_mxl_notes <= 0:
        args.min_segment_mxl_notes = 64 if args.chunk_mode == "alignment_segment" else 0
    if args.max_midi_mxl_ratio <= 0:
        args.max_midi_mxl_ratio = 0.0 if args.chunk_mode == "note_count" else 4.0
    if args.max_midi_notes_per_chunk <= 0:
        if args.chunk_mode == "note_count":
            args.max_midi_notes_per_chunk = 0
        else:
            args.max_midi_notes_per_chunk = 128
    if args.min_monotonic_ratio <= 0:
        if args.chunk_mode == "note_count":
            args.min_monotonic_ratio = 0.0
        elif args.chunk_mode in {"score_time", "score_time_padding"}:
            args.min_monotonic_ratio = 0.95
        else:
            # Pseudo-beat chunks are tiny, so strict local monotonicity splits
            # ordinary polyphonic passages into many unusably short segments.
            args.min_monotonic_ratio = 0.01

    args.identity_cache_keys = _load_identity_cache_keys(args.identity_cache_manifest)

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.manifest, low_memory=False)
    required_cols = {"performance_midi_path", "score_xml_path", "score_midi_path", "raw_alignment_path"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"manifest missing columns: {missing}")
    if args.manifest_status_filter:
        if "chunk_status" not in df.columns:
            raise ValueError("--manifest-status-filter requires an input manifest with a chunk_status column")
        df = df[df["chunk_status"].astype(str).eq(args.manifest_status_filter)].copy()
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
        if "n_chord_monotonic_rescued" in ok_df.columns:
            print(
                "chord-aware monotonic rescued chunks:",
                int(ok_df["n_chord_monotonic_rescued"].fillna(0).sum()),
            )


if __name__ == "__main__":
    main()
