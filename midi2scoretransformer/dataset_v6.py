"""
This file contains a torch.utils.data.Dataset wrapper for the
ASAP dataset, which is a collection of MIDI files and corresponding MusicXML files.

The first run is significantly slower as metadata and caches are built.
Subsequent runs are much faster.
"""
from functools import lru_cache
from tqdm import tqdm
from joblib import Parallel, delayed
import hashlib
import json
import os
import random
from typing import Dict, Optional, Tuple, Union

import pandas as pd
import torch
from music21 import key, pitch
from torch.utils.data import Dataset

from constants import SKIP, TEST_PIECE_IDS, TO_IGNORE_INDICES
from utils import cat_dict, cut_pad
from pathlib import Path
from tokenizer_v2 import MultistreamTokenizer

import signal
from contextlib import contextmanager


def _pianocore_path_variants(path_like: str) -> list[str]:
    s = str(path_like).replace("\\", "/").strip().lstrip("./")
    if not s or s.lower() in {"nan", "none", "null"}:
        return []

    variants: list[str] = []

    def add(x: str):
        x = x.replace("\\", "/").strip().lstrip("./")
        if x and x not in variants:
            variants.append(x)

    add(s)
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

class ASAPDataset(Dataset):
    """Implements a torch-compatible interface to the ASAP Dataset"""
    def __init__(
        self,
        data_dir: str = "/mnt/ssd/hbli/datasets/PM2S_dataset/midi2scoretransformer/",
        split: str = "train",
        seq_length: Optional[int] = None,
        cache: bool = True,
        padding: str = 'per-beat',
        augmentations: Dict[str, Union[float, Dict[str, float]]] = {},
        return_continous: bool = False,
        return_paths: bool = False,
        id: str="diffusion_2024_04_18",
    ):
        """
        Parameters
        ----------
        data_dir : str (default='./data/')
            Path to the data directory
        split : str (default='train')
            Which split to use. One of ["all", "train", "validation", "test"]
        seq_length : Optional[int]
            If not None, will cut/pad to this length
        cache : bool (default=True)
            Whether to cache the parsed MIDI/MXL files
        padding : str (default='per-beat')
            How to pad the data. One of ["per-beat", "end", None]
        augmentations : Dict[str, Union[float, Dict[str, float]]]
            Augmentations to apply to the data. If a key is not given, the augmentation
            is ignored.
            Possible keys are ["transpose", "tempo_jitter", "onset_jitter", "random_crop", "random_shift"]
            - transpose: int
                Whether to transpose the data by a random amount up to the given value.
            - random_crop: Union[bool, int]
                Whether to crop the data to a random length between 16
            - tempo_jitter: Tuple[float, float]
                Whether to jitter the tempo by a random amount between the given values.
            - onset_jitter: float
                Whether to jitter the onset by a random amount according to the given value.
                Multiplicative (the intra-onset intervals are scaled by N(1,onset_jitter^2)).
        return_continous : bool (default=False)
            Whether to return the data as continous values, or as a dictionary of bucketed tensors.
        id : str (default="diffusion_2023_10_13")
            A unique identifier for the dataset. This is used to ensure that the cache is not
            reused between different datasets.
        """
        # Get metadata
        self.data_dir = data_dir
        self.split = split
        self.seq_length = seq_length
        self.cache = cache
        assert padding in ('per-beat', 'end', None)
        self.padding = padding
        self.augmentations = augmentations
        self.return_continous = return_continous
        self.return_paths = return_paths
        self.id = id
        self.metadata = self._load_metadata(data_dir, split)

        if self.cache:
            os.makedirs(os.path.join(data_dir, "cache"), exist_ok=True)

    def _load_metadata(self, data_dir: str, split: str) -> pd.DataFrame:
        data_real = pd.read_csv(data_dir + "/ACPAS-dataset/metadata_R.csv")
        data_synthetic = pd.read_csv(data_dir + "/ACPAS-dataset/metadata_S.csv")
        asap_annotations = json.load(
            open(data_dir + "/asap-dataset/asap_annotations.json")
        )
        UNALIGNED = set(
            "{ASAP}/" + k
            for k, v in asap_annotations.items()
            if not v["score_and_performance_aligned"]
        )
        # Filter
        data = pd.concat([data_real, data_synthetic])
        # Initial filtering
        data = data[(data["source"] == "ASAP") & data["aligned"]]
        data = data[~data["performance_MIDI_external"].isin(SKIP)]
        data = data[~data["performance_MIDI_external"].isin(UNALIGNED)]
        data = data.drop_duplicates(subset=["performance_MIDI_external"])
        # Filter by annotations
        data.reset_index(inplace=True)
        data.drop(TO_IGNORE_INDICES, inplace=True)

        # Select first piece from each composer for testing (may have multiple performances)
        # test_ids = data.groupby("composer").first()["piece_id"].values
        test_idx = data["piece_id"].isin(TEST_PIECE_IDS)

        if split == "all":
            return data
        elif split == "test":
            return data[test_idx]
        elif split == "validation":
            return data[(data["piece_id"] % 10 == 0) & (~data["piece_id"].isin(TEST_PIECE_IDS))]
        elif split == "train":
            d = data[(data["piece_id"] % 10 != 0) & (~data["piece_id"].isin(TEST_PIECE_IDS))]
            try:
                self.lengths = []
                for idx in range(len(d)):
                    sample = d.iloc[idx]
                    sample_path = sample["performance_MIDI_external"].replace(
                        "{ASAP}", f"{self.data_dir}asap-dataset"
                    )
                    # fmt: off
                    pkl_file = os.path.join(self.data_dir, "cache", f"{sha256(sample_path + self.id)}.pkl")
                    # fmt: on
                    input_stream, output_stream = torch.load(pkl_file, weights_only=False)
                    self.lengths.append(len(input_stream['onset']))
                self.lengths = torch.FloatTensor(self.lengths)
            # When creating the cache for the first time, we don't have the lengths yet,
            # so we will just sample uniformly.
            except FileNotFoundError as e:
                self.lengths = torch.ones(len(d))
            return d
        else:
            raise ValueError(f"Invalid split: {split}")

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        if self.split == "train":
            idx: int = torch.multinomial(self.lengths, 1, replacement=True).item()
        sample = self.metadata.iloc[idx]
        sample_path = sample["performance_MIDI_external"].replace(
            "{ASAP}", f"{self.data_dir}asap-dataset"
        )
        sample_dir = os.path.dirname(sample_path)

        pkl_file = os.path.join(self.data_dir, "cache", f"{sha256(sample_path + self.id)}.pkl")

        if (not self.cache) or (not os.path.exists(pkl_file)):
            score_path = sample_dir + "/xml_score.musicxml"
            input_stream = MultistreamTokenizer.parse_midi(sample_path)
            output_stream = MultistreamTokenizer.parse_mxl(score_path)
            torch.save((input_stream, output_stream), pkl_file)

        input_stream, output_stream = torch.load(pkl_file, weights_only=False)

        if self.augmentations.get("transpose", False):
            # shift = random.randint(-6, 6)
            max_semitones = int(self.augmentations["transpose"])
            shift = random.randint(-max_semitones, max_semitones)
            input_stream["pitch"], output_stream["pitch"], output_stream["accidental"], output_stream["keysignature"] = self._transpose(
                shift,
                midi_stream=input_stream["pitch"],
                mxl_stream=output_stream["pitch"],
                accidental_stream=output_stream["accidental"],
                keysignature_stream=output_stream["keysignature"]
            )

        # tempo augmentation: alpha
        if (v := self.augmentations.get("tempo_jitter", False)):
            alpha = random.uniform(*v)
            input_stream["onset"] = input_stream["onset"] * alpha
        # duration jitter: beta
        if (v := self.augmentations.get("duration_jitter", False)):
            beta = random.uniform(*v)
            input_stream["duration"] = input_stream["duration"] * beta
        # onset jitter
        if (v := self.augmentations.get("onset_jitter", False)):
            onset = input_stream["onset"]  # (T,)
            T = onset.shape[0]
            if T > 1:
                intervals = onset[1:] - onset[:-1]                     # (T-1,)
                noise = torch.randn_like(intervals) * v + 1.0          # N(1, v^2)
                intervals_j = intervals * noise
                onset_j = torch.empty_like(onset)
                onset_j[0] = onset[0]
                onset_j[1:] = onset[0] + torch.cumsum(intervals_j, dim=0)
                input_stream["onset"] = onset_j
        # velocity jitter: not use here
        if (v := self.augmentations.get("velocity_jitter", False)):
            input_stream["velocity"] += torch.round(torch.randn(input_stream["velocity"].shape) * v).long()
            input_stream["velocity"] = torch.clamp(input_stream["velocity"], 1, 127)

        if self.return_continous:
            return input_stream, output_stream

        input_stream = MultistreamTokenizer.bucket_midi(input_stream)
        T = input_stream["pad"].shape[0]
        input_stream["unconditional"] = torch.zeros((T, 1), dtype=torch.float32)
        output_stream = MultistreamTokenizer.bucket_mxl(output_stream)

        if self.seq_length is not None:
            seq_length = self.seq_length
        else:
            # need buffer due to padding with 'per-beat' option
            seq_length = max(len(input_stream['onset']), len(output_stream['offset'])) + 256

        chunk_annots = json.load(open(sample_path.replace(".mid", "_chunks.json")))

        if (v := self.augmentations.get("random_crop", False)):
            min_beats = 16
            if v is True:
                n_0 = random.randint(0, max(len(chunk_annots["midi"]) - min_beats, 0))
            elif isinstance(v, int):
                average = sum([len(x) for x in chunk_annots["midi"]])/len(chunk_annots["midi"])
                n_0 = random.choice(range(0, max(len(chunk_annots["midi"]) - min_beats, 1), max(1, int(v/average))))
            else:
                raise ValueError("Invalid random_crop value")
        else:
            n_0 = 0

        def process_chunk(stream, chunk, padding, length):
            if padding == "per-beat":
                return {k: cut_pad(v[chunk], length, 0) for k, v in stream.items()}
            return {k: v[chunk] for k, v in stream.items()}

        new_input_stream = None  # just a sentry
        for midi_chunk, mxl_chunk in zip(
            chunk_annots["midi"][n_0:], chunk_annots["mxl"][n_0:]
        ):
            length = max(len(midi_chunk), len(mxl_chunk))
            if new_input_stream is not None and len(new_input_stream["onset"]) + length > seq_length + self.augmentations.get("random_shift", 0):
                break
            in_chunk = process_chunk(input_stream, midi_chunk, self.padding, length)
            out_chunk = process_chunk(output_stream, mxl_chunk, self.padding, length)
            if new_input_stream is None:
                new_input_stream = in_chunk
                new_output_stream = out_chunk
            else:
                new_input_stream = cat_dict(new_input_stream, in_chunk)
                new_output_stream = cat_dict(new_output_stream, out_chunk)
        if (v := self.augmentations.get("random_shift", False)):
            shift = random.randint(0, v - 1)
            for k, v in new_input_stream.items():
                new_input_stream[k] = v[shift:]
            for k, v in new_output_stream.items():
                new_output_stream[k] = v[shift:]
        if self.padding is not None:
            # Cut/Pad to exact seq-length
            for k, v in new_input_stream.items():
                input_stream[k] = cut_pad(v, seq_length, 0)
            for k, v in new_output_stream.items():
                output_stream[k] = cut_pad(v, seq_length, 0)
        if self.return_paths:
            return input_stream, output_stream, sample_path, sample_dir + "/xml_score.musicxml"
        return input_stream, output_stream

    @lru_cache(None)
    @staticmethod
    def _accidental_map(p, a, i):
        def alter_map(accidental):
            alter_to_value = {None: 5, -2.0: 0, -1.0: 1, 0.0: 2, 1.0: 3, 2.0: 4}
            alter = accidental.alter if isinstance(accidental, pitch.Accidental) else accidental
            # 6 if not known
            return alter_to_value.get(alter, 6)

        if i is None:
            return a
        accidental_mapping = {0: 2, 1: 1, 2: 0, 3: -1, 4: -2}
        alter = accidental_mapping.get(a, 0)
        p_obj = pitch.Pitch()
        p_obj.midi = p + alter
        if a in accidental_mapping:
            p_obj.accidental = -accidental_mapping[a]
        p_obj.spellingIsInferred = False
        tp = p_obj.transpose(i)
        accepted_pitches = {
            'C', 'B#', 'D--', 'C#', 'B##', 'D-', 'D', 'C##', 'E--', 'D#', 'E-', 'F--',
            'E', 'D##', 'F-', 'F', 'E#', 'G--', 'F#', 'E##', 'G-', 'G', 'F##', 'A--',
            'G#', 'A-', 'A', 'G##', 'B--', 'A#', 'B-', 'C--', 'B', 'A##', 'C-'
        }
        if tp.name not in accepted_pitches:
            return None
        return alter_map(tp.accidental)

    @lru_cache(None)
    @staticmethod
    def _ks_map(ks: int, i: str):
        if i is None:
            return ks
        if ks == 15:
            return None
        k_obj = key.KeySignature(ks - 7)
        ns = k_obj.transpose(i).sharps
        if not -7 <= ns <= 7:
            return None
        return ns + 7

    @staticmethod
    def _transpose(shift, midi_stream, mxl_stream=None, accidental_stream=None, keysignature_stream=None):
        """Transpose pitches by a random amount between -6 and 6. If accidental_stream and
        keysignature_stream are provided, they will be adjusted following the procedure
        in https://arxiv.org/pdf/2107.14009.pdf

        In more detail, pitches are simply shifted by the desired amount.
        Then, all musical intervals with that shift are tried.
        If the transposed accidentals or key signatures are invalid, they are set to ignore_index.
        Among the valid transpositions, the one with the lowest number of accidentals is selected.

        Parameters
        ----------
        shift : int
            The amount of transposition
        midi_stream : torch.Tensor
            The MIDI pitch stream
        mxl_stream : torch.Tensor|None
            The MusicXML pitch stream, if provided.
        accidental_stream : torch.Tensor|None
            The accidental stream, if provided.
        keysignature_stream : torch.Tensor|None
            The key signature stream, if provided.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            The transposed MIDI pitch stream, the transposed MusicXML pitch stream
            the transposed accidental stream, and the transposed key signature stream.
        """
        assert (accidental_stream is None) == (keysignature_stream is None), "Either both or none of the accidentals and key signatures should be provided"
        if accidental_stream is not None:
            assert mxl_stream is not None, "Need mxl pitch stream alongside accidentals"

        def shift_pitch(stream, shift):
            stream = stream + shift
            stream[stream > 127] -= 12
            stream[stream < 0] += 12
            return stream

        midi_stream = shift_pitch(midi_stream, shift)
        results = [midi_stream]
        if mxl_stream is not None and accidental_stream is not None and keysignature_stream is not None:
            shift_pc = ((shift + 6) % 12) - 6
            if shift == 0 and random.random() < 0.5:
                # Always include the original version some of the time.
                pass
            else:
                INTERVALS = {
                    -6: ["d5", "A4"],
                    -5: ["P5", "d6", "AA4"],
                    -4: ["m6", "A5"],
                    -3: ["M6", "d7", "AA5"],
                    -2: ["m7", "A6"],
                    -1: ["d1", "M7", "AA6"],
                    0: [None, "P1", "d2", "A7"],
                    1: ["m2", "A1"],
                    2: ["M2", "d3", "AA1"],
                    3: ["m3", "A2"],
                    4: ["M3", "d4", "AA2"],
                    5: ["P4", "A3"],
                    6: ["d5", "A4"],
                }
                intervals = INTERVALS[shift_pc]
                m = mxl_stream.numpy()
                a = accidental_stream.numpy()
                ks = keysignature_stream.unique()
                best_error = float('inf')
                errors = dict()
                for interv in intervals:
                    accidental_cand = torch.zeros_like(accidental_stream)
                    keysignature_cand: torch.Tensor = keysignature_stream.clone()
                    for k in ks:
                        val = ASAPDataset._ks_map(int(k), interv)
                        if val is None:  # invalid key signature
                            accidental_cand.fill_(6)
                            keysignature_cand.fill_(15)
                            break
                        keysignature_cand[keysignature_cand == k] = val
                    else:  # valid keysignatures, so we look for accidentals
                        for i in range(len(mxl_stream)):
                            val = ASAPDataset._accidental_map(m[i], a[i], interv)
                            if val is None:  # invalid accidental
                                accidental_cand.fill_(6)
                                keysignature_cand.fill_(15)
                                break
                            accidental_cand[i] = val * 1.0
                    error = (accidental_cand[accidental_cand != 5] - 2).abs().sum()
                    errors[interv] = error
                    # print(f"Error: {error} for {interv}", accidental_cand)
                    if error < best_error:
                        best_error = error
                        accidental_stream = accidental_cand
                        keysignature_stream = keysignature_cand
        if mxl_stream is not None:
            mxl_stream = shift_pitch(mxl_stream, shift)
            results.append(mxl_stream)
        if accidental_stream is not None:
            results.append(accidental_stream)
            results.append(keysignature_stream)
        return results


class PianoCoReDataset(Dataset):
    """
    PianoCoRe paired dataset for MIDI2ScoreTransformer.

    Expected manifest columns:
      - performance_midi_path: raw performance MIDI, model input x
      - score_xml_path: raw MusicXML/MXL, model target y
      - chunk_path: PianoCoRe chunk json generated by chunker_pianocore.py

    Optional columns:
      - score_midi_path, raw_alignment_path, performance_dataset, score_dataset,
        raw_recall, raw_precision, pm2s_split/split.

    This class deliberately follows ASAPDataset.__getitem__:
      raw files -> parse_midi/parse_mxl cache -> bucket_midi/bucket_mxl
      -> read chunk json -> concatenate chunks -> cut/pad.
    """

    def __init__(
        self,
        manifest_path: str,
        pianocore_root: str = "/mnt/ssd/hbli/datasets/pianocore/",
        split: str = "train",
        seq_length: Optional[int] = None,
        cache: bool = True,
        cache_dir: str = "",
        padding: str = "per-beat",
        augmentations: Dict[str, Union[float, Dict[str, float]]] = {},
        return_continous: bool = False,
        return_paths: bool = False,
        id: str = "pianocore_paired_v1_2026_05_18",
        split_col: str = "",
        require_chunk_file: bool = True,
        max_rows: int = 0,
        skip_on_error: bool = True,
        max_retries: int = 50,
        fail_log_path: str = "",
        fail_log_flush: bool = True,
        respect_alignment_segments: bool = True,
        sample_unit: str = "row",
        min_valid_start_len: int = 384,
        max_per_score: int = 0,
        score_balanced_sampling: bool = False,
        score_key_col: str = "score_xml_path",
        pad_bad_chunks: bool = False,
    ):
        super().__init__()
        assert padding in ("per-beat", "end", None)
        if sample_unit not in {"row", "chunk", "valid_chunk_start"}:
            raise ValueError(
                f"Invalid PianoCoRe sample_unit={sample_unit}. "
                "Must be 'row', 'chunk', or 'valid_chunk_start'."
            )
        self.manifest_path = str(manifest_path)
        self.pianocore_root = str(Path(pianocore_root).resolve())
        self.split = split
        self.seq_length = seq_length
        self.cache = bool(cache)
        self.padding = padding
        self.augmentations = augmentations or {}
        self.return_continous = bool(return_continous)
        self.return_paths = bool(return_paths)
        self.id = id
        self.require_chunk_file = bool(require_chunk_file)
        self.max_rows = int(max_rows)
        self.skip_on_error = bool(skip_on_error)
        self.max_retries = int(max_retries)
        self.fail_log_flush = bool(fail_log_flush)
        self.respect_alignment_segments = bool(respect_alignment_segments)
        self.sample_unit = sample_unit
        self.min_valid_start_len = int(min_valid_start_len)
        self.max_per_score = int(max_per_score)
        self.score_balanced_sampling = bool(score_balanced_sampling)
        self.score_key_col = str(score_key_col)
        self.pad_bad_chunks = bool(pad_bad_chunks)
        if self.sample_unit == "valid_chunk_start" and self.min_valid_start_len <= 0:
            raise ValueError("--pianocore_min_valid_start_len must be positive for valid_chunk_start sampling")

        self.metadata = self._load_manifest(self.manifest_path, split, split_col)
        if len(self.metadata) == 0:
            raise ValueError(f"PianoCoReDataset has no rows for split={split}: {manifest_path}")

        if not cache_dir:
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(self.manifest_path)), "cache_pianocore")
        self.cache_dir = cache_dir
        if self.cache:
            os.makedirs(self.cache_dir, exist_ok=True)

        if not fail_log_path:
            fail_log_path = os.path.join(self.cache_dir, "pianocore_dataset_failures.log")
        self.fail_log_path = fail_log_path
        os.makedirs(os.path.dirname(os.path.abspath(self.fail_log_path)), exist_ok=True)
        self.sample_index = self._build_sample_index()
        self.row_to_sample_entries = self._build_row_to_sample_entries()
        self.score_groups = self._build_score_groups()

    def _load_manifest(self, manifest_path: str, split: str, split_col: str) -> pd.DataFrame:
        df = pd.read_csv(manifest_path, low_memory=False)
        required = {"performance_midi_path", "score_xml_path"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"PianoCoRe manifest missing required columns: {missing}")

        if "chunk_path" not in df.columns:
            # fallback only; recommended chunker_pianocore.py writes chunk_path explicitly
            df["chunk_path"] = df["performance_midi_path"].map(
                lambda x: str(Path(str(x)).with_suffix("")) + "_pianocore_chunks.json"
            )

        if split != "all":
            if split_col and split_col in df.columns:
                df = df[df[split_col].astype(str).eq(split)]
            elif "pm2s_split" in df.columns:
                df = df[df["pm2s_split"].astype(str).eq(split)]
            elif "split" in df.columns:
                df = df[df["split"].astype(str).eq(split)]
            # If no split column exists, keep all rows. The manifest can already be train-only.

        df = df[df["performance_midi_path"].notna() & df["score_xml_path"].notna() & df["chunk_path"].notna()]
        if self.require_chunk_file:
            exists = df["chunk_path"].map(lambda p: os.path.exists(self._resolve_path(p)))
            df = df[exists]
        if self.max_rows and self.max_rows > 0:
            df = df.head(self.max_rows).copy()
        if self.max_per_score and self.max_per_score > 0:
            key_col = self.score_key_col if self.score_key_col in df.columns else "score_xml_path"
            df = (
                df.sort_values([key_col, "id"] if "id" in df.columns else [key_col])
                .groupby(key_col, sort=False, group_keys=False)
                .head(self.max_per_score)
            )

        df = df.reset_index(drop=True)
        return df

    def __len__(self) -> int:
        if self.score_balanced_sampling:
            return len(self.score_groups)
        if self.sample_index is not None:
            return len(self.sample_index)
        return len(self.metadata)

    def _start_effective_len(
        self,
        chunk_annots: Dict[str, list],
        start_chunk: int,
    ) -> int:
        n_chunks = min(len(chunk_annots["midi"]), len(chunk_annots["mxl"]))
        segment_ids = self._alignment_segment_ids(chunk_annots)
        active_segment_id = None
        if self.respect_alignment_segments and segment_ids is not None:
            active_segment_id = segment_ids[start_chunk]

        seq_length = self.seq_length
        total = 0
        for chunk_i, (midi_chunk, mxl_chunk) in enumerate(
            zip(chunk_annots["midi"][start_chunk:n_chunks], chunk_annots["mxl"][start_chunk:n_chunks]),
            start=start_chunk,
        ):
            if active_segment_id is not None and segment_ids[chunk_i] != active_segment_id:
                break
            length = max(len(midi_chunk), len(mxl_chunk))
            if length == 0:
                continue
            if seq_length is not None and total > 0 and total + length > seq_length:
                break
            total += length
            if seq_length is not None and total >= seq_length:
                return int(seq_length)
        return int(total)

    def _build_sample_index(self) -> Optional[list[dict[str, int]]]:
        if self.sample_unit == "row":
            return None

        sample_index: list[dict[str, int]] = []
        n_candidate_starts = 0
        n_dropped_short_starts = 0
        for row_idx, sample in self.metadata.iterrows():
            row_id = str(sample.get("id", row_idx))
            try:
                chunk_path = self._resolve_path(sample["chunk_path"])
                chunk_annots = self._load_chunks(chunk_path)
                n_chunks = min(len(chunk_annots["midi"]), len(chunk_annots["mxl"]))
                for chunk_idx in range(n_chunks):
                    if len(chunk_annots["midi"][chunk_idx]) == 0 or len(chunk_annots["mxl"][chunk_idx]) == 0:
                        continue
                    if self.pad_bad_chunks and not self._chunk_is_trainable(chunk_annots, chunk_idx):
                        continue
                    n_candidate_starts += 1
                    effective_len = None
                    if self.sample_unit == "valid_chunk_start":
                        effective_len = self._start_effective_len(chunk_annots, chunk_idx)
                        if effective_len < self.min_valid_start_len:
                            n_dropped_short_starts += 1
                            continue
                    entry = {"row_idx": int(row_idx), "chunk_idx": int(chunk_idx)}
                    if effective_len is not None:
                        entry["effective_len"] = int(effective_len)
                    sample_index.append(entry)
            except Exception as e:
                if not self.skip_on_error:
                    raise
                self._log_failure(row_id, e)

        if len(sample_index) == 0:
            raise ValueError(f"PianoCoReDataset sample_unit={self.sample_unit} produced no sample units")
        self.n_candidate_starts = int(n_candidate_starts)
        self.n_dropped_short_starts = int(n_dropped_short_starts)
        return sample_index

    def _build_row_to_sample_entries(self) -> dict[int, list[dict[str, int]]]:
        if self.sample_index is None:
            return {}
        out: dict[int, list[dict[str, int]]] = {}
        for entry in self.sample_index:
            out.setdefault(int(entry["row_idx"]), []).append(entry)
        return out

    def _build_score_groups(self) -> list[list[int]]:
        if not self.score_balanced_sampling:
            return []
        key_col = self.score_key_col if self.score_key_col in self.metadata.columns else "score_xml_path"
        row_pool: Optional[set[int]] = None
        if self.sample_index is not None:
            row_pool = set(self.row_to_sample_entries.keys())

        groups: list[list[int]] = []
        for _, group in self.metadata.groupby(key_col, sort=False):
            rows = [int(i) for i in group.index]
            if row_pool is not None:
                rows = [i for i in rows if i in row_pool]
            if rows:
                groups.append(rows)
        if not groups:
            raise ValueError("PianoCoRe score-balanced sampling produced no score groups")
        return groups

    def _row_and_start_chunk(self, idx: int) -> tuple[int, Optional[int]]:
        if self.score_balanced_sampling:
            rows = self.score_groups[int(idx) % len(self.score_groups)]
            if self.sample_index is None:
                return int(random.choice(rows)), None
            row_idx = int(random.choice(rows))
            starts = self.row_to_sample_entries.get(row_idx, [])
            if not starts:
                return row_idx, None
            entry = random.choice(starts)
            return int(entry["row_idx"]), int(entry["chunk_idx"])
        if self.sample_index is None:
            return int(idx), None
        entry = self.sample_index[int(idx)]
        return int(entry["row_idx"]), int(entry["chunk_idx"])

    def _row_id_for_item(self, idx: int) -> str:
        row_idx, _ = self._row_and_start_chunk(idx)
        return str(self.metadata.iloc[row_idx].get("id", row_idx))

    def _resolve_path(self, path_like: str) -> str:
        p = Path(str(path_like))
        if p.is_absolute():
            return str(p)
        root = Path(self.pianocore_root)
        for variant in _pianocore_path_variants(str(path_like)):
            candidate = root / variant
            if candidate.exists():
                return str(candidate)
        variants = _pianocore_path_variants(str(path_like))
        if variants:
            base = variants[0]
            for prefix in ("PianoCoRe/raw/", "PianoCoRe/", "raw/"):
                if base.startswith(prefix):
                    base = base[len(prefix):]
                    break
            return str(root / "PianoCoRe" / "raw" / base)
        return str(root / p)

    def _cache_path(self, performance_midi_path: str, score_xml_path: str) -> str:
        key = performance_midi_path + "||" + score_xml_path + "||" + self.id
        return os.path.join(self.cache_dir, f"{sha256(key)}.pkl")

    def _log_failure(self, row_id: str, err: Exception):
        try:
            line = f"{row_id}\t{type(err).__name__}\t{str(err).replace(chr(10), ' ')}\n"
            with open(self.fail_log_path, "a", encoding="utf-8") as f:
                f.write(line)
                if self.fail_log_flush:
                    f.flush()
        except Exception:
            pass

    def _load_or_build_cache(self, performance_midi_path: str, score_xml_path: str):
        pkl_file = self._cache_path(performance_midi_path, score_xml_path)
        if self.cache and os.path.exists(pkl_file):
            return torch.load(pkl_file, weights_only=False)

        input_stream = MultistreamTokenizer.parse_midi(performance_midi_path)
        try:
            with _time_limit(60):
                output_stream = MultistreamTokenizer.parse_mxl(score_xml_path)
        except _ParseTimeout as e:
            raise RuntimeError(f"HARD_TIMEOUT_60S: {score_xml_path}") from e

        if self.cache:
            torch.save((input_stream, output_stream), pkl_file)
        return input_stream, output_stream

    def _load_chunks(self, chunk_path: str) -> Dict[str, list]:
        if not os.path.exists(chunk_path):
            raise FileNotFoundError(f"PianoCoRe chunk json not found: {chunk_path}")
        with open(chunk_path, "r", encoding="utf-8") as f:
            chunk_annots = json.load(f)
        if "midi" not in chunk_annots or "mxl" not in chunk_annots:
            raise ValueError(f"Invalid chunk json without midi/mxl keys: {chunk_path}")
        if len(chunk_annots["midi"]) == 0 or len(chunk_annots["mxl"]) == 0:
            raise ValueError(f"Empty chunk json: {chunk_path}")
        return chunk_annots

    def _choose_start_chunk(self, chunk_annots: Dict[str, list]) -> int:
        if not (v := self.augmentations.get("random_crop", False)):
            return 0
        n_chunks = len(chunk_annots["midi"])
        if n_chunks <= 1:
            return 0
        choices_all = [
            i for i in range(n_chunks)
            if (not self.pad_bad_chunks) or self._chunk_is_trainable(chunk_annots, i)
        ]
        if not choices_all:
            choices_all = list(range(n_chunks))
        if v is True:
            return random.choice(choices_all)
        if isinstance(v, int):
            avg = sum(len(x) for x in chunk_annots["midi"]) / max(n_chunks, 1)
            step = max(1, int(v / max(avg, 1)))
            choices = [i for i in range(0, n_chunks, step) if i in set(choices_all)]
            if not choices:
                choices = choices_all
            return random.choice(choices)
        raise ValueError("Invalid random_crop value")

    @staticmethod
    def _alignment_segment_ids(chunk_annots: Dict[str, list]) -> Optional[list]:
        segment_ids = chunk_annots.get("alignment_segment_id")
        if segment_ids is None:
            return None
        n_chunks = len(chunk_annots["midi"])
        if len(segment_ids) != n_chunks:
            raise ValueError(
                f"Invalid alignment_segment_id length={len(segment_ids)} for n_chunks={n_chunks}"
            )
        return segment_ids

    @staticmethod
    def _chunk_is_trainable(chunk_annots: Dict[str, list], chunk_i: int) -> bool:
        flags = chunk_annots.get("chunk_is_trainable")
        if flags is None:
            flags = chunk_annots.get("chunk_trainable")
        if flags is None:
            return True
        if chunk_i >= len(flags):
            raise ValueError(
                f"Invalid chunk_is_trainable length={len(flags)} for chunk_i={chunk_i}"
            )
        return bool(flags[chunk_i])

    @staticmethod
    def _chunk_pad_len(chunk_annots: Dict[str, list], chunk_i: int, default_length: int) -> int:
        lengths = chunk_annots.get("chunk_pad_len")
        if lengths is None:
            lengths = chunk_annots.get("pad_len")
        if lengths is None or chunk_i >= len(lengths):
            return max(int(default_length), 1)
        return max(int(lengths[chunk_i]), 1)

    @staticmethod
    def _process_chunk(stream: Dict[str, torch.Tensor], chunk: list, padding, length: int):
        if len(chunk) == 0:
            # Empty side in a paired chunk is not useful for this model.
            raise ValueError("Empty midi/mxl chunk")
        idx = torch.as_tensor(chunk, dtype=torch.long)
        if padding == "per-beat":
            return {k: cut_pad(v[idx], length, 0) for k, v in stream.items()}
        return {k: v[idx] for k, v in stream.items()}

    @staticmethod
    def _process_padding_chunk(stream: Dict[str, torch.Tensor], length: int):
        return {
            k: torch.zeros((length, *v.shape[1:]), dtype=v.dtype, device=v.device)
            for k, v in stream.items()
        }

    def _get_one(self, idx: int):
        row_idx, fixed_start_chunk = self._row_and_start_chunk(idx)
        sample = self.metadata.iloc[row_idx]
        row_id = str(sample.get("id", row_idx))
        performance_midi_path = self._resolve_path(sample["performance_midi_path"])
        score_xml_path = self._resolve_path(sample["score_xml_path"])
        chunk_path = self._resolve_path(sample["chunk_path"])

        input_stream, output_stream = self._load_or_build_cache(performance_midi_path, score_xml_path)

        if self.augmentations.get("transpose", False):
            max_semitones = int(self.augmentations["transpose"])
            shift = random.randint(-max_semitones, max_semitones)
            input_stream = dict(input_stream)
            output_stream = dict(output_stream)
            input_stream["pitch"], output_stream["pitch"], output_stream["accidental"], output_stream["keysignature"] = ASAPDataset._transpose(
                shift,
                midi_stream=input_stream["pitch"].clone(),
                mxl_stream=output_stream["pitch"].clone(),
                accidental_stream=output_stream["accidental"].clone(),
                keysignature_stream=output_stream["keysignature"].clone(),
            )

        if (v := self.augmentations.get("tempo_jitter", False)):
            input_stream = dict(input_stream)
            input_stream["onset"] = input_stream["onset"] * random.uniform(*v)
        if (v := self.augmentations.get("duration_jitter", False)):
            input_stream = dict(input_stream)
            input_stream["duration"] = input_stream["duration"] * random.uniform(*v)
        if (v := self.augmentations.get("onset_jitter", False)):
            input_stream = dict(input_stream)
            onset = input_stream["onset"]
            if onset.shape[0] > 1:
                intervals = onset[1:] - onset[:-1]
                noise = torch.randn_like(intervals) * v + 1.0
                onset_j = torch.empty_like(onset)
                onset_j[0] = onset[0]
                onset_j[1:] = onset[0] + torch.cumsum(intervals * noise, dim=0)
                input_stream["onset"] = onset_j
        if (v := self.augmentations.get("velocity_jitter", False)):
            input_stream = dict(input_stream)
            input_stream["velocity"] += torch.round(torch.randn(input_stream["velocity"].shape) * v).long()
            input_stream["velocity"] = torch.clamp(input_stream["velocity"], 1, 127)

        if self.return_continous:
            return input_stream, output_stream

        input_stream = MultistreamTokenizer.bucket_midi(input_stream)
        T = input_stream["pad"].shape[0]
        input_stream["unconditional"] = torch.zeros((T, 1), dtype=torch.float32)
        output_stream = MultistreamTokenizer.bucket_mxl(output_stream)

        if self.seq_length is not None:
            seq_length = self.seq_length
        else:
            seq_length = max(len(input_stream["onset"]), len(output_stream["offset"])) + 256

        chunk_annots = self._load_chunks(chunk_path)
        n_0 = fixed_start_chunk if fixed_start_chunk is not None else self._choose_start_chunk(chunk_annots)
        segment_ids = self._alignment_segment_ids(chunk_annots)
        active_segment_id = None
        if self.respect_alignment_segments and segment_ids is not None:
            active_segment_id = segment_ids[n_0]

        new_input_stream = None
        new_output_stream = None
        extra_shift = int(self.augmentations.get("random_shift", 0) or 0)
        for chunk_i, (midi_chunk, mxl_chunk) in enumerate(
            zip(chunk_annots["midi"][n_0:], chunk_annots["mxl"][n_0:]),
            start=n_0,
        ):
            if active_segment_id is not None and segment_ids[chunk_i] != active_segment_id:
                break
            is_trainable = (not self.pad_bad_chunks) or self._chunk_is_trainable(chunk_annots, chunk_i)
            length = max(len(midi_chunk), len(mxl_chunk))
            if not is_trainable:
                length = self._chunk_pad_len(chunk_annots, chunk_i, length)
            if length == 0:
                continue
            if new_input_stream is not None and len(new_input_stream["onset"]) + length > seq_length + extra_shift:
                break
            if is_trainable:
                in_chunk = self._process_chunk(input_stream, midi_chunk, self.padding, length)
                out_chunk = self._process_chunk(output_stream, mxl_chunk, self.padding, length)
            else:
                in_chunk = self._process_padding_chunk(input_stream, length)
                out_chunk = self._process_padding_chunk(output_stream, length)
            if new_input_stream is None:
                new_input_stream = in_chunk
                new_output_stream = out_chunk
            else:
                new_input_stream = cat_dict(new_input_stream, in_chunk)
                new_output_stream = cat_dict(new_output_stream, out_chunk)

        if new_input_stream is None or new_output_stream is None:
            raise RuntimeError(f"No usable chunks for row_id={row_id}: {chunk_path}")

        if extra_shift > 0:
            shift = random.randint(0, extra_shift - 1)
            for k, v in new_input_stream.items():
                new_input_stream[k] = v[shift:]
            for k, v in new_output_stream.items():
                new_output_stream[k] = v[shift:]

        if self.padding is not None:
            for k, v in new_input_stream.items():
                input_stream[k] = cut_pad(v, seq_length, 0)
            for k, v in new_output_stream.items():
                output_stream[k] = cut_pad(v, seq_length, 0)

        if self.return_paths:
            return input_stream, output_stream, performance_midi_path, score_xml_path
        return input_stream, output_stream

    def __getitem__(self, idx: int):
        if not self.skip_on_error:
            return self._get_one(idx)

        last_err = None
        cur = idx
        for _ in range(self.max_retries):
            row_id = self._row_id_for_item(cur)
            try:
                return self._get_one(cur)
            except Exception as e:
                last_err = e
                self._log_failure(row_id, e)
                cur = random.randint(0, len(self) - 1)
        raise RuntimeError(
            f"PianoCoReDataset: failed to get a valid sample after {self.max_retries} retries. "
            f"Last error: {repr(last_err)}"
        )


class UnpairedXMLDataset(Dataset):
    """
    Unpaired MusicXML dataset (e.g., PDMX xmls) for the 'unpaired xml' training.

    Output:
      - y: bucketed mxl streams (same format as ASAPDataset output_stream)
      - x: surrogate encoder inputs:
           pitch copied from y["pitch"], onset/duration/velocity all-zeros, pad copied from y["pad"].

    NOTE:
      - Does NOT use ASAP chunk annotations.
      - Cropping is done in note-index space.
      - Transposition uses ASAPDataset._transpose to keep pitch/accidental/keysignature consistent.
    """

    def __init__(
        self,
        mxl_paths: list[str] | str,
        seq_length: int = 512,
        cache: bool = True,
        cache_dir: str = "",
        augmentations: Dict[str, Union[float, Dict[str, float]]] = {},
        return_paths: bool = False,
        id: str = "unpaired_xml_musicrender_2026_05_14",
        surrogate_mode: str = "pitch_onset_duration",
        mxl_root: str = "/mnt/ssd/hbli/datasets/PDMX/mxl",
        rendered_midi_root: str = "/mnt/ssd/hbli/datasets/PDMX/render",
        # [ADDED] knobs for "skip + log bad xmls"
        skip_on_error: bool = True,
        max_retries: int = 50,
        fail_log_path: str = "",
        fail_log_flush: bool = True,
    ):
        super().__init__()
        self.seq_length = int(seq_length)
        self.cache = bool(cache)
        self.augmentations = augmentations or {}
        self.return_paths = bool(return_paths)
        self.id = id

        allowed_modes = {"pitch_only", "pitch_onset", "pitch_onset_duration"}
        if surrogate_mode not in allowed_modes:
            raise ValueError(f"Invalid surrogate_mode={surrogate_mode}. Must be one of {allowed_modes}")
        self.surrogate_mode = surrogate_mode

        self.mxl_root = str(Path(mxl_root).resolve())
        self.rendered_midi_root = str(Path(rendered_midi_root).resolve())

        # [ADDED]
        self.skip_on_error = bool(skip_on_error)
        self.max_retries = int(max_retries)
        self.fail_log_flush = bool(fail_log_flush)

        # Load list
        if isinstance(mxl_paths, str):
            # treat as txt file
            txt = mxl_paths
            if not os.path.exists(txt):
                raise FileNotFoundError(f"UnpairedXMLDataset list file not found: {txt}")
            with open(txt, "r", encoding="utf-8") as f:
                self.mxl_paths = [ln.strip() for ln in f if ln.strip()]
            if len(self.mxl_paths) == 0:
                raise ValueError(f"No mxl paths in: {txt}")
            base_dir = os.path.dirname(os.path.abspath(txt))
        else:
            self.mxl_paths = list(mxl_paths)
            if len(self.mxl_paths) == 0:
                raise ValueError("Empty mxl_paths list.")
            base_dir = os.path.dirname(os.path.abspath(self.mxl_paths[0]))

        # Cache dir
        if not cache_dir:
            cache_dir = os.path.join(base_dir, "cache_unpaired")
        self.cache_dir = cache_dir
        if self.cache:
            os.makedirs(self.cache_dir, exist_ok=True)

        # [ADDED] failure log path
        if not fail_log_path:
            fail_log_path = os.path.join(self.cache_dir, "unpaired_failures.log")
        self.fail_log_path = fail_log_path
        # ensure parent dir exists
        os.makedirs(os.path.dirname(os.path.abspath(self.fail_log_path)), exist_ok=True)

    def __len__(self):
        return len(self.mxl_paths)

    def _cache_path(self, mxl_path: str) -> str:
        return os.path.join(self.cache_dir, f"{sha256(mxl_path + self.id)}.pkl")

    def _mxl_to_rendered_midi_path(self, mxl_path: str) -> str:
        """
        Convert:
        /mnt/ssd/hbli/datasets/PDMX/mxl/1/11/xxx.mxl
        to:
        /mnt/ssd/hbli/datasets/PDMX/render/1/11/xxx.midi
        """
        mxl_path_abs = Path(mxl_path).resolve()
        mxl_root = Path(self.mxl_root).resolve()
        rendered_root = Path(self.rendered_midi_root).resolve()

        rel = mxl_path_abs.relative_to(mxl_root)
        rendered_midi_path = (rendered_root / rel).with_suffix(".midi")
        return str(rendered_midi_path)

    def _midi_cache_path(self, rendered_midi_path: str) -> str:
        return os.path.join(self.cache_dir, f"{sha256(rendered_midi_path + self.id + '_mr')}.pkl")


    def _load_continuous_midi(self, rendered_midi_path: str) -> Dict[str, torch.Tensor]:
        """
        Load continuous midi streams (NOT bucketed).
        Keys: onset/duration/pitch/velocity
        """
        pkl = self._midi_cache_path(rendered_midi_path)
        if self.cache and os.path.exists(pkl):
            return torch.load(pkl, weights_only=False)

        if not os.path.exists(rendered_midi_path):
            raise FileNotFoundError(f"Rendered MIDI not found: {rendered_midi_path}")

        cont = MultistreamTokenizer.parse_midi(rendered_midi_path)

        if self.cache:
            torch.save(cont, pkl)
        return cont

    def _log_failure(self, mxl_path: str, err: Exception):
        # minimal, append-only logging
        try:
            line = f"{mxl_path}\t{type(err).__name__}\t{str(err).replace(chr(10), ' ')}\n"
            with open(self.fail_log_path, "a", encoding="utf-8") as f:
                f.write(line)
                if self.fail_log_flush:
                    f.flush()
        except Exception:
            # never let logging crash training
            pass

    def _load_continuous_mxl(self, mxl_path: str) -> Dict[str, torch.Tensor]:
        """
        Load continuous mxl streams (NOT bucketed).
        Keys: offset/downbeat/duration/pitch/accidental/keysignature/velocity/grace/trill/staccato/voice/stem/hand
        """
        pkl = self._cache_path(mxl_path)
        if self.cache and os.path.exists(pkl):
            return torch.load(pkl, weights_only=False)

        try:
            with _time_limit(60):
                cont = MultistreamTokenizer.parse_mxl(mxl_path)
        except _ParseTimeout as e:
            self._log_failure(mxl_path, e)
            raise RuntimeError(f"HARD_TIMEOUT_60S: {mxl_path}") from e
        
        if self.cache:
            torch.save(cont, pkl)
        return cont

    def _sample_crop_indices(self, n: int) -> tuple[int, int]:
        """
        Sample a note-index crop window shared by both XML y and rendered-MIDI x.
        """
        do_crop = self.augmentations.get("random_crop", True)
        if (not do_crop) or (n <= self.seq_length) or n <= 0:
            return 0, n

        start = random.randint(0, max(n - self.seq_length, 0))
        end = start + self.seq_length
        return start, end


    def _slice_stream_dict(self, cont: Dict[str, torch.Tensor], start: int, end: int) -> Dict[str, torch.Tensor]:
        out = {}
        for k, v in cont.items():
            out[k] = v[start:end]
        return out

    def _pad_bucketed(self, y: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        After bucket_mxl: y[k] is (T,C), y["pad"] is (T,)
        pad/cut to seq_length.
        """
        T = int(y["pad"].shape[0])
        if T > self.seq_length:
            for k in y.keys():
                y[k] = y[k][: self.seq_length]
            return y

        if T < self.seq_length:
            pad_len = self.seq_length - T
            for k, v in y.items():
                if k == "pad":
                    y[k] = torch.cat([v, torch.zeros((pad_len,), dtype=v.dtype)], dim=0)
                else:
                    C = v.shape[1]
                    y[k] = torch.cat([v, torch.zeros((pad_len, C), dtype=v.dtype)], dim=0)
        return y

    def _make_surrogate_x_cont(self, midi_cont: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Build surrogate x directly from MusicRender MIDI.

        surrogate_mode:
        - pitch_only
        - pitch_onset
        - pitch_onset_duration
        """
        T = int(midi_cont["pitch"].shape[0])
        if T == 0:
            return {
                "onset": torch.FloatTensor([]),
                "duration": torch.FloatTensor([]),
                "pitch": torch.LongTensor([]),
                "velocity": torch.LongTensor([]),
            }

        onset_sec = midi_cont["onset"].clone().float()
        duration_sec = midi_cont["duration"].clone().float()

        # Important: after crop, rebase so the first note starts at 0
        onset_sec = onset_sec - onset_sec[0]

        if self.surrogate_mode == "pitch_only":
            x_onset = torch.zeros_like(onset_sec)
            x_duration = torch.zeros_like(duration_sec)
        elif self.surrogate_mode == "pitch_onset":
            x_onset = onset_sec
            x_duration = torch.zeros_like(duration_sec)
        elif self.surrogate_mode == "pitch_onset_duration":
            x_onset = onset_sec
            x_duration = duration_sec
        else:
            raise ValueError(f"Unsupported surrogate_mode={self.surrogate_mode}")

        x_cont = {
            "onset": x_onset,
            "duration": x_duration,
            "pitch": midi_cont["pitch"].clone().long(),
            # keep velocity neutral for now
            "velocity": torch.zeros((T,), dtype=torch.long),
        }
        return x_cont

    def _get_one(self, idx: int):
        mxl_path = self.mxl_paths[idx]
        rendered_midi_path = self._mxl_to_rendered_midi_path(mxl_path)

        # 1) continuous parse (or cache)
        cont = self._load_continuous_mxl(mxl_path)
        midi_cont = self._load_continuous_midi(rendered_midi_path)

        # 2) strict alignment sanity check
        if int(cont["pitch"].shape[0]) != int(midi_cont["pitch"].shape[0]):
            raise RuntimeError(
                f"Length mismatch between XML and rendered MIDI: "
                f"{mxl_path} -> len(xml)={int(cont['pitch'].shape[0])}, "
                f"len(midi)={int(midi_cont['pitch'].shape[0])}"
            )

        if not torch.equal(cont["pitch"].long(), midi_cont["pitch"].long()):
            raise RuntimeError(f"Pitch mismatch between XML and rendered MIDI: {mxl_path}")

        # 3) transpose on BOTH midi x and xml y so they stay consistent
        if self.augmentations.get("transpose", False):
            max_semitones = int(self.augmentations["transpose"])
            shift = random.randint(-max_semitones, max_semitones)

            midi_pitch_t, mxl_pitch_t, acc2, ks2 = ASAPDataset._transpose(
                shift,
                midi_stream=midi_cont["pitch"].clone().long(),
                mxl_stream=cont["pitch"].clone().long(),
                accidental_stream=cont["accidental"].clone(),
                keysignature_stream=cont["keysignature"].clone(),
            )

            cont = dict(cont)
            midi_cont = dict(midi_cont)

            cont["pitch"] = mxl_pitch_t
            cont["accidental"] = acc2
            cont["keysignature"] = ks2
            midi_cont["pitch"] = midi_pitch_t

        # 4) sample ONE shared crop window
        n = int(cont["pitch"].shape[0])
        start, end = self._sample_crop_indices(n)
        cont = self._slice_stream_dict(cont, start, end)
        midi_cont = self._slice_stream_dict(midi_cont, start, end)

        # 5) build surrogate x from rendered MIDI
        x_cont = self._make_surrogate_x_cont(midi_cont)

        # 6) bucket x and y separately
        x = MultistreamTokenizer.bucket_midi(x_cont)
        x = self._pad_bucketed(x)

        y = MultistreamTokenizer.bucket_mxl(cont)
        y = self._pad_bucketed(y)

        # 7) mark this batch as unpaired
        T = x["pad"].shape[0]
        x["unconditional"] = torch.ones((T, 1), dtype=torch.float32)

        if self.return_paths:
            return x, y, mxl_path, rendered_midi_path
        return x, y

    def __getitem__(self, idx: int):
        if not self.skip_on_error:
            return self._get_one(idx)

        last_err = None
        cur = idx
        for _ in range(self.max_retries):
            mxl_path = self.mxl_paths[cur]
            try:
                return self._get_one(cur)
            except Exception as e:
                last_err = e
                self._log_failure(mxl_path, e)
                # choose another sample to try
                cur = random.randint(0, len(self.mxl_paths) - 1)

        raise RuntimeError(
            f"UnpairedXMLDataset: failed to get a valid sample after {self.max_retries} retries. "
            f"Last error: {repr(last_err)}"
        )

class _ParseTimeout(Exception):
    pass

@contextmanager
def _time_limit(seconds: int):
    """
    Hard timeout using SIGALRM (Linux/Unix only).
    Works inside DataLoader worker processes (they are separate processes).
    """
    seconds = int(seconds)

    def _handler(signum, frame):
        raise _ParseTimeout(f"parse_mxl exceeded {seconds}s")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def sha256(string: str) -> str:
    h = hashlib.new("sha256")
    h.update(string.encode())
    return h.hexdigest()

if __name__ == "__main__":
    print("Initializing ASAPDataset")
    for split in ("all", "train", "validation", "test"):
        q = ASAPDataset("/mnt/ssd/hbli/datasets/PM2S_dataset/midi2scoretransformer/", split, seq_length=None, padding=None, cache=True, return_continous=False)
        print(split, len(q))

    q = ASAPDataset("/mnt/ssd/hbli/datasets/PM2S_dataset/midi2scoretransformer/", "all", seq_length=None, padding=None, cache=True, return_continous=True)
    print("Filling cache")
    # You can parallelize this loop, but you have to comment out both uses of lru_cache
    for i in tqdm(range(len(q))):
        q[i]
