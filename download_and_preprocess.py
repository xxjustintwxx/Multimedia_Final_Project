#!/usr/bin/env python3
"""
Download HaGRIDv2 512px parquet-by-parquet and preprocess with MediaPipe.
Saves only processed (crop, landmarks) pairs — no raw images kept.

Output structure:
  data/crops/<split>/<label>/<parquet_idx>_<row_idx>.jpg
  data/landmarks/<split>/<label>/<parquet_idx>_<row_idx>.npy

Labels: 0=N/A, 1=fist, 2=like, 3=ok, 4=one, 5=palm
"""

import io
import os
import sys

# Force HF cache to /work BEFORE any huggingface imports
_HF_CACHE = "/work/xxjustin77xx/tmp/hf_cache"
os.environ["HF_HOME"]           = _HF_CACHE
os.environ["HF_HUB_CACHE"]      = _HF_CACHE + "/hub"
os.environ["HF_DATASETS_CACHE"] = _HF_CACHE + "/datasets"
os.makedirs(_HF_CACHE + "/hub", exist_ok=True)
os.makedirs(_HF_CACHE + "/datasets", exist_ok=True)

import json
import time
import signal
import logging
import multiprocessing as mp
from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd
from PIL import Image
from huggingface_hub import hf_hub_download, list_repo_files

sys.path.insert(0, str(Path(__file__).parent))
from hand_preprocess import MediaPipeHandPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

DATASET_ID = "testdummyvt/hagRIDv2_512px"
OUTPUT_DIR = Path("/work/xxjustin77xx/Multimedia_Final_Project/data")

# Two-handed gestures to skip entirely
TWO_HANDED_LABELS = {5, 6, 7, 8, 23, 30, 33}  # grip, hand_heart*, holy, take_picture, timeout, xsign

# HaGRIDv2 integer label → our class (None = skip)
TARGET_MAP = {2: 1, 9: 2, 14: 3, 15: 4, 16: 5}  # fist=1, like=2, ok=3, one=4, palm=5


def our_label(dataset_label: int) -> int | None:
    if dataset_label in TWO_HANDED_LABELS:
        return None
    return TARGET_MAP.get(dataset_label, 0)  # default → N/A


# ── worker entry point ────────────────────────────────────────────────────────

def process_parquet(args: tuple) -> dict:
    """
    Downloads one parquet file, runs MediaPipe on every kept image, saves outputs.
    Returns a stats dict.
    """
    parquet_path_hf, out_split, parquet_idx = args

    stats = {"processed": 0, "failed_mediapipe": 0, "skipped_two_handed": 0}

    try:
        local_path = hf_hub_download(
            DATASET_ID,
            parquet_path_hf,
            repo_type="dataset",
        )
        df = pd.read_parquet(local_path)
    except Exception as e:
        log.warning(f"[{out_split}] parquet {parquet_idx:05d} download error: {e}")
        return stats

    preprocessor = MediaPipeHandPreprocessor()
    try:
        for row_idx, row in df.iterrows():
            ds_label = int(row["label"])
            label = our_label(ds_label)
            if label is None:
                stats["skipped_two_handed"] += 1
                continue

            img_data = row["image"]
            if isinstance(img_data, dict):
                img_bytes = img_data.get("bytes")
                if not img_bytes:
                    continue
                image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            elif isinstance(img_data, Image.Image):
                image = img_data.convert("RGB")
            else:
                continue

            result = preprocessor.preprocess_image(image)
            if result is None:
                stats["failed_mediapipe"] += 1
                continue

            crop, landmarks = result
            stem = f"{parquet_idx:05d}_{row_idx:06d}"

            crop_dir = OUTPUT_DIR / "crops" / out_split / str(label)
            lm_dir = OUTPUT_DIR / "landmarks" / out_split / str(label)
            crop_dir.mkdir(parents=True, exist_ok=True)
            lm_dir.mkdir(parents=True, exist_ok=True)

            Image.fromarray(crop).save(crop_dir / f"{stem}.jpg", quality=90)
            np.save(lm_dir / f"{stem}.npy", landmarks)
            stats["processed"] += 1

    finally:
        preprocessor.close()

    return stats


# ── main ─────────────────────────────────────────────────────────────────────

def list_parquet_files():
    """Returns {split_out_name: [(hf_path, idx), ...]}."""
    log.info("Listing parquet files in dataset repo …")
    all_files = list(list_repo_files(DATASET_ID, repo_type="dataset"))
    splits = {
        "train": ("data/train/", "train"),
        "val":   ("data/val/", "val"),
        "test":  ("data/test/", "test"),
    }
    result = {}
    for _, (prefix, out_name) in splits.items():
        files = sorted(f for f in all_files if f.startswith(prefix) and f.endswith(".parquet"))
        result[out_name] = [(f, i) for i, f in enumerate(files)]
        log.info(f"  {out_name}: {len(files)} parquet files")
    return result


def resume_index(out_split: str) -> set[int]:
    """Return set of parquet indices already fully processed (have at least one output)."""
    done = set()
    for label in range(6):
        crop_dir = OUTPUT_DIR / "crops" / out_split / str(label)
        if not crop_dir.exists():
            continue
        for f in crop_dir.iterdir():
            try:
                pidx = int(f.stem.split("_")[0])
                done.add(pidx)
            except ValueError:
                pass
    return done


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    os.makedirs("/work/xxjustin77xx/tmp/hf_home", exist_ok=True)
    os.makedirs("/work/xxjustin77xx/tmp/hf_cache", exist_ok=True)

    num_workers = max(1, min(mp.cpu_count() - 1, 11))
    log.info(f"Using {num_workers} worker processes")

    split_files = list_parquet_files()

    # Save progress file
    progress_path = OUTPUT_DIR / "progress.json"

    for out_split, file_list in split_files.items():
        already_done = resume_index(out_split)
        todo = [(hf_path, out_split, idx) for hf_path, idx in file_list if idx not in already_done]
        total = len(file_list)
        skipping = len(already_done)

        log.info(f"\n=== {out_split}: {len(todo)} parquet files to process ({skipping}/{total} already done) ===")

        if not todo:
            log.info(f"  All {total} files already done, skipping.")
            continue

        split_stats = {"processed": 0, "failed_mediapipe": 0, "skipped_two_handed": 0}
        completed = skipping

        with mp.Pool(processes=num_workers) as pool:
            for stats in pool.imap_unordered(process_parquet, todo, chunksize=1):
                for k in split_stats:
                    split_stats[k] += stats[k]
                completed += 1
                if completed % 10 == 0 or completed == total:
                    pct = 100 * completed / total
                    log.info(
                        f"  [{out_split}] {completed}/{total} ({pct:.1f}%) "
                        f"| saved={split_stats['processed']} "
                        f"| mediapipe_fail={split_stats['failed_mediapipe']}"
                    )
                    # Persist progress
                    prog = {}
                    if progress_path.exists():
                        with open(progress_path) as f:
                            prog = json.load(f)
                    prog[out_split] = {"completed_files": completed, "total_files": total, **split_stats}
                    with open(progress_path, "w") as f:
                        json.dump(prog, f, indent=2)

        log.info(
            f"[{out_split}] DONE — saved={split_stats['processed']}, "
            f"mediapipe_fail={split_stats['failed_mediapipe']}, "
            f"skipped_two_handed={split_stats['skipped_two_handed']}"
        )

    # Final count summary
    log.info("\n=== Final output counts ===")
    label_names = {0: "N/A", 1: "fist", 2: "like", 3: "ok", 4: "one", 5: "palm"}
    for split in ["train", "val", "test"]:
        counts = {}
        for label in range(6):
            d = OUTPUT_DIR / "crops" / split / str(label)
            counts[label_names[label]] = len(list(d.iterdir())) if d.exists() else 0
        log.info(f"  {split}: {counts}")


if __name__ == "__main__":
    main()
