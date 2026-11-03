
"""
AI-Driven Multi-Task CNN — complete program (revised v2)
Author: Muhammad Talha Malik

Revisions vs prior:
  - Concurrent multi-task: ignore missing labels; shared backbone + both heads
  - Exact content-hash + near-duplicate (dHash) audit and grouping
  - Near-dupe groups forced into a single split (no cross-split leakage)
  - No patient-independent claim (image-level stratified / official folders only)
  - Gradient sanity check in concurrent training
  - Stronger training defaults (honest target note 0.92 — never fabricated)
  - Triage disabled until verified normal-liver cohort exists

pip install torch albumentations opencv-python-headless pandas numpy matplotlib reportlab pydicom
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image as PDFImage
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

plt.switch_backend("Agg")

AUTHOR_NAME = "Muhammad Talha Malik"
AUTHOR_AFFILIATION = "Independent Research / Academic Study"
REPORT_TITLE = (
    "AI-Driven Multi-Task CNN for Concurrent Parenchymal Staging and "
    "Differential Focal Lesion Characterization on Grayscale Ultrasound"
)

AUDIT_ENABLED = True
NEAR_DUP_ENABLED = True
NEAR_DUP_HAMMING_MAX = 2  
ENABLE_TRIAGE = False
GRAD_CHECK_CONCURRENT = True  

KAGGLE_DATASETS = {
    "fibrosis": "vibhingupta028/liver-histopathology-fibrosis-ultrasound-images",
    "tumor": "tofael088/smc-lud-liver-ultrasound-dataset",
}
BASE_DIR = Path(__file__).resolve().parent
DATASET_CACHE_ROOT = BASE_DIR / "auto_fetched_ultrasound_data" / "kaggle"
RESEARCH_ROOT = BASE_DIR / "research_outputs"
LIVE_REPORT_ROOT = BASE_DIR / "live_reports"
INFERENCE_REPORT_ROOT = BASE_DIR / "inference_reports"
FROZEN_SPLITS_DIR = RESEARCH_ROOT / "frozen_splits"

IMAGE_SIZE = 224
DEFAULT_EPOCHS = 60
DEFAULT_PATIENCE = 12
BATCH_SIZE = 24
SEED = 42
TARGET_ACC_NOTE = 0.92  

FIBROSIS_LABELS = ["F0", "F1", "F2", "F3", "F4"]
TUMOR_LABELS = ["HCC", "Hemangioma"]
TRIAGE_LABELS = ["Fibrosis", "Tumor"]
TRIAGE_MIN_CONF = 0.55

LIVE_ANALYZE_INTERVAL_SEC = float(os.environ.get("LIVE_ANALYZE_INTERVAL_SEC", "0.75"))
LIVE_PDF_EVERY_N = int(os.environ.get("LIVE_PDF_EVERY_N", "0"))
LIVE_MAX_SECONDS = float(os.environ.get("LIVE_MAX_SECONDS", "0"))

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "").strip().lower()

LIMITATION_SPLIT = (
    "Splits are image-level (fibrosis: stratified random; tumor: official "
    "train_clean/val_clean/test_clean when present). Exact content-hash and "
    "near-duplicate (dHash) isolation reduce leakage but do NOT guarantee "
    "patient-level independence. Patient IDs are unavailable in the public "
    "sources used here; residual same-patient or serial-exam leakage remains possible."
)

class KaggleRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None and req.host != redirected.host:
            redirected.remove_header("Authorization")
        return redirected


def _kaggle_credentials():
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if username and key:
        return username, key
    path = Path.home() / ".kaggle" / "kaggle.json"
    if path.is_file():
        with path.open(encoding="utf-8") as f:
            cred = json.load(f)
        username, key = cred.get("username"), cred.get("key")
        if username and key:
            return username, key
    raise RuntimeError("Set KAGGLE_USERNAME and KAGGLE_KEY or ~/.kaggle/kaggle.json")


def _find_dataset_content(dataset_key: str, extracted_root: Path) -> Path:
    if dataset_key == "fibrosis":
        expected = {"F0", "F1", "F2", "F3", "F4"}
        for directory, child_dirs, _ in os.walk(extracted_root):
            if expected.issubset(set(child_dirs)):
                return Path(directory)
    elif dataset_key == "tumor":
        for directory, child_dirs, _ in os.walk(extracted_root):
            if {"train_clean", "val_clean", "test_clean"}.issubset(set(child_dirs)):
                if Path(directory).name == "clean_ver_for_train":
                    return Path(directory)
        for directory, child_dirs, _ in os.walk(extracted_root):
            lower = {c.lower() for c in child_dirs}
            if "hcc" in lower or "hemangioma" in lower:
                return Path(directory)
    raise RuntimeError(f"Dataset '{dataset_key}' folders not found.")


def ensure_kaggle_dataset(dataset_key: str) -> Path:
    if dataset_key not in KAGGLE_DATASETS:
        raise ValueError(dataset_key)
    cache_path = DATASET_CACHE_ROOT / dataset_key
    marker = cache_path / ".kaggle_download_complete"
    if marker.is_file():
        return _find_dataset_content(dataset_key, cache_path)
    if cache_path.exists():
        raise RuntimeError(f"Incomplete cache {cache_path}. Delete and retry.")

    username, key = _kaggle_credentials()
    DATASET_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{dataset_key}-", dir=DATASET_CACHE_ROOT))
    try:
        dataset_ref = KAGGLE_DATASETS[dataset_key]
        request = urllib.request.Request(
            f"https://www.kaggle.com/api/v1/datasets/download/{dataset_ref}"
        )
        token = base64.b64encode(f"{username}:{key}".encode()).decode("ascii")
        request.add_header("Authorization", f"Basic {token}")
        opener = urllib.request.build_opener(KaggleRedirectHandler())
        archive_path = staging / "dataset.zip"
        print(f"[KAGGLE] Downloading {dataset_ref}...")
        with opener.open(request, timeout=300) as resp, archive_path.open("wb") as out:
            shutil.copyfileobj(resp, out, length=1024 * 1024)
        extracted_root = staging / "extracted"
        extracted_root.mkdir()
        with zipfile.ZipFile(archive_path) as archive:
            root = extracted_root.resolve()
            for member in archive.infolist():
                member_path = (extracted_root / member.filename).resolve()
                try:
                    safe = os.path.commonpath((str(root), str(member_path))) == str(root)
                except ValueError:
                    safe = False
                if not safe:
                    raise RuntimeError("Unsafe path in archive.")
            archive.extractall(extracted_root)
        _find_dataset_content(dataset_key, extracted_root)
        archive_path.unlink(missing_ok=True)
        try:
            os.replace(extracted_root, cache_path)
        except PermissionError:
            shutil.copytree(extracted_root, cache_path)
        marker.write_text(dataset_ref, encoding="utf-8")
        return _find_dataset_content(dataset_key, cache_path)
    except (OSError, urllib.error.URLError, zipfile.BadZipFile) as err:
        if cache_path.is_dir() and not marker.is_file():
            shutil.rmtree(cache_path, ignore_errors=True)
        raise RuntimeError(f"Could not fetch dataset: {err}") from err
    finally:
        shutil.rmtree(staging, ignore_errors=True)

def _load_gray_resized(path: str, size: int = IMAGE_SIZE) -> np.ndarray | None:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)


def _content_hash(path: str) -> str | None:
    img = _load_gray_resized(path)
    if img is None:
        return None
    return hashlib.sha256(img.tobytes()).hexdigest()


def _dhash64(path: str, hash_size: int = 8) -> int | None:
    """Difference hash (64-bit). Robust to mild resize/compression."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    small = cv2.resize(
        img, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA
    )
    diff = small[:, 1:] > small[:, :-1]
    bits = 0
    for i, v in enumerate(diff.flatten()):
        if v:
            bits |= 1 << i
    return int(bits)


def _hamming64(a: int, b: int) -> int:
    return int((a ^ b).bit_count())


def _union_find_groups(n: int, edges: list[tuple[int, int]]) -> list[int]:
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, j in edges:
        union(i, j)
    return [find(i) for i in range(n)]



def map_fibrosis_dataset(root_path: Path) -> pd.DataFrame:
    records = []
    ext = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    for path in Path(root_path).rglob("*"):
        if path.suffix.lower() not in ext:
            continue
        parts_lower = {p.lower() for p in path.parts}
        stage = next((s for s in range(5) if f"f{s}" in parts_lower), None)
        if stage is None:
            continue
        records.append(
            {
                "image_path": str(path).replace("\\", "/"),
                "master_label": 0 if ENABLE_TRIAGE else -100,
                "fibrosis_stage": stage,
                "tumor_type": -100,
                "source": "fibrosis",
                "split_hint": "",
            }
        )
    df = pd.DataFrame(records)
    if df.empty or set(df["fibrosis_stage"].unique()) != set(range(5)):
        raise RuntimeError("Fibrosis dataset must contain F0-F4 images.")
    print("[SUCCESS] Fibrosis raw:", df["fibrosis_stage"].value_counts().sort_index().to_dict())
    return df


def map_tumor_dataset(root_path: Path, prefer_clean: bool = True) -> pd.DataFrame:
    records = []
    ext = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    for path in Path(root_path).rglob("*"):
        if path.suffix.lower() not in ext:
            continue
        lower = str(path).replace("\\", "/").lower()
        if prefer_clean and "caliper" in lower and "clean" not in lower:
            continue
        if "hemangioma" in lower:
            tumor_type = 1
        elif "hcc" in lower:
            tumor_type = 0
        else:
            continue
        split_hint = ""
        for part in path.parts:
            if part in {"train_clean", "val_clean", "test_clean", "train", "val", "test"}:
                split_hint = part
                break
        records.append(
            {
                "image_path": str(path).replace("\\", "/"),
                "master_label": 1 if ENABLE_TRIAGE else -100,
                "fibrosis_stage": -100,
                "tumor_type": tumor_type,
                "source": "tumor",
                "split_hint": split_hint,
            }
        )
    df = pd.DataFrame(records)
    if df.empty or set(df["tumor_type"].unique()) != {0, 1}:
        raise RuntimeError("Tumor dataset must contain HCC and Hemangioma.")
    print("[SUCCESS] Tumor raw:", df["tumor_type"].value_counts().sort_index().to_dict())
    return df


def audit_dataframe(df: pd.DataFrame, task: str) -> tuple[pd.DataFrame, dict]:
    if not AUDIT_ENABLED or df is None or df.empty:
        return df, {"task": task, "skipped": True, "note": "AUDIT_ENABLED=False"}

    print(f"[AUDIT] {task}: exact-hash + near-dup over {len(df)} images...")
    rows = []
    bad = 0
    seen_exact: set[str] = set()
    exact_dup_removed = 0
    folder_tokens: dict[str, int] = {}
    cross_label_near = 0


    for _, r in df.iterrows():
        path = r["image_path"]
        h = _content_hash(path)
        dh = _dhash64(path) if NEAR_DUP_ENABLED else None
        if h is None:
            bad += 1
            continue
        if h in seen_exact:
            exact_dup_removed += 1
            continue
        seen_exact.add(h)
        d = r.to_dict()
        d["content_hash"] = h
        d["dhash"] = dh if dh is not None else -1
        d["near_group_id"] = -1
        rows.append(d)
        parts = Path(path).parts
        for p in parts[-4:]:
            pl = p.lower()
            if pl in {"f0", "f1", "f2", "f3", "f4", "hcc", "hemangioma"} or "clean" in pl:
                folder_tokens[p] = folder_tokens.get(p, 0) + 1


    near_edges = []
    near_pairs_sample = []
    if NEAR_DUP_ENABLED and rows:
        dhashes = [int(r["dhash"]) for r in rows]
        n = len(rows)

        for i in range(n):
            if dhashes[i] < 0:
                continue
            for j in range(i + 1, n):
                if dhashes[j] < 0:
                    continue
                dist = _hamming64(dhashes[i], dhashes[j])
                if dist <= NEAR_DUP_HAMMING_MAX:
                    near_edges.append((i, j))
                    if len(near_pairs_sample) < 20:
                        near_pairs_sample.append(
                            {
                                "i": rows[i]["image_path"],
                                "j": rows[j]["image_path"],
                                "hamming": dist,
                            }
                        )

                    if task == "fibrosis":
                        if rows[i]["fibrosis_stage"] != rows[j]["fibrosis_stage"]:
                            cross_label_near += 1
                    else:
                        if rows[i]["tumor_type"] != rows[j]["tumor_type"]:
                            cross_label_near += 1

        roots = _union_find_groups(n, near_edges)

        root_to_gid: dict[int, int] = {}
        gid = 0
        for i, rt in enumerate(roots):
            if rt not in root_to_gid:
                root_to_gid[rt] = gid
                gid += 1
            rows[i]["near_group_id"] = root_to_gid[rt]
        n_groups = gid
        multi = sum(
            1
            for g in range(n_groups)
            if sum(1 for r in rows if r["near_group_id"] == g) > 1
        )
    else:
        for i, r in enumerate(rows):
            r["near_group_id"] = i
        n_groups = len(rows)
        multi = 0

    clean = pd.DataFrame(rows)
    audit = {
        "task": task,
        "author": AUTHOR_NAME,
        "kaggle_ref": KAGGLE_DATASETS.get(task, ""),
        "raw_indexed": int(len(df)),
        "unreadable": int(bad),
        "exact_duplicates_removed": int(exact_dup_removed),
        "unique_images_after_exact": int(len(clean)),
        "near_dup_enabled": NEAR_DUP_ENABLED,
        "near_dup_hamming_max": NEAR_DUP_HAMMING_MAX,
        "near_dup_edges": int(len(near_edges)) if NEAR_DUP_ENABLED else 0,
        "near_groups_total": int(n_groups),
        "near_groups_with_multiple": int(multi),
        "cross_label_near_pair_count": int(cross_label_near),
        "near_pair_samples": near_pairs_sample[:10],
        "class_counts": {},
        "split_hint_counts": {},
        "folder_label_tokens": folder_tokens,
        "folder_mapping": [],
        "sample_hashes": [r["content_hash"] for r in rows[:5]],
        "available_metadata_columns": list(clean.columns),
        "source_label": task,
        "enable_triage": ENABLE_TRIAGE,
        "limitation": LIMITATION_SPLIT,
    }
    if task == "fibrosis":
        vc = clean["fibrosis_stage"].value_counts().sort_index()
        audit["class_counts"] = {FIBROSIS_LABELS[i]: int(vc.get(i, 0)) for i in range(5)}
        audit["folder_mapping"] = ["Class from path/folder token F0..F4"]
        if set(clean["fibrosis_stage"].unique()) != set(range(5)):
            raise RuntimeError("Audit: missing F0-F4 after dedup")
    else:
        if "split_hint" in clean.columns:
            audit["split_hint_counts"] = {
                str(k): int(v) for k, v in clean["split_hint"].value_counts().items()
            }
            audit["official_clean_splits_verified"] = {
                "train_clean", "val_clean", "test_clean"
            }.issubset(set(clean["split_hint"].astype(str)))
        vc = clean["tumor_type"].value_counts().sort_index()
        audit["class_counts"] = {TUMOR_LABELS[i]: int(vc.get(i, 0)) for i in range(2)}
        audit["folder_mapping"] = [
            "Class from path token hcc|hemangioma",
            "Prefer train_clean/val_clean/test_clean when present",
        ]
        if set(clean["tumor_type"].unique()) != {0, 1}:
            raise RuntimeError("Audit: need HCC and Hemangioma after dedup")

    print(json.dumps({k: v for k, v in audit.items() if k != "near_pair_samples"}, indent=2))
    RESEARCH_ROOT.mkdir(parents=True, exist_ok=True)
    (RESEARCH_ROOT / f"audit_{task}.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    lines = [
        f"Dataset audit — {task}",
        f"Author: {AUTHOR_NAME}",
        f"Kaggle: {audit['kaggle_ref']}",
        f"Raw indexed: {audit['raw_indexed']}",
        f"Unreadable: {audit['unreadable']}",
        f"Exact duplicates removed: {audit['exact_duplicates_removed']}",
        f"Unique after exact: {audit['unique_images_after_exact']}",
        f"Near-dup groups (multi-member): {audit['near_groups_with_multiple']}",
        f"Cross-label near pairs: {audit['cross_label_near_pair_count']}",
        f"Class counts: {json.dumps(audit['class_counts'])}",
        f"Split hints: {json.dumps(audit.get('split_hint_counts', {}))}",
        f"Limitation: {LIMITATION_SPLIT}",
        f"ENABLE_TRIAGE={ENABLE_TRIAGE}",
    ]
    (RESEARCH_ROOT / f"audit_{task}_report.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(f"[AUDIT] Wrote {RESEARCH_ROOT / f'audit_{task}_report.txt'}")
    return clean.reset_index(drop=True), audit


def load_task_metadata(task: str) -> pd.DataFrame:
    root = ensure_kaggle_dataset(task)
    if task == "fibrosis":
        df = map_fibrosis_dataset(root)
    elif task == "tumor":
        df = map_tumor_dataset(root)
    else:
        raise ValueError(task)
    df, _ = audit_dataframe(df, task)
    return df



def _stratified_group_split(df: pd.DataFrame, label_col: str, seed: int = SEED):
    """
    Stratified split at near-group level when near_group_id exists.
    Entire near-duplicate group goes to train OR val OR test only.
    """
    work = df.copy()
    if "near_group_id" not in work.columns:
        work["near_group_id"] = np.arange(len(work))


    group_rows = []
    for gid, g in work.groupby("near_group_id"):
        labels = g[label_col].value_counts()
        maj = int(labels.idxmax())
        group_rows.append({"near_group_id": gid, "label": maj, "n": len(g)})
    gdf = pd.DataFrame(group_rows)

    train_g, val_g, test_g = [], [], []
    rs = seed
    for _, grp in gdf.groupby("label"):
        shuffled = grp.sample(frac=1, random_state=rs)
        rs += 1
        if len(shuffled) < 3:
            raise ValueError(
                f"Need >=3 near-groups per class in {label_col} for group split"
            )
        n_test = max(1, round(len(shuffled) * 0.15))
        n_val = max(1, round(len(shuffled) * 0.15))
        test_g.extend(shuffled.iloc[:n_test]["near_group_id"].tolist())
        val_g.extend(shuffled.iloc[n_test : n_test + n_val]["near_group_id"].tolist())
        train_g.extend(shuffled.iloc[n_test + n_val :]["near_group_id"].tolist())

    def take(gids):
        return work[work["near_group_id"].isin(gids)].copy()

    return {
        "train": take(train_g),
        "validation": take(val_g),
        "test": take(test_g),
    }


def _assert_no_cross_split_leak(splits: dict, task: str):
    hash_to_split: dict[str, str] = {}
    dhash_to_split: dict[int, str] = {}
    group_to_split: dict[int, str] = {}

    for name, frame in splits.items():
        if frame is None or frame.empty:
            continue
        for _, row in frame.iterrows():
            path = str(row["image_path"])
            h = row["content_hash"] if "content_hash" in frame.columns else _content_hash(path)
            if h is None:
                continue
            h = str(h)
            prev = hash_to_split.get(h)
            if prev is not None and prev != name:
                raise RuntimeError(
                    f"[{task}] Exact duplicate hash crosses splits ({prev} vs {name}). "
                    f"Delete {FROZEN_SPLITS_DIR} and rebuild."
                )
            hash_to_split[h] = name

            if NEAR_DUP_ENABLED and "dhash" in frame.columns:
                dh = int(row["dhash"])
                if dh >= 0:
                    
                    prev_d = dhash_to_split.get(dh)
                    if prev_d is not None and prev_d != name:
                        raise RuntimeError(
                            f"[{task}] Identical dHash crosses splits ({prev_d} vs {name})."
                        )
                    dhash_to_split[dh] = name

            if "near_group_id" in frame.columns:
                gid = int(row["near_group_id"])
                prev_g = group_to_split.get(gid)
                if prev_g is not None and prev_g != name:
                    raise RuntimeError(
                        f"[{task}] near_group_id={gid} crosses splits ({prev_g} vs {name})."
                    )
                group_to_split[gid] = name

    print(
        f"[FROZEN] {task}: cross-split exact/near-group check OK "
        f"({len(hash_to_split)} hashes)."
    )


def _save_frozen_task_splits(task: str, splits: dict):
    FROZEN_SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        cols = [
            c
            for c in (
                "image_path",
                "master_label",
                "fibrosis_stage",
                "tumor_type",
                "source",
                "split_hint",
                "content_hash",
                "dhash",
                "near_group_id",
            )
            if c in frame.columns
        ]
        frame[cols].to_csv(FROZEN_SPLITS_DIR / f"{task}_{name}.csv", index=False)
    meta = {
        "task": task,
        "author": AUTHOR_NAME,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "sizes": {k: len(v) for k, v in splits.items()},
        "enable_triage": ENABLE_TRIAGE,
        "near_dup_enabled": NEAR_DUP_ENABLED,
        "note": LIMITATION_SPLIT,
    }
    (FROZEN_SPLITS_DIR / f"{task}_split_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"[FROZEN] Saved {task} splits -> {FROZEN_SPLITS_DIR}")


def _load_frozen_task_splits(task: str) -> dict | None:
    paths = {
        name: FROZEN_SPLITS_DIR / f"{task}_{name}.csv"
        for name in ("train", "validation", "test")
    }
    if not all(p.is_file() for p in paths.values()):
        return None
    out = {}
    for name, p in paths.items():
        df = pd.read_csv(p)
        for col in ("master_label", "fibrosis_stage", "tumor_type", "dhash", "near_group_id"):
            if col in df.columns:
                df[col] = df[col].astype(int)
        if "source" not in df.columns:
            df["source"] = task
        if "split_hint" not in df.columns:
            df["split_hint"] = ""
        if not ENABLE_TRIAGE:
            df["master_label"] = -100
        elif task == "fibrosis":
            df["master_label"] = 0
        elif task == "tumor":
            df["master_label"] = 1
        out[name] = df
    print(f"[FROZEN] Loaded existing {task} splits.")
    return out


def prepare_splits(df: pd.DataFrame, mode: str):
    if mode == "fibrosis":
        frozen = _load_frozen_task_splits("fibrosis")
        if frozen is not None:
            _assert_no_cross_split_leak(frozen, "fibrosis")
            return frozen
        sub = df[df["fibrosis_stage"] != -100].copy()
        splits = _stratified_group_split(sub, "fibrosis_stage")
        _assert_no_cross_split_leak(splits, "fibrosis")
        _save_frozen_task_splits("fibrosis", splits)
        return splits

    if mode == "tumor":
        frozen = _load_frozen_task_splits("tumor")
        if frozen is not None:
            _assert_no_cross_split_leak(frozen, "tumor")
            return frozen
        sub = df[df["tumor_type"] != -100].copy()
        hints = set(sub["split_hint"].astype(str).unique())
        if {"train_clean", "val_clean", "test_clean"}.issubset(hints):
            splits = {
                "train": sub[sub["split_hint"] == "train_clean"].copy(),
                "validation": sub[sub["split_hint"] == "val_clean"].copy(),
                "test": sub[sub["split_hint"] == "test_clean"].copy(),
            }
            method = "official_clean_folders"

            try:
                _assert_no_cross_split_leak(splits, "tumor")
            except RuntimeError as e:
                print(f"[WARN] Official tumor splits failed leak check: {e}")
                print("[WARN] Falling back to near-group stratified split.")
                splits = _stratified_group_split(sub, "tumor_type")
                method = "stratified_near_group_fallback"
        else:
            splits = _stratified_group_split(sub, "tumor_type")
            method = "stratified_near_group"
        _assert_no_cross_split_leak(splits, "tumor")
        _save_frozen_task_splits("tumor", splits)
        FROZEN_SPLITS_DIR.mkdir(parents=True, exist_ok=True)
        (FROZEN_SPLITS_DIR / "tumor_split_method.txt").write_text(method, encoding="utf-8")
        return splits

    if mode == "concurrent":
        fib_part = df[df["source"] == "fibrosis"]
        tum_part = df[df["source"] == "tumor"]
        if fib_part.empty:
            fib_part = load_task_metadata("fibrosis")
        if tum_part.empty:
            tum_part = load_task_metadata("tumor")
        f_splits = prepare_splits(fib_part, "fibrosis")
        t_splits = prepare_splits(tum_part, "tumor")
        return {
            name: pd.concat([f_splits[name], t_splits[name]], ignore_index=True)
            for name in ("train", "validation", "test")
        }
    raise ValueError(mode)


def inspect_frozen_splits() -> dict:
    report = {
        "author": AUTHOR_NAME,
        "checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "ok": True,
        "limitation": LIMITATION_SPLIT,
        "details": {},
    }
    for task in ("fibrosis", "tumor"):
        frozen = _load_frozen_task_splits(task)
        if frozen is None:
            report["ok"] = False
            report["details"][task] = "MISSING frozen CSVs"
            continue
        try:
            _assert_no_cross_split_leak(frozen, task)
        except RuntimeError as e:
            report["ok"] = False
            report["details"][task] = str(e)
            continue
        ids = {}
        for split_name, frame in frozen.items():
            if "content_hash" in frame.columns and frame["content_hash"].notna().any():
                keyset = set(frame["content_hash"].astype(str))
            else:
                keyset = set(frame["image_path"].astype(str))
            ids[split_name] = keyset
            report["details"][f"{task}_{split_name}_n"] = len(frame)
        inter_tv = ids["train"] & ids["validation"]
        inter_tt = ids["train"] & ids["test"]
        inter_vt = ids["validation"] & ids["test"]
        if inter_tv or inter_tt or inter_vt:
            report["ok"] = False
            report["details"][f"{task}_id_overlap"] = {
                "train_val": len(inter_tv),
                "train_test": len(inter_tt),
                "val_test": len(inter_vt),
            }
        else:
            report["details"][f"{task}_id_overlap"] = "none"

    f = _load_frozen_task_splits("fibrosis")
    t = _load_frozen_task_splits("tumor")
    if f is not None and t is not None:
        for split_name in ("train", "validation", "test"):
            conc_paths = set(
                pd.concat([f[split_name], t[split_name]], ignore_index=True)[
                    "image_path"
                ].astype(str)
            )
            sep_paths = set(f[split_name]["image_path"].astype(str)) | set(
                t[split_name]["image_path"].astype(str)
            )
            match = conc_paths == sep_paths
            report["details"][f"concurrent_{split_name}_matches_union"] = match
            if not match:
                report["ok"] = False

    RESEARCH_ROOT.mkdir(parents=True, exist_ok=True)
    out = RESEARCH_ROOT / "frozen_splits_inspection.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"[FROZEN INSPECT] {out} | ok={report['ok']}")
    return report



class GrayscaleUltrasoundDataset(Dataset):
    def __init__(self, metadata_df: pd.DataFrame, augment: bool = False):
        self.metadata = metadata_df.reset_index(drop=True)
        transforms = [A.Resize(IMAGE_SIZE, IMAGE_SIZE)]
        if augment:
            transforms.extend(
                [
                    A.HorizontalFlip(p=0.5),
                    A.Affine(
                        translate_percent=(-0.05, 0.05),
                        scale=(0.90, 1.10),
                        rotate=(-12, 12),
                        border_mode=cv2.BORDER_REFLECT_101,
                        p=0.6,
                    ),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.15, contrast_limit=0.15, p=0.6
                    ),
                    A.GaussNoise(std_range=(0.02, 0.08), p=0.25),
                    A.CLAHE(clip_limit=2.0, p=0.3),
                ]
            )
        transforms.extend([A.Normalize(mean=(0.456,), std=(0.224,)), ToTensorV2()])
        self.transforms = A.Compose(transforms)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        image = cv2.imread(row["image_path"], cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Cannot read: {row['image_path']}")
        image_3d = np.stack([image, image, image], axis=-1)
        out = self.transforms(image=image_3d)["image"][0:1, :, :]
        return (
            out,
            torch.tensor(int(row["master_label"]), dtype=torch.long),
            torch.tensor(int(row["fibrosis_stage"]), dtype=torch.long),
            torch.tensor(int(row["tumor_type"]), dtype=torch.long),
        )

class ResidualConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.convolutions = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
        )
        self.skip = (
            nn.Identity()
            if stride == 1 and in_ch == out_ch
            else nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.GroupNorm(8, out_ch),
            )
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.convolutions(x) + self.skip(x))


class HierarchicalMultiHeadCNN(nn.Module):
    """Shared residual backbone; triage present but unused when ENABLE_TRIAGE=False."""

    def __init__(self):
        super().__init__()
        self.shared_cnn_backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
            ResidualConvBlock(32, 48, stride=1),
            ResidualConvBlock(48, 64, stride=2),
            ResidualConvBlock(64, 80, stride=1),
            ResidualConvBlock(80, 96, stride=2),
            ResidualConvBlock(96, 112, stride=1),
            ResidualConvBlock(112, 128, stride=2),
            ResidualConvBlock(128, 160, stride=1),
            ResidualConvBlock(160, 192, stride=2),
        )
        self.average_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.feature_projection = nn.Sequential(
            nn.Linear(384, 320),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.35),
            nn.Linear(320, 192),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.25),
        )
        self.master_triage_head = nn.Linear(192, 2)
        self.diffuse_staging_head = nn.Linear(192, 5)
        self.differential_tumor_head = nn.Linear(192, 2)

    def forward(self, x):
        fmap = self.shared_cnn_backbone(x)
        feats = torch.cat(
            (self.average_pool(fmap).flatten(1), self.max_pool(fmap).flatten(1)), dim=1
        )
        feats = self.feature_projection(feats)
        return (
            self.master_triage_head(feats),
            self.diffuse_staging_head(feats),
            self.differential_tumor_head(feats),
        )

class ConcurrentMultiTaskLoss(nn.Module):
    def __init__(
        self,
        fibrosis_weights=None,
        tumor_weights=None,
        gamma: float = 2.0,
        use_triage: bool = False,
    ):
        super().__init__()
        if fibrosis_weights is not None:
            self.register_buffer("fibrosis_weights", fibrosis_weights.float())
        else:
            self.fibrosis_weights = None
        if tumor_weights is not None:
            self.register_buffer("tumor_weights", tumor_weights.float())
        else:
            self.tumor_weights = None
        self.gamma = gamma
        self.use_triage = use_triage

    def _focal_ce(self, logits, targets, weight=None):
        ce = nn.functional.cross_entropy(
            logits, targets, weight=weight, reduction="none", ignore_index=-100
        )
        valid = targets != -100
        if not valid.any():
            return logits.new_tensor(0.0)
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce)[valid].mean()

    def forward(self, triage_logits, stage_logits, tumor_logits, master, stage, tumor):
        parts = []
        if self.use_triage and (master != -100).any():
            parts.append(
                nn.functional.cross_entropy(triage_logits, master, ignore_index=-100)
            )
        if (stage != -100).any():
            parts.append(self._focal_ce(stage_logits, stage, self.fibrosis_weights))
        if (tumor != -100).any():
            parts.append(
                self._focal_ce(tumor_logits, tumor, self.tumor_weights)
            )
        if not parts:
            return triage_logits.new_tensor(0.0)
        return sum(parts) / len(parts)


def _calculate_roc_curves(targets, probabilities, class_names):
    curves = []
    t_arr, p_arr = np.asarray(targets), np.asarray(probabilities)
    for i, name in enumerate(class_names):
        binary = t_arr == i
        pos, neg = int(binary.sum()), int((~binary).sum())
        if not pos or not neg:
            curves.append({"class": name, "auc": None, "fpr": [], "tpr": []})
            continue
        order = np.argsort(-p_arr[:, i])
        scores, sorted_t = p_arr[order, i], binary[order]
        tp, fp = np.cumsum(sorted_t), np.cumsum(~sorted_t)
        ends = np.r_[np.flatnonzero(np.diff(scores)), len(scores) - 1]
        tpr = np.concatenate(([0.0], tp[ends] / pos))
        fpr = np.concatenate(([0.0], fp[ends] / neg))
        auc = float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2))
        curves.append(
            {"class": name, "auc": auc, "fpr": fpr.tolist(), "tpr": tpr.tolist()}
        )
    return curves


def calculate_metrics(targets, predictions, class_names, probabilities):
    n = len(class_names)
    matrix = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(targets, predictions):
        matrix[int(t), int(p)] += 1
    roc_curves = _calculate_roc_curves(targets, probabilities, class_names)
    per_class = []
    for i, name in enumerate(class_names):
        tp = int(matrix[i, i])
        pred_sum, true_sum = int(matrix[:, i].sum()), int(matrix[i, :].sum())
        precision = tp / pred_sum if pred_sum else 0.0
        recall = tp / true_sum if true_sum else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        per_class.append(
            {
                "class": name,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "support": true_sum,
                "roc_auc": roc_curves[i]["auc"],
            }
        )
    aucs = [c["auc"] for c in roc_curves if c["auc"] is not None]
    return {
        "accuracy": float(np.trace(matrix) / matrix.sum()) if matrix.sum() else 0.0,
        "macro_precision": float(np.mean([x["precision"] for x in per_class])),
        "macro_recall": float(np.mean([x["recall"] for x in per_class])),
        "macro_f1": float(np.mean([x["f1_score"] for x in per_class])),
        "macro_roc_auc": float(np.mean(aucs)) if aucs else None,
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
        "roc_curves": roc_curves,
        "meets_target_acc_0_92": bool(
            matrix.sum() and (np.trace(matrix) / matrix.sum()) >= TARGET_ACC_NOTE
        ),
    }



def _plot_training_history(history, output_dir: Path):
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_loss"], marker="o", label="Train loss")
    ax.plot(epochs, history["validation_loss"], marker="o", label="Val loss")
    ax.set(title="Training history", xlabel="Epoch", ylabel="Loss")
    ax.grid(alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if history.get("validation_score"):
        ax2 = ax.twinx()
        ax2.plot(
            epochs,
            history["validation_score"],
            marker="s",
            color="tab:green",
            label="Val mean macro-F1",
        )
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("Val score")
        h2, l2 = ax2.get_legend_handles_labels()
        handles.extend(h2)
        labels.extend(l2)
    ax.legend(handles, labels, loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "training_validation_loss.png", dpi=180)
    plt.close(fig)


def _plot_training_extended(history: dict, output_dir: Path):
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(epochs, history["train_loss"], "o-", label="Train loss")
    axes[0].plot(epochs, history["validation_loss"], "o-", label="Val loss")
    axes[0].set_title("CNN loss curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].plot(
        epochs, history["validation_score"], "s-", color="tab:green", label="Val mean macro-F1"
    )
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("CNN validation score (prespecified)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "cnn_training_extended.png", dpi=180)
    plt.close(fig)

    gap = [
        history["validation_loss"][i] - history["train_loss"][i]
        for i in range(len(epochs))
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(epochs, gap, color="#6A1B9A", alpha=0.85)
    ax.axhline(0, color="gray", lw=1)
    ax.set_title("Val−Train loss gap (overfit signal)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gap")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "cnn_overfit_gap.png", dpi=180)
    plt.close(fig)


def _plot_confusion_matrix(matrix, class_names, output_dir: Path, prefix: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted",
        ylabel="True",
        title=f"{prefix} confusion matrix",
    )
    thr = matrix.max() / 2 if matrix.size else 0
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            ax.text(
                c,
                r,
                str(matrix[r, c]),
                ha="center",
                va="center",
                color="white" if matrix[r, c] > thr else "black",
            )
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_confusion_matrix.png", dpi=180)
    plt.close(fig)


def _plot_class_metrics(metrics, output_dir: Path, prefix: str):
    items = metrics["per_class"]
    names = [x["class"] for x in items]
    x = np.arange(len(names))
    w = 0.24
    fig, ax = plt.subplots(figsize=(8, 5))
    for off, key, lab in (
        (-w, "precision", "Precision"),
        (0, "recall", "Recall"),
        (w, "f1_score", "F1"),
    ):
        ax.bar(x + off, [it[key] for it in items], w, label=lab)
    ax.set(xticks=x, xticklabels=names, ylim=(0, 1), title=f"{prefix} metrics")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_class_metrics.png", dpi=180)
    plt.close(fig)


def _plot_metric_hbars(metrics: dict, output_dir: Path, prefix: str):
    items = metrics["per_class"]
    names = [x["class"] for x in items]
    y = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(12, max(3.5, 0.45 * len(names) + 1)))
    for ax, key, title, color in zip(
        axes,
        ("precision", "recall", "f1_score"),
        ("Precision", "Recall", "F1-score"),
        ("#174A70", "#2E7D4F", "#8B4513"),
    ):
        vals = [float(it[key]) for it in items]
        ax.barh(y, vals, color=color, height=0.65)
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel(title)
        ax.set_title(f"{prefix} — {title}")
        ax.grid(axis="x", alpha=0.3)
        for yi, v in zip(y, vals):
            ax.text(min(v + 0.02, 0.98), yi, f"{v:.2f}", va="center", fontsize=8)
    fig.suptitle(f"{prefix} per-class readings (horizontal bars)", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_hbar_class_readings.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, max(3, 0.4 * len(names) + 1)))
    supports = [int(it["support"]) for it in items]
    ax.barh(y, supports, color="#5C6BC0", height=0.65)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Test image count (support)")
    ax.set_title(f"{prefix} — class support")
    ax.grid(axis="x", alpha=0.3)
    mx = max(supports) if supports else 1
    for yi, v in zip(y, supports):
        ax.text(v + mx * 0.01, yi, str(v), va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_hbar_support.png", dpi=180)
    plt.close(fig)


def _plot_macro_summary_hbar(task_metrics: dict, output_dir: Path):
    labels, accs, f1s, aucs = [], [], [], []
    for task, m in task_metrics.items():
        labels.append(task)
        accs.append(m["accuracy"])
        f1s.append(m["macro_f1"])
        aucs.append(m["macro_roc_auc"] if m["macro_roc_auc"] is not None else 0.0)
    y = np.arange(len(labels))
    h = 0.22
    fig, ax = plt.subplots(figsize=(8, 3 + 0.5 * len(labels)))
    ax.barh(y - h, accs, height=h, label="Accuracy", color="#174A70")
    ax.barh(y, f1s, height=h, label="Macro-F1", color="#2E7D4F")
    ax.barh(y + h, aucs, height=h, label="Macro ROC-AUC", color="#C62828")
    ax.axvline(TARGET_ACC_NOTE, color="orange", ls="--", label=f"Target note {TARGET_ACC_NOTE}")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Score")
    ax.set_title("Macro readings summary (horizontal)")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "macro_summary_hbar.png", dpi=180)
    plt.close(fig)


def _plot_roc_curves(roc_curves, output_dir: Path, prefix: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    for curve in roc_curves:
        if curve["auc"] is not None:
            ax.plot(
                curve["fpr"],
                curve["tpr"],
                label=f"{curve['class']} AUC={curve['auc']:.3f}",
            )
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="FPR", ylabel="TPR", title=f"{prefix} ROC")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_roc_curves.png", dpi=180)
    plt.close(fig)


def _plot_live_confidence_hbar(bars: dict, output_dir: Path) -> Path:
    labels = list(bars.keys())
    values = list(bars.values())
    colors_list = []
    for lb in labels:
        if lb.startswith("Triage"):
            colors_list.append("#174A70")
        elif lb.startswith("Fib"):
            colors_list.append("#2E7D4F")
        else:
            colors_list.append("#C62828")
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(labels) + 1)))
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors_list, height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Softmax confidence")
    ax.set_title("Model readings — horizontal bar chart")
    ax.grid(axis="x", alpha=0.3)
    for yi, v in zip(y, values):
        ax.text(min(v + 0.01, 0.92), yi, f"{v:.1%}", va="center", fontsize=7)
    fig.tight_layout()
    path = output_dir / "live_confidence_hbar.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_gradcam(model, dataset, task, class_names, output_dir, device, prefix):
    target_layer = model.shared_cnn_backbone[-1]
    captured = {}

    def hook_fn(*args):
        out = args[-1]
        captured["activation"] = out
        out.register_hook(lambda g: captured.__setitem__("gradient", g))

    handle = target_layer.register_forward_hook(hook_fn)
    saved = set()
    try:
        model.eval()
        for i in range(len(dataset)):
            img, _, stage, tumor = dataset[i]
            target = int(stage.item()) if task == "fibrosis" else int(tumor.item())
            if target < 0 or target in saved or target >= len(class_names):
                continue
            model.zero_grad(set_to_none=True)
            captured.clear()
            outputs = model(img.unsqueeze(0).to(device))
            logits = outputs[1] if task == "fibrosis" else outputs[2]
            pred = int(logits.argmax(1).item())
            logits[0, pred].backward()
            act = captured["activation"].detach()[0]
            grad = captured["gradient"].detach()[0]
            weights = grad.mean(dim=(1, 2), keepdim=True)
            heat = torch.relu((weights * act).sum(dim=0))
            if heat.max() > 0:
                heat = heat / heat.max()
            heat_np = cv2.resize(heat.cpu().numpy(), (img.shape[-1], img.shape[-2]))
            gray = (img[0].cpu().numpy() * 0.224 + 0.456).clip(0, 1)
            fig, axes = plt.subplots(1, 2, figsize=(9, 4))
            axes[0].imshow(gray, cmap="gray")
            axes[0].set_title(f"Input: {class_names[target]}")
            axes[1].imshow(gray, cmap="gray")
            axes[1].imshow(heat_np, cmap="jet", alpha=0.45, vmin=0, vmax=1)
            axes[1].set_title(f"Grad-CAM: {class_names[pred]}")
            for a in axes:
                a.axis("off")
            fig.tight_layout()
            fig.savefig(output_dir / f"gradcam_{prefix}_{class_names[target]}.png", dpi=180)
            plt.close(fig)
            saved.add(target)
            if len(saved) == len(class_names):
                break
    finally:
        handle.remove()


def _pdf_image(path: Path, max_w, max_h):
    if not path.is_file():
        raise FileNotFoundError(path)
    image = PDFImage(str(path))
    scale = min(max_w / image.imageWidth, max_h / image.imageHeight)
    image.drawWidth = image.imageWidth * scale
    image.drawHeight = image.imageHeight * scale
    return image

def generate_local_interpretation(report: dict) -> list:
    lines = [
        f"Report author: {AUTHOR_NAME}. Metrics from held-out tests "
        f"(frozen IDs under {FROZEN_SPLITS_DIR.name} when present).",
        LIMITATION_SPLIT,
        f"Mode: {report.get('mode', '')}. Shared residual grayscale CNN. "
        f"ENABLE_TRIAGE={ENABLE_TRIAGE}. Heads: fibrosis F0-F4 and HCC vs Hemangioma. "
        f"Epoch budget up to {DEFAULT_EPOCHS}, patience {DEFAULT_PATIENCE}. "
        "Validation selection uses mean of task macro-F1 only.",
        "No Normal class was trained: public datasets do not provide a verified "
        "normal-liver cohort in this pipeline.",
        f"Research accuracy target note: {TARGET_ACC_NOTE:.0%} (not forced; "
        "actual held-out metrics reported honestly).",
    ]
    for task_key, metrics in report.get("task_metrics", {}).items():
        names = report["task_class_names"][task_key]
        n_cls = len(names)
        f1, acc = metrics["macro_f1"], metrics["accuracy"]
        level = (
            "limited"
            if f1 < 0.4
            else ("moderate" if f1 < 0.7 else "comparatively strong on this split")
        )
        flag = "meets" if metrics.get("meets_target_acc_0_92") else "below"
        lines.append(
            f"{task_key.upper()}: accuracy {acc:.1%} ({flag} {TARGET_ACC_NOTE:.0%} target note), "
            f"macro-F1 {f1:.3f} ({level}). Chance ~{1 / n_cls:.1%}."
        )
        if metrics["macro_roc_auc"] is not None:
            lines.append(
                f"{task_key.upper()}: macro ROC-AUC {metrics['macro_roc_auc']:.3f}."
            )
        weak = [
            f"{it['class']}(F1={it['f1_score']:.2f},n={it['support']})"
            for it in metrics["per_class"]
            if it["support"] and it["f1_score"] < 0.45
        ]
        if weak:
            lines.append(f"{task_key.upper()} weaker classes: " + "; ".join(weak) + ".")
    lines.append(
        "Grad-CAM is qualitative saliency only and does not prove localization "
        "or clinical accuracy."
    )
    lines.append(
        "No external multi-center or target-scanner validation was performed. "
        "Research prototype — not clinical diagnosis."
    )
    return lines


def maybe_llm_enrich_interpretation(report: dict, local_lines: list) -> list:
    if LLM_PROVIDER not in {"openai", "xai"}:
        return local_lines
    metrics_blob = json.dumps(
        {
            "mode": report.get("mode"),
            "task_metrics": {
                k: {
                    kk: vv
                    for kk, vv in m.items()
                    if kk not in {"roc_curves", "confusion_matrix"}
                }
                for k, m in report.get("task_metrics", {}).items()
            },
            "best_epoch": report.get("best_epoch"),
        },
        indent=2,
    )[:12000]
    prompt = (
        f"Author {AUTHOR_NAME}. Write 4-6 scientific paragraphs. No clinical diagnosis. "
        f"Do not invent numbers.\nJSON:\n{metrics_blob}\nLOCAL:\n" + "\n".join(local_lines)
    )
    try:
        if LLM_PROVIDER == "openai":
            key = os.environ.get("OPENAI_API_KEY", "")
            if not key:
                return local_lines
            url = "https://api.openai.com/v1/chat/completions"
            body = {
                "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                "messages": [
                    {"role": "system", "content": "Scientific writing assistant."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
            }
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        else:
            key = os.environ.get("XAI_API_KEY", "")
            if not key:
                return local_lines
            url = "https://api.x.ai/v1/chat/completions"
            body = {
                "model": os.environ.get("XAI_MODEL", "grok-2-latest"),
                "messages": [
                    {"role": "system", "content": "Scientific writing assistant."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
            }
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        req = urllib.request.Request(
            url, data=json.dumps(body).encode(), headers=headers, method="POST"
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
        text = data["choices"][0]["message"]["content"].strip()
        extra = [p.strip() for p in text.split("\n") if p.strip()]
        return local_lines + ["--- AI platform synthesis ---"] + extra
    except Exception as err:
        print(f"[WARN] LLM skipped: {err}")
        return local_lines


def generate_research_pdf(report: dict, output_dir: Path) -> Path:
    output_path = output_dir / "research_report.pdf"
    page_w, _ = A4
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=1.8 * cm,
        leftMargin=1.8 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.7 * cm,
        title=REPORT_TITLE,
        author=AUTHOR_NAME,
    )
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            alignment=TA_CENTER,
            fontSize=14,
            leading=18,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionHeading",
            parent=styles["Heading2"],
            textColor=colors.HexColor("#174A70"),
        )
    )

    story = [
        Paragraph(escape(REPORT_TITLE), styles["ReportTitle"]),
        Paragraph(
            f"<b>Author:</b> {escape(AUTHOR_NAME)}<br/>{escape(AUTHOR_AFFILIATION)}<br/>"
            f"Mode: <b>{escape(report['mode'])}</b> | ENABLE_TRIAGE={ENABLE_TRIAGE} | "
            f"{escape(report['generated_at'])}",
            styles["Normal"],
        ),
        Spacer(1, 0.25 * cm),
        Paragraph("AI-assisted interpretation", styles["SectionHeading"]),
    ]
    for line in report["local_interpretation"]:
        story.append(Paragraph(escape(line), styles["BodyText"]))
        story.append(Spacer(1, 0.06 * cm))

    story.append(Paragraph("Setup", styles["SectionHeading"]))
    story.append(
        Paragraph(
            f"Input {IMAGE_SIZE}px. Max epochs {DEFAULT_EPOCHS}, patience {DEFAULT_PATIENCE}. "
            f"Best epoch {report['best_epoch']}/{report['epochs']}. "
            f"Model selection: mean task macro-F1. "
            f"Frozen splits: {escape(FROZEN_SPLITS_DIR.name)}. {escape(report['evaluation_note'])}",
            styles["BodyText"],
        )
    )
    for src in report.get("datasets", []):
        story.append(Paragraph(f"Dataset: {escape(src)}", styles["BodyText"]))

    story.append(Paragraph("Training history", styles["SectionHeading"]))
    hist = [["Epoch", "Train", "Val", "Score"]]
    for i, (a, b, c) in enumerate(
        zip(
            report["training_history"]["train_loss"],
            report["training_history"]["validation_loss"],
            report["training_history"]["validation_score"],
        ),
        1,
    ):
        hist.append([i, f"{a:.4f}", f"{b:.4f}", f"{c:.4f}"])
    ht = Table(hist, repeatRows=1)
    ht.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#174A70")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ]
        )
    )
    story.append(ht)
    story.append(PageBreak())
    story.append(Paragraph("Held-out test results", styles["SectionHeading"]))
    for task_key, metrics in report["task_metrics"].items():
        story.append(Paragraph(task_key.title(), styles["Heading3"]))
        summary = [
            ["Metric", "Value"],
            ["Accuracy", f"{metrics['accuracy']:.4f}"],
            [f"Meets {TARGET_ACC_NOTE:.0%} target note", str(metrics.get("meets_target_acc_0_92"))],
            ["Macro F1", f"{metrics['macro_f1']:.4f}"],
            [
                "Macro ROC AUC",
                f"{metrics['macro_roc_auc']:.4f}"
                if metrics["macro_roc_auc"] is not None
                else "N/A",
            ],
        ]
        st = Table(summary, colWidths=[8 * cm, 5 * cm], repeatRows=1)
        st.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#174A70")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ]
            )
        )
        story.append(st)
        ct = [["Class", "P", "R", "F1", "AUC", "N"]]
        for it in metrics["per_class"]:
            ct.append(
                [
                    it["class"],
                    f"{it['precision']:.3f}",
                    f"{it['recall']:.3f}",
                    f"{it['f1_score']:.3f}",
                    f"{it['roc_auc']:.3f}" if it["roc_auc"] is not None else "N/A",
                    str(it["support"]),
                ]
            )
        ctab = Table(ct, repeatRows=1)
        ctab.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#174A70")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                ]
            )
        )
        story.append(ctab)
        story.append(Spacer(1, 0.2 * cm))

    story.append(PageBreak())
    story.append(Paragraph("Charts and readings", styles["SectionHeading"]))
    chart_list = [
        ("training_validation_loss.png", "Training / validation loss"),
        ("cnn_training_extended.png", "Extended CNN training curves"),
        ("cnn_overfit_gap.png", "Overfit gap by epoch"),
        ("macro_summary_hbar.png", "Macro metrics horizontal summary"),
    ]
    for task_key in report["task_metrics"]:
        chart_list.extend(
            [
                (f"{task_key}_confusion_matrix.png", f"{task_key} confusion"),
                (f"{task_key}_class_metrics.png", f"{task_key} bar metrics"),
                (f"{task_key}_hbar_class_readings.png", f"{task_key} hbar P/R/F1"),
                (f"{task_key}_hbar_support.png", f"{task_key} support hbar"),
                (f"{task_key}_roc_curves.png", f"{task_key} ROC"),
            ]
        )
    for fname, cap in chart_list:
        p = output_dir / fname
        if p.is_file():
            story.append(Paragraph(escape(cap), styles["BodyText"]))
            story.append(_pdf_image(p, page_w - 4 * cm, 8 * cm))
            story.append(Spacer(1, 0.12 * cm))

    story.append(PageBreak())
    story.append(Paragraph("Grad-CAM", styles["SectionHeading"]))
    for path in sorted(output_dir.glob("gradcam_*.png")):
        story.append(Paragraph(escape(path.stem), styles["BodyText"]))
        story.append(_pdf_image(path, page_w - 4 * cm, 7 * cm))
    story.append(Paragraph("Limitations", styles["SectionHeading"]))
    story.append(Paragraph(escape(report["clinical_note"]), styles["BodyText"]))

    def page_num(canvas, document):
        canvas.setFont("Helvetica", 8)
        canvas.drawString(1.8 * cm, 0.8 * cm, AUTHOR_NAME)
        canvas.drawRightString(page_w - 1.8 * cm, 0.8 * cm, f"Page {document.page}")

    doc.build(story, onFirstPage=page_num, onLaterPages=page_num)
    return output_path


def generate_live_pdf(frame_gray, predictions, gradcam_path, hbar_path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_path = output_dir / "live_frame.png"
    cv2.imwrite(str(frame_path), frame_gray)

    pdf_path = output_dir / "live_interpretation_report.pdf"
    page_w, _ = A4
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, author=AUTHOR_NAME)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("Ultrasound AI — Inference Snapshot (Research)", styles["Title"]),
        Paragraph(f"Author: {escape(AUTHOR_NAME)}", styles["Normal"]),
        Paragraph(escape(predictions.get("summary", "")), styles["BodyText"]),
        Paragraph(
            "NOT a clinical diagnosis. Research prototype only. "
            "No external validation. ENABLE_TRIAGE="
            + str(ENABLE_TRIAGE),
            styles["BodyText"],
        ),
        Spacer(1, 0.2 * cm),
    ]
    if frame_path.is_file():
        story.append(_pdf_image(frame_path, page_w - 4 * cm, 7.5 * cm))
    if hbar_path and Path(hbar_path).is_file():
        story.append(Paragraph("Confidence readings", styles["Heading2"]))
        story.append(_pdf_image(Path(hbar_path), page_w - 4 * cm, 9 * cm))
    if gradcam_path and Path(gradcam_path).is_file():
        story.append(Paragraph("Grad-CAM (qualitative only)", styles["Heading2"]))
        story.append(_pdf_image(Path(gradcam_path), page_w - 4 * cm, 7.5 * cm))
    doc.build(story)
    return pdf_path


def _grad_nonzero(params) -> bool:
    return any(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in params
    )


@torch.no_grad()
def evaluate_task(model, loader, device, task: str):
    model.eval()
    targets, preds, probs = [], [], []
    for images, _, stage, tumor in loader:
        images = images.to(device)
        _, staging, tumor_logits = model(images)
        logits = staging if task == "fibrosis" else tumor_logits
        y = stage if task == "fibrosis" else tumor
        mask = y != -100
        if not mask.any():
            continue
        logits, y = logits[mask], y[mask]
        targets.extend(y.cpu().tolist())
        preds.extend(logits.argmax(1).cpu().tolist())
        probs.extend(torch.softmax(logits, 1).cpu().tolist())
    names = FIBROSIS_LABELS if task == "fibrosis" else TUMOR_LABELS
    if not targets:
        raise RuntimeError(f"No labels for {task}")
    return calculate_metrics(targets, preds, names, probs)


def train_engine(mode: str, epochs: int = DEFAULT_EPOCHS, patience: int = DEFAULT_PATIENCE):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[HARDWARE] {device.type.upper()} | epochs={epochs} patience={patience}")
    print(f"[CONFIG] IMAGE_SIZE={IMAGE_SIZE} ENABLE_TRIAGE={ENABLE_TRIAGE}")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    datasets_used, frames = [], []
    if mode in {"fibrosis", "concurrent"}:
        frames.append(load_task_metadata("fibrosis"))
        datasets_used.append(KAGGLE_DATASETS["fibrosis"])
    if mode in {"tumor", "concurrent"}:
        frames.append(load_task_metadata("tumor"))
        datasets_used.append(KAGGLE_DATASETS["tumor"])
    df = pd.concat(frames, ignore_index=True)
    splits = prepare_splits(df, mode)
    if mode == "concurrent":
        inspect_frozen_splits()

    datasets = {
        name: GrayscaleUltrasoundDataset(frame, augment=(name == "train"))
        for name, frame in splits.items()
    }
    gen = torch.Generator().manual_seed(SEED)
    loaders = {
        "train": DataLoader(
            datasets["train"], batch_size=BATCH_SIZE, shuffle=True, generator=gen, num_workers=0
        ),
        "validation": DataLoader(
            datasets["validation"], batch_size=BATCH_SIZE, shuffle=False, num_workers=0
        ),
        "test": DataLoader(
            datasets["test"], batch_size=BATCH_SIZE, shuffle=False, num_workers=0
        ),
    }

    if mode == "concurrent":
        fib_test_df = splits["test"][splits["test"]["fibrosis_stage"] != -100]
        t_test_df = splits["test"][splits["test"]["tumor_type"] != -100]
        fib_test_loader = DataLoader(
            GrayscaleUltrasoundDataset(fib_test_df), batch_size=BATCH_SIZE, shuffle=False
        )
        tum_test_loader = DataLoader(
            GrayscaleUltrasoundDataset(t_test_df), batch_size=BATCH_SIZE, shuffle=False
        )
    else:
        fib_test_loader = tum_test_loader = loaders["test"]

    model = HierarchicalMultiHeadCNN().to(device)
    fibrosis_weights = None
    tumor_weights = None
    if mode in {"fibrosis", "concurrent"}:
        counts = (
            splits["train"]
            .loc[splits["train"]["fibrosis_stage"] >= 0, "fibrosis_stage"]
            .value_counts()
            .reindex(range(5), fill_value=0)
            .to_numpy(dtype=np.float32)
        )
        if mode == "fibrosis" and np.any(counts == 0):
            raise ValueError("All F0-F4 must appear in training split.")
        if not np.any(counts == 0):
            fibrosis_weights = torch.tensor(
                counts.sum() / (5 * counts), dtype=torch.float32, device=device
            )
    if mode in {"tumor", "concurrent"}:
        tcounts = (
            splits["train"]
            .loc[splits["train"]["tumor_type"] >= 0, "tumor_type"]
            .value_counts()
            .reindex(range(2), fill_value=0)
            .to_numpy(dtype=np.float32)
        )
        if not np.any(tcounts == 0):
            tumor_weights = torch.tensor(
                tcounts.sum() / (2 * tcounts), dtype=torch.float32, device=device
            )

    criterion = ConcurrentMultiTaskLoss(
        fibrosis_weights=fibrosis_weights,
        tumor_weights=tumor_weights,
        use_triage=ENABLE_TRIAGE,
    )
    
    optimizer = optim.AdamW(
        [
            {"params": model.shared_cnn_backbone.parameters(), "lr": 1.5e-4},
            {"params": model.feature_projection.parameters(), "lr": 2e-4},
            {"params": model.master_triage_head.parameters(), "lr": 2.5e-4},
            {"params": model.diffuse_staging_head.parameters(), "lr": 2.5e-4},
            {"params": model.differential_tumor_head.parameters(), "lr": 2.5e-4},
        ],
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
    )

    history = {"train_loss": [], "validation_loss": [], "validation_score": []}
    best_score, best_val_loss, best_epoch = -1.0, float("inf"), 0
    best_state, stale = None, 0
    grad_checked = False

    print("\n" + "#" * 60)
    print(f"  TRAINING: {mode.upper()}  |  {AUTHOR_NAME}")
    print("#" * 60)

    for epoch in range(epochs):
        model.train()
        loss_sum, n = 0.0, 0
        for images, master, stage, tumor in loaders["train"]:
            images = images.to(device)
            master, stage, tumor = master.to(device), stage.to(device), tumor.to(device)
            triage, staging, tumor_logits = model(images)
            loss = criterion(triage, staging, tumor_logits, master, stage, tumor)
            optimizer.zero_grad()
            loss.backward()

            if (
                GRAD_CHECK_CONCURRENT
                and mode == "concurrent"
                and not grad_checked
                and (stage != -100).any()
                and (tumor != -100).any()
            ):
                bb = _grad_nonzero(model.shared_cnn_backbone.parameters())
                fh = _grad_nonzero(model.diffuse_staging_head.parameters())
                th = _grad_nonzero(model.differential_tumor_head.parameters())
                print(f"[GRAD CHECK] backbone={bb} fibrosis_head={fh} tumor_head={th}")
                if not (bb and fh and th):
                    print(
                        "[GRAD WARN] Expected non-zero grads on backbone + both heads "
                        "when batch has both tasks. Check label masks / loss."
                    )
                grad_checked = True

            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += loss.item() * images.size(0)
            n += images.size(0)
        train_loss = loss_sum / max(n, 1)

        model.eval()
        val_loss_sum, vn = 0.0, 0
        with torch.no_grad():
            for images, master, stage, tumor in loaders["validation"]:
                images = images.to(device)
                master, stage, tumor = master.to(device), stage.to(device), tumor.to(device)
                triage, staging, tumor_logits = model(images)
                loss = criterion(triage, staging, tumor_logits, master, stage, tumor)
                val_loss_sum += loss.item() * images.size(0)
                vn += images.size(0)
        val_loss = val_loss_sum / max(vn, 1)

        scores = []
        if mode in {"fibrosis", "concurrent"}:
            scores.append(
                evaluate_task(model, loaders["validation"], device, "fibrosis")["macro_f1"]
            )
        if mode in {"tumor", "concurrent"}:
            scores.append(
                evaluate_task(model, loaders["validation"], device, "tumor")["macro_f1"]
            )
        val_score = float(np.mean(scores))
        history["train_loss"].append(train_loss)
        history["validation_loss"].append(val_loss)
        history["validation_score"].append(val_score)
        scheduler.step(val_loss)

        improved = val_score > best_score or (
            val_score == best_score and val_loss < best_val_loss
        )
        if improved:
            best_score, best_val_loss, best_epoch, stale = (
                val_score,
                val_loss,
                epoch + 1,
                0,
            )
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            stale += 1
        print(
            f"Epoch [{epoch + 1}/{epochs}] train={train_loss:.4f} "
            f"val={val_loss:.4f} score={val_score:.4f}"
        )
        if stale >= patience:
            print(f"[EARLY STOP] {patience} epochs without improvement")
            break

    if best_state is None:
        raise RuntimeError("No checkpoint")
    model.load_state_dict(best_state)

    task_metrics, task_class_names = {}, {}
    if mode in {"fibrosis", "concurrent"}:
        loader = fib_test_loader if mode == "concurrent" else loaders["test"]
        task_metrics["fibrosis"] = evaluate_task(model, loader, device, "fibrosis")
        task_class_names["fibrosis"] = FIBROSIS_LABELS
    if mode in {"tumor", "concurrent"}:
        loader = tum_test_loader if mode == "concurrent" else loaders["test"]
        task_metrics["tumor"] = evaluate_task(model, loader, device, "tumor")
        task_class_names["tumor"] = TUMOR_LABELS

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = RESEARCH_ROOT / f"{mode}_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model_state.pt")
    if mode == "concurrent":
        torch.save(model.state_dict(), RESEARCH_ROOT / "best_multitask.pt")

    _plot_training_history(history, output_dir)
    _plot_training_extended(history, output_dir)
    for task_key, metrics in task_metrics.items():
        names = task_class_names[task_key]
        _plot_confusion_matrix(
            np.asarray(metrics["confusion_matrix"]), names, output_dir, task_key
        )
        _plot_class_metrics(metrics, output_dir, task_key)
        _plot_metric_hbars(metrics, output_dir, task_key)
        _plot_roc_curves(metrics["roc_curves"], output_dir, task_key)
        subset = (
            splits["test"][splits["test"]["fibrosis_stage"] != -100]
            if task_key == "fibrosis"
            else splits["test"][splits["test"]["tumor_type"] != -100]
        )
        save_gradcam(
            model,
            GrayscaleUltrasoundDataset(subset),
            task_key,
            names,
            output_dir,
            device,
            task_key,
        )
    _plot_macro_summary_hbar(task_metrics, output_dir)

    report = {
        "mode": mode,
        "author": AUTHOR_NAME,
        "datasets": datasets_used,
        "frozen_splits_dir": str(FROZEN_SPLITS_DIR),
        "enable_triage": ENABLE_TRIAGE,
        "image_size": IMAGE_SIZE,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "epochs": len(history["train_loss"]),
        "best_epoch": best_epoch,
        "training_history": history,
        "task_metrics": task_metrics,
        "task_class_names": task_class_names,
        "evaluation_note": (
            "Test metrics use frozen split CSVs so separate baselines and concurrent "
            "multi-task share the same held-out image IDs. "
            f"AUDIT_ENABLED={AUDIT_ENABLED}, NEAR_DUP={NEAR_DUP_ENABLED}. "
            f"Selection metric: mean task macro-F1. "
            f"Accuracy target note {TARGET_ACC_NOTE} is not enforced — numbers are honest. "
            + LIMITATION_SPLIT
        ),
        "clinical_note": (
            f"Research prototype by {AUTHOR_NAME}. Not for clinical diagnosis. "
            "No external multi-center validation. Grad-CAM is qualitative only. "
            f"Triage disabled (ENABLE_TRIAGE={ENABLE_TRIAGE}). "
            + LIMITATION_SPLIT
        ),
    }
    local = generate_local_interpretation(report)
    report["local_interpretation"] = maybe_llm_enrich_interpretation(report, local)
    (output_dir / "research_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    pdf_path = generate_research_pdf(report, output_dir)

    print("\n" + "=" * 60)
    print(f"  Author: {AUTHOR_NAME}")
    for k, m in task_metrics.items():
        print(
            f"  {k}: acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f} "
            f"target92={m.get('meets_target_acc_0_92')}"
        )
    print(f"  PDF: {pdf_path}")
    print("=" * 60)
    return output_dir


def write_fair_comparison():
    def latest_metrics(prefix: str):
        dirs = sorted(
            RESEARCH_ROOT.glob(f"{prefix}_*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for d in dirs:
            p = d / "research_metrics.json"
            if p.is_file():
                return json.loads(p.read_text(encoding="utf-8"))
        return None

    fib, tum, conc = (
        latest_metrics("fibrosis"),
        latest_metrics("tumor"),
        latest_metrics("concurrent"),
    )
    if not fib or not tum or not conc:
        print("[FAIR] Need fibrosis + tumor + concurrent metrics.")
        return
    rows = []
    if "fibrosis" in fib.get("task_metrics", {}) and "fibrosis" in conc.get(
        "task_metrics", {}
    ):
        a, b = fib["task_metrics"]["fibrosis"], conc["task_metrics"]["fibrosis"]
        rows.append(
            {
                "task": "fibrosis",
                "separate_accuracy": a["accuracy"],
                "separate_macro_f1": a["macro_f1"],
                "multitask_accuracy": b["accuracy"],
                "multitask_macro_f1": b["macro_f1"],
                "delta_macro_f1": b["macro_f1"] - a["macro_f1"],
                "separate_meets_0_92": a.get("meets_target_acc_0_92"),
                "multitask_meets_0_92": b.get("meets_target_acc_0_92"),
            }
        )
    if "tumor" in tum.get("task_metrics", {}) and "tumor" in conc.get("task_metrics", {}):
        a, b = tum["task_metrics"]["tumor"], conc["task_metrics"]["tumor"]
        rows.append(
            {
                "task": "tumor",
                "separate_accuracy": a["accuracy"],
                "separate_macro_f1": a["macro_f1"],
                "multitask_accuracy": b["accuracy"],
                "multitask_macro_f1": b["macro_f1"],
                "delta_macro_f1": b["macro_f1"] - a["macro_f1"],
                "separate_meets_0_92": a.get("meets_target_acc_0_92"),
                "multitask_meets_0_92": b.get("meets_target_acc_0_92"),
            }
        )
    out = RESEARCH_ROOT / "fair_comparison_latest"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "fair_comparison.csv", index=False)
    (out / "fair_comparison.json").write_text(
        json.dumps(
            {
                "author": AUTHOR_NAME,
                "note": "Same frozen held-out IDs; ablation separate vs concurrent heads",
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\n[ABLATION / FAIR COMPARISON]")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"[FAIR] {out / 'fair_comparison.csv'}")


def load_dicom_as_gray(path: Path) -> np.ndarray:
    path = Path(path)
    try:
        import pydicom
    except ImportError as e:
        raise RuntimeError("Install DICOM support: pip install pydicom") from e
    ds = pydicom.dcmread(str(path), force=True)
    arr = ds.pixel_array
    if arr.ndim == 3:
        if arr.shape[-1] in (3, 4):
            arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            arr = arr[0]
    arr = arr.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1) or 1)
    intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
    arr = arr * slope + intercept
    amin, amax = float(arr.min()), float(arr.max())
    if amax > amin:
        arr = (arr - amin) / (amax - amin) * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def load_any_gray(path: Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() in {".dcm", ".dicom"}:
        return load_dicom_as_gray(path)
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return img


def list_dicom_files(folder: Path) -> list:
    folder = Path(folder)
    files = [p for p in sorted(folder.rglob("*")) if p.is_file()]
    explicit = [p for p in files if p.suffix.lower() in {".dcm", ".dicom"}]
    if explicit:
        return explicit
    img_ext = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return [p for p in files if p.suffix.lower() in img_ext]


def live_source_configured() -> bool:
    return bool(
        os.environ.get("ULTRASOUND_DICOM", "").strip()
        or os.environ.get("ULTRASOUND_DICOM_DIR", "").strip()
        or os.environ.get("ULTRASOUND_DEVICE", "").strip()
    )


def open_live_source():
    dcm_file = os.environ.get("ULTRASOUND_DICOM", "").strip().strip('"')
    dcm_dir = os.environ.get("ULTRASOUND_DICOM_DIR", "").strip().strip('"')
    source = os.environ.get("ULTRASOUND_DEVICE", "").strip()

    if dcm_file:
        path = Path(dcm_file)
        if not path.is_file():
            print(f"[DICOM] Not found: {path}")
            return None, None
        return "dicom_file", path

    if dcm_dir:
        folder = Path(dcm_dir)
        if not folder.is_dir():
            print(f"[DICOM] Folder not found: {folder}")
            return None, None
        files = list_dicom_files(folder)
        if not files:
            print(f"[DICOM] No files in {folder}")
            return None, None
        print(f"[DICOM] Continuous folder stream: {len(files)} files (loop)")
        return "dicom_dir", files

    if not source:
        return None, None
    try:
        cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
        if not cap.isOpened():
            cap.release()
            print(f"[LIVE] Cannot open device: {source}")
            return None, None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print(f"[LIVE] Continuous camera stream: {source}")
        return "camera", cap
    except Exception as err:
        print(f"[LIVE] Camera error: {err}")
        return None, None


def read_next_frame(mode, handle, dir_index: list):
    if mode == "camera":
        ok, frame = handle.read()
        if not ok or frame is None:
            return None
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    if mode == "dicom_file":
        try:
            p = handle
            if p.suffix.lower() in {".dcm", ".dicom"}:
                return load_dicom_as_gray(p)
            return cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        except Exception as err:
            print(f"[DICOM] Read error: {err}")
            return None

    if mode == "dicom_dir":
        files = handle
        if not files:
            return None
        i = dir_index[0] % len(files)
        dir_index[0] = i + 1
        fp = files[i]
        try:
            if fp.suffix.lower() in {".dcm", ".dicom"}:
                return load_dicom_as_gray(fp)
            return cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
        except Exception as err:
            print(f"[DICOM] Skip {fp.name}: {err}")
            return None
    return None


def _frame_quality_ok(gray: np.ndarray) -> bool:
    if gray is None or gray.size == 0 or gray.ndim != 2:
        return False
    if float(gray.std()) < 5.0:
        return False
    m = float(gray.mean())
    if m < 5 or m > 250:
        return False
    return True


def preprocess_frame(gray: np.ndarray) -> torch.Tensor:
    gray = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE))
    x = (gray.astype(np.float32) / 255.0 - 0.456) / 0.224
    return torch.from_numpy(x).unsqueeze(0).unsqueeze(0)


def analyze_frame_heads(model, frame, device):
    if not _frame_quality_ok(frame):
        return {
            "ok": False,
            "unable": True,
            "summary": (
                f"Author {AUTHOR_NAME}. Input quality check failed. "
                "Unable to assess. Research prototype only — not a diagnosis."
            ),
            "primary": "unable",
            "bars": {},
            "triage_i": None,
            "triage_conf": 0.0,
            "fib_name": None,
            "fib_conf": 0.0,
            "tum_name": None,
            "tum_conf": 0.0,
            "overlay": "QC FAIL — Unable to assess",
            "f_prob": None,
            "u_prob": None,
        }

    model.eval()
    tensor = preprocess_frame(frame).to(device)
    with torch.no_grad():
        triage, staging, tumor_logits = model(tensor)
        t_prob = torch.softmax(triage, 1)[0].cpu().numpy()
        f_prob = torch.softmax(staging, 1)[0].cpu().numpy()
        u_prob = torch.softmax(tumor_logits, 1)[0].cpu().numpy()

    triage_i = int(t_prob.argmax())
    triage_conf = float(t_prob[triage_i])
    fib_name = FIBROSIS_LABELS[int(f_prob.argmax())]
    tum_name = TUMOR_LABELS[int(u_prob.argmax())]
    fib_conf = float(f_prob.max())
    tum_conf = float(u_prob.max())

    bars = {}
    if ENABLE_TRIAGE:
        bars.update(
            {f"Triage:{TRIAGE_LABELS[i]}": float(t_prob[i]) for i in range(len(TRIAGE_LABELS))}
        )
    bars.update({f"Fib:{n}": float(f_prob[i]) for i, n in enumerate(FIBROSIS_LABELS)})
    bars.update({f"Tum:{n}": float(u_prob[i]) for i, n in enumerate(TUMOR_LABELS)})

    if not ENABLE_TRIAGE:
        summary = (
            f"Author {AUTHOR_NAME}. Triage disabled (no verified Normal cohort). "
            f"Fibrosis head: {fib_name} ({fib_conf:.1%}). "
            f"Tumor head: {tum_name} ({tum_conf:.1%}). "
            "Parallel research scores only — not a diagnosis."
        )
        primary = "parallel"
        overlay = f"Fib {fib_name} {fib_conf:.0%} | Tum {tum_name} {tum_conf:.0%} (no triage)"
    elif triage_conf < TRIAGE_MIN_CONF:
        summary = (
            f"Author {AUTHOR_NAME}. Triage confidence low ({triage_conf:.1%}). "
            f"Unable to assess primary pathway. "
            f"Secondary: Fibrosis {fib_name} ({fib_conf:.1%}); Tumor {tum_name} ({tum_conf:.1%}). "
            "Research only — not a diagnosis."
        )
        primary = "unable"
        overlay = f"LOW CONF ({triage_conf:.0%}) — Unable to assess"
    elif triage_i == 0:
        summary = (
            f"Author {AUTHOR_NAME}. Primary: Fibrosis — {fib_name} ({fib_conf:.1%}). "
            f"Tumor secondary only: {tum_name} ({tum_conf:.1%}) — not asserted. "
            "Research only — not a diagnosis."
        )
        primary = "fibrosis"
        overlay = f"PRIMARY Fibrosis: {fib_name} {fib_conf:.0%} | sec Tumor {tum_name}"
    else:
        summary = (
            f"Author {AUTHOR_NAME}. Primary: Tumor — {tum_name} ({tum_conf:.1%}). "
            f"Fibrosis secondary only: {fib_name} ({fib_conf:.1%}) — not asserted. "
            "Research only — not a diagnosis."
        )
        primary = "tumor"
        overlay = f"PRIMARY Tumor: {tum_name} {tum_conf:.0%} | sec Fibrosis {fib_name}"

    return {
        "ok": True,
        "unable": primary == "unable",
        "summary": summary,
        "primary": primary,
        "bars": bars,
        "triage_i": triage_i if ENABLE_TRIAGE else None,
        "triage_conf": triage_conf if ENABLE_TRIAGE else 0.0,
        "fib_name": fib_name,
        "fib_conf": fib_conf,
        "tum_name": tum_name,
        "tum_conf": tum_conf,
        "overlay": overlay,
        "f_prob": f_prob,
        "u_prob": u_prob,
    }


def _live_gradcam(model, frame, device, pred_info: dict) -> np.ndarray | None:
    if pred_info.get("unable") or not pred_info.get("ok"):
        return None
    if pred_info.get("f_prob") is None or pred_info.get("u_prob") is None:
        return None
    captured = {}
    layer = model.shared_cnn_backbone[-1]

    def hook_fn(*args):
        out = args[-1]
        captured["activation"] = out
        out.register_hook(lambda g: captured.__setitem__("gradient", g))

    h = layer.register_forward_hook(hook_fn)
    try:
        model.zero_grad(set_to_none=True)
        tensor = preprocess_frame(frame).to(device)
        _, staging2, tumor2 = model(tensor)
        if not ENABLE_TRIAGE:
            if pred_info["fib_conf"] >= pred_info["tum_conf"]:
                score = staging2[0, int(np.argmax(pred_info["f_prob"]))]
                title = pred_info["fib_name"]
            else:
                score = tumor2[0, int(np.argmax(pred_info["u_prob"]))]
                title = pred_info["tum_name"]
        else:
            triage_i = pred_info["triage_i"]
            if triage_i == 0:
                score = staging2[0, int(np.argmax(pred_info["f_prob"]))]
                title = pred_info["fib_name"]
            else:
                score = tumor2[0, int(np.argmax(pred_info["u_prob"]))]
                title = pred_info["tum_name"]
        score.backward()
        act = captured["activation"].detach()[0]
        grad = captured["gradient"].detach()[0]
        weights = grad.mean(dim=(1, 2), keepdim=True)
        heat = torch.relu((weights * act).sum(dim=0))
        if heat.max() > 0:
            heat = heat / heat.max()
        heat_np = cv2.resize(heat.cpu().numpy(), (frame.shape[1], frame.shape[0]))
        heat_u8 = (heat_np * 255).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
        base = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame.copy()
        if base.shape[:2] != heat_color.shape[:2]:
            heat_color = cv2.resize(heat_color, (base.shape[1], base.shape[0]))
        blend = cv2.addWeighted(base, 0.55, heat_color, 0.45, 0)
        cv2.putText(
            blend,
            f"Grad-CAM: {title} (qualitative)",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return blend
    except Exception as err:
        print(f"[WARN] Grad-CAM skipped: {err}")
        return None
    finally:
        h.remove()


def save_live_snapshot(model, frame, device, pred_info: dict, root: Path | None = None) -> Path:
    root = root or LIVE_REPORT_ROOT
    output_dir = root / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    disp = (
        cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
        if frame is not None
        else np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.uint8)
    )
    hbar_path = None
    if pred_info.get("bars"):
        hbar_path = _plot_live_confidence_hbar(pred_info["bars"], output_dir)
    grad_path = output_dir / "live_gradcam.png"
    cam = _live_gradcam(model, frame, device, pred_info)
    if cam is not None:
        cv2.imwrite(str(grad_path), cam)
    else:
        grad_path = None
    (output_dir / "prediction.json").write_text(
        json.dumps(
            {k: v for k, v in pred_info.items() if k not in {"f_prob", "u_prob"}},
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )
    return generate_live_pdf(
        disp,
        {"summary": pred_info.get("summary", ""), "bars": pred_info.get("bars") or {}},
        grad_path,
        hbar_path,
        output_dir,
    )


def run_uploaded_image_inference(model, device, image_path: Path):
    image_path = Path(image_path)
    print(f"[INFERENCE] Uploaded image: {image_path}")
    gray = load_any_gray(image_path)
    pred = analyze_frame_heads(model, gray, device)
    print(pred.get("summary", ""))
    pdf = save_live_snapshot(model, gray, device, pred, root=INFERENCE_REPORT_ROOT)
    print(f"[INFERENCE PDF] {pdf}")
    try:
        open_pdf(pdf)
    except Exception:
        pass
    return pdf


def run_continuous_live(model, device):
    mode, handle = open_live_source()
    if mode is None:
        print("[LIVE] No continuous source available.")
        return

    print("[MODE] Continuous live imaging + analysis (research demo).")
    print(f"  Analyze interval: {LIVE_ANALYZE_INTERVAL_SEC:.2f}s")
    print("  Keys: [q] quit  [s] save PDF  [p] pause/resume")
    print("  NOT clinical real-time ultrasound.")

    window = "US Multi-Task CNN — Live (Research)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    dir_index = [0]
    last_analyze_t = 0.0
    last_pred = {
        "overlay": "Waiting for first analysis...",
        "summary": "",
        "ok": False,
        "unable": True,
        "bars": {},
    }
    analysis_count = 0
    paused = False
    t0 = datetime.now().timestamp()
    static_gray = None
    last_gray = None

    try:
        while True:
            now = datetime.now().timestamp()
            if LIVE_MAX_SECONDS > 0 and (now - t0) > LIVE_MAX_SECONDS:
                print("[LIVE] Max duration reached.")
                break

            if mode == "dicom_file":
                if static_gray is None:
                    static_gray = read_next_frame(mode, handle, dir_index)
                gray = static_gray
            else:
                gray = read_next_frame(mode, handle, dir_index)
                if gray is None and mode == "dicom_dir":
                    for _ in range(min(10, len(handle))):
                        gray = read_next_frame(mode, handle, dir_index)
                        if gray is not None:
                            break
                if gray is None and mode == "camera":
                    print("[LIVE] Camera frame grab failed; stopping.")
                    break
                if gray is None:
                    continue

            last_gray = gray

            if not paused and (now - last_analyze_t) >= LIVE_ANALYZE_INTERVAL_SEC:
                last_pred = analyze_frame_heads(model, gray, device)
                last_analyze_t = now
                analysis_count += 1
                print(f"[LIVE #{analysis_count}] {last_pred.get('overlay', '')}")
                if LIVE_PDF_EVERY_N > 0 and analysis_count % LIVE_PDF_EVERY_N == 0:
                    pdf = save_live_snapshot(model, gray, device, last_pred)
                    print(f"[INFERENCE PDF] {pdf}")

            vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if gray.ndim == 2 else gray.copy()
            lines = [
                "RESEARCH DEMO — NOT A DIAGNOSIS",
                last_pred.get("overlay", ""),
                f"Analyses: {analysis_count} | {'PAUSED' if paused else 'RUNNING'} | q quit | s save | p pause",
            ]
            y = 28
            for line in lines:
                cv2.putText(
                    vis, line[:95], (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA
                )
                cv2.putText(
                    vis, line[:95], (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 220, 40), 1, cv2.LINE_AA
                )
                y += 24

            cv2.imshow(window, vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[LIVE] Quit.")
                break
            if key == ord("p"):
                paused = not paused
                print(f"[LIVE] Analysis {'paused' if paused else 'resumed'}.")
            if key == ord("s") and last_gray is not None:
                pdf = save_live_snapshot(model, last_gray, device, last_pred)
                print(f"[INFERENCE PDF] {pdf}")
                try:
                    open_pdf(pdf)
                except Exception:
                    pass

            if mode == "dicom_dir":
                cv2.waitKey(max(1, int(LIVE_ANALYZE_INTERVAL_SEC * 200)))
    finally:
        if mode == "camera" and handle is not None:
            handle.release()
        cv2.destroyAllWindows()
        print(f"[LIVE] Stopped. Total analyses: {analysis_count}")


def _latest_dir(prefix: str):
    if not RESEARCH_ROOT.is_dir():
        return None
    cands = [
        p
        for p in RESEARCH_ROOT.glob(f"{prefix}_*")
        if p.is_dir() and (p / "research_report.pdf").is_file()
    ]
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


def _latest_pdf(prefix: str):
    d = _latest_dir(prefix)
    return (d / "research_report.pdf") if d else None


def models_ready() -> bool:
    marker = RESEARCH_ROOT / ".initial_training_complete"
    ckpt = RESEARCH_ROOT / "best_multitask.pt"
    return bool(
        marker.is_file()
        and ckpt.is_file()
        and _latest_pdf("fibrosis")
        and _latest_pdf("tumor")
    )


def mark_initial_training_complete():
    RESEARCH_ROOT.mkdir(parents=True, exist_ok=True)
    (RESEARCH_ROOT / ".initial_training_complete").write_text(
        datetime.now().astimezone().isoformat(timespec="seconds"), encoding="utf-8"
    )


def open_pdf(path: Path):
    path = path.resolve()
    if not path.is_file():
        print(f"[ERROR] Missing: {path}")
        return
    print(f"[OPEN] {path}")
    try:
        os.startfile(str(path))
    except AttributeError:
        subprocess.run(
            ["open" if sys.platform == "darwin" else "xdg-open", str(path)],
            check=False,
        )


def run_first_time_three_step_training():
    print("\n" + "=" * 60)
    print(f"  Author: {AUTHOR_NAME}")
    print("  FIRST TIME — automatic 3-step training")
    print(
        f"  Epochs={DEFAULT_EPOCHS}, patience={DEFAULT_PATIENCE}, "
        f"AUDIT={AUDIT_ENABLED}, NEAR_DUP={NEAR_DUP_ENABLED}, TRIAGE={ENABLE_TRIAGE}"
    )
    print("=" * 60)
    print("\n>>> STEP 1/3 FIBROSIS")
    train_engine("fibrosis")
    print("\n>>> STEP 2/3 TUMOR")
    train_engine("tumor")
    print("\n>>> STEP 3/3 CONCURRENT")
    train_engine("concurrent")
    mark_initial_training_complete()
    write_fair_comparison()
    inspect_frozen_splits()
    print("\n[SUCCESS] Setup complete.")
    print(f"  Frozen IDs: {FROZEN_SPLITS_DIR}")
    print(f"  Fair CSV:   {RESEARCH_ROOT / 'fair_comparison_latest' / 'fair_comparison.csv'}")
    print(f"  Live model: {RESEARCH_ROOT / 'best_multitask.pt'}")
    print(f"  Audit:      {RESEARCH_ROOT / 'audit_fibrosis_report.txt'}")
    print(f"  Audit:      {RESEARCH_ROOT / 'audit_tumor_report.txt'}")


def load_trained_model(device):
    ckpt = RESEARCH_ROOT / "best_multitask.pt"
    if not ckpt.is_file():
        d = _latest_dir("concurrent")
        if d and (d / "model_state.pt").is_file():
            ckpt = d / "model_state.pt"
        else:
            raise RuntimeError("No concurrent checkpoint")
    model = HierarchicalMultiHeadCNN().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    print(f"[MODEL] {ckpt}")
    return model


def post_training_menu():
    while True:
        print("\n" + "-" * 50)
        print(f"  {AUTHOR_NAME} — models ready")
        print("-" * 50)
        print("1. Show fibrosis PDF")
        print("2. Show tumor PDF")
        print("3. Show concurrent PDF")
        print("4. Show fair / ablation comparison CSV")
        print("5. Inspect frozen splits (same held-out IDs)")
        print("6. Infer uploaded image (path prompt)")
        print("7. Retrain all 3")
        print("8. Exit")
        print("-" * 50)
        choice = input("Select 1-8: ").strip()
        if choice == "1":
            pdf = _latest_pdf("fibrosis")
            open_pdf(pdf) if pdf else print("[ERROR] No fibrosis PDF")
        elif choice == "2":
            pdf = _latest_pdf("tumor")
            open_pdf(pdf) if pdf else print("[ERROR] No tumor PDF")
        elif choice == "3":
            pdf = _latest_pdf("concurrent")
            open_pdf(pdf) if pdf else print("[ERROR] No concurrent PDF")
        elif choice == "4":
            csv = RESEARCH_ROOT / "fair_comparison_latest" / "fair_comparison.csv"
            if csv.is_file():
                print(csv.read_text(encoding="utf-8"))
            else:
                write_fair_comparison()
        elif choice == "5":
            inspect_frozen_splits()
        elif choice == "6":
            path = input("Image path (png/jpg/dcm): ").strip().strip('"')
            if path:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = load_trained_model(device)
                run_uploaded_image_inference(model, device, Path(path))
        elif choice == "7":
            if input("Retrain all? yes/no: ").strip().lower() in {"y", "yes"}:
                
                if FROZEN_SPLITS_DIR.is_dir():
                    print(f"[INFO] Consider deleting {FROZEN_SPLITS_DIR} for fresh near-group splits.")
                run_first_time_three_step_training()
        elif choice in {"8", ""}:
            print("[STATUS] Exit.")
            sys.exit(0)
        else:
            print("Invalid.")


def parse_args():
    p = argparse.ArgumentParser(description="US multi-task CNN research pipeline")
    p.add_argument(
        "--image",
        type=str,
        default=os.environ.get("ULTRASOUND_IMAGE", ""),
        help="Uploaded grayscale ultrasound image/DICOM for offline inference",
    )
    p.add_argument("--retrain", action="store_true", help="Force 3-step retrain")
    p.add_argument("--inspect-splits", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("[RUNNING] Grayscale ultrasound multi-task pipeline")
    print(f"[AUTHOR] {AUTHOR_NAME}")
    print(
        f"[CONFIG] epochs={DEFAULT_EPOCHS} patience={DEFAULT_PATIENCE} "
        f"AUDIT={AUDIT_ENABLED} NEAR_DUP={NEAR_DUP_ENABLED} "
        f"TRIAGE={ENABLE_TRIAGE} SIZE={IMAGE_SIZE}"
    )
    print(
        f"[NOTE] Target accuracy note {TARGET_ACC_NOTE:.0%} is aspirational; "
        "held-out metrics are never fabricated."
    )
    print(f"[LIMITATION] {LIMITATION_SPLIT}")

    if args.inspect_splits:
        inspect_frozen_splits()
        sys.exit(0)

    if args.retrain or not models_ready():
        if not models_ready():
            print("[STATUS] First-time run detected — starting automatic 3-step training.")
        else:
            print("[STATUS] --retrain requested.")
        run_first_time_three_step_training()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.image.strip():
        model = load_trained_model(device)
        run_uploaded_image_inference(model, device, Path(args.image.strip().strip('"')))
        sys.exit(0)

    if live_source_configured():
        print("[STATUS] Live imaging env detected — continuous analysis mode.")
        model = load_trained_model(device)
        run_continuous_live(model, device)
        sys.exit(0)

    print("[STATUS] No live imaging / --image → report menu.")
    post_training_menu()