"""Shared input pipeline for NeuroScan Phase 2.

All three modelling tracks build their datasets here, so that the only thing
that differs between our results is the model — not the data, the split, the
augmentation or the preprocessing.

What it reads (all produced by the Phase 1 notebook):

    data/cleaned_manifest.csv      6,762 rows; the train/val/test split is
                                   already decided and must not be re-rolled
    data/preprocessing_stats.json  class order, class weights, scaling constants
    processed_data/                224x224 greyscale PNGs

Two pipelines are provided:

    make_datasets()        the clean pipeline — the one we actually train on
    make_leaky_datasets()  the *old* pipeline, reading the raw directories the
                           way the original Phase 2 notebook did. It still
                           contains the 187 exact duplicates, 90 near-duplicates
                           and 161 train/test leaks that Phase 1 found. It
                           exists only so the leakage experiment can show how
                           much those inflate the reported accuracy. Never
                           report a headline number from it.

IMPORTANT — preprocessing. Keras' EfficientNet models carry their own
Normalization layer and expect inputs in [0, 255]. Feeding them the
standardised values from Phase 1 (train_mean 0.1749 / train_std 0.1953) would
normalise the data twice and quietly cost you accuracy. Pick the `preprocess`
mode that matches your backbone:

    "efficientnet"  leave as 0-255 float   (EfficientNetB0 — Laiba)
    "resnet50"      applications.resnet50.preprocess_input
    "mobilenetv2"   applications.mobilenet_v2.preprocess_input
    "rescale"       x / 255 -> [0, 1]      (from-scratch CNN)
    "standardize"   Phase 1 constants      (from-scratch CNN, zero-centred)
    "raw"           untouched float32

Typical use:

    from neuroscan import data
    ds = data.make_datasets(data.DataConfig(preprocess="efficientnet"))
    model.fit(ds["train"], validation_data=ds["val"],
              class_weight=ds["info"]["class_weights"], epochs=15)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable

import pandas as pd

__all__ = [
    "DataConfig",
    "repo_root",
    "load_stats",
    "load_manifest",
    "class_names",
    "class_weights",
    "verify",
    "make_datasets",
    "make_leaky_datasets",
]

SPLITS = ("train", "val", "test")


def repo_root() -> str:
    """The repository root, inferred from this file's location."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class DataConfig:
    """Everything that defines an input pipeline.

    Keep `seed`, `image_size` and the split fixed across all three tracks;
    vary only `preprocess`, `channels` and `batch_size` to suit your model.
    """

    root: str | None = None
    image_size: tuple[int, int] = (224, 224)
    channels: int = 3          # 3 for ImageNet backbones, 1 for a scratch CNN
    batch_size: int = 32
    preprocess: str = "efficientnet"
    augment: bool = True       # training split only, never val/test
    seed: int = 42
    shuffle_buffer: int = 4096
    cache: bool = False        # True is much faster, ~210 MB RAM for train
    validation_split: float = 0.2   # used by make_leaky_datasets only

    def __post_init__(self) -> None:
        if self.root is None:
            self.root = repo_root()
        if self.channels not in (1, 3):
            raise ValueError(f"channels must be 1 or 3, got {self.channels}")
        if self.preprocess not in _PREPROCESS_MODES:
            raise ValueError(
                f"unknown preprocess {self.preprocess!r}; "
                f"choose one of {sorted(_PREPROCESS_MODES)}"
            )


# --------------------------------------------------------------------------
# Phase 1 artefacts
# --------------------------------------------------------------------------

def load_stats(root: str | None = None) -> dict:
    """The contents of data/preprocessing_stats.json."""
    root = root or repo_root()
    with open(os.path.join(root, "data", "preprocessing_stats.json"), encoding="utf-8") as fh:
        return json.load(fh)


def load_manifest(root: str | None = None) -> pd.DataFrame:
    """data/cleaned_manifest.csv, with paths normalised for this platform."""
    root = root or repo_root()
    df = pd.read_csv(os.path.join(root, "data", "cleaned_manifest.csv"))
    for col in ("processed_path", "original_path"):
        if col in df.columns:
            df[col] = df[col].str.replace("\\", "/", regex=False)
    return df


def class_names(root: str | None = None) -> list[str]:
    """Class order. Everything downstream — labels, confusion matrices,
    report rows — must use this order."""
    return list(load_stats(root)["classes"])


def class_weights(root: str | None = None) -> dict[int, float]:
    """Phase 1 class weights, keyed by integer label as Keras expects."""
    return {int(k): float(v) for k, v in load_stats(root)["class_weights"].items()}


def verify(root: str | None = None, full: bool = False) -> dict:
    """Check the Phase 1 outputs are present and self-consistent.

    Run this before training; it is far cheaper than discovering a missing
    processed_data/ after half an epoch. With full=True every file on the
    manifest is stat-ed (a few seconds); otherwise only a sample is.
    """
    root = root or repo_root()
    stats, df = load_stats(root), load_manifest(root)
    names = list(stats["classes"])
    problems: list[str] = []

    if len(df) != stats["n_clean"]:
        problems.append(f"manifest has {len(df)} rows, stats say n_clean={stats['n_clean']}")

    counts = {s: int((df["final_split"] == s).sum()) for s in SPLITS}
    for split, expected in stats["counts"].items():
        want = sum(expected.values())
        if counts.get(split) != want:
            problems.append(f"{split}: manifest has {counts.get(split)}, stats say {want}")

    bad_labels = sorted(set(df["label"]) - set(names))
    if bad_labels:
        problems.append(f"labels not in the class list: {bad_labels}")

    mapping = df.groupby("label")["label_idx"].nunique()
    if (mapping > 1).any():
        problems.append("a label maps to more than one label_idx")
    for name, idx in stats["class_to_idx"].items():
        seen = df.loc[df["label"] == name, "label_idx"].unique()
        if len(seen) and int(seen[0]) != int(idx):
            problems.append(f"{name}: manifest idx {seen[0]} != stats idx {idx}")

    paths = df["processed_path"]
    checked = paths if full else paths.sample(min(200, len(paths)), random_state=0)
    missing = [p for p in checked if not os.path.exists(os.path.join(root, p))]
    if missing:
        problems.append(
            f"{len(missing)} of {len(checked)} checked files are missing, e.g. {missing[:3]}. "
            "processed_data/ is gitignored — re-run Section 7 of the Phase 1 notebook."
        )

    return {
        "ok": not problems,
        "problems": problems,
        "counts": counts,
        "n_rows": len(df),
        "classes": names,
        "checked_files": len(checked),
        "full_check": full,
    }


# --------------------------------------------------------------------------
# preprocessing
# --------------------------------------------------------------------------

def _preprocess_fn(mode: str, stats: dict) -> Callable:
    """Return a float32 tensor -> float32 tensor mapping for `mode`.

    Input is always 0-255 float32, matching what decode_png gives us.
    """
    import tensorflow as tf
    from tensorflow.keras import applications

    if mode in ("efficientnet", "raw"):
        # EfficientNet rescales and normalises inside the model.
        return lambda x: x
    if mode == "resnet50":
        return applications.resnet50.preprocess_input
    if mode == "mobilenetv2":
        return applications.mobilenet_v2.preprocess_input
    if mode == "rescale":
        return lambda x: x / 255.0
    if mode == "standardize":
        mean = tf.constant(stats["train_mean"], tf.float32)
        std = tf.constant(stats["train_std"], tf.float32)
        return lambda x: ((x / 255.0) - mean) / std
    raise ValueError(mode)


_PREPROCESS_MODES = {
    "efficientnet", "resnet50", "mobilenetv2", "rescale", "standardize", "raw",
}


def _augmenter(seed: int):
    """Augmentation matching the original Phase 2 notebook's ImageDataGenerator
    (horizontal_flip, rotation 20 deg, zoom 0.15, shift 0.1, fill 'nearest'),
    so the leakage experiment compares data, not augmentation policy.

    Note: horizontal flip is inherited from that notebook. Brain MRI is not
    left-right symmetric, so it is worth an ablation at some point.
    """
    from tensorflow.keras import layers, Sequential

    return Sequential(
        [
            layers.RandomFlip("horizontal", seed=seed),
            layers.RandomRotation(20 / 360, fill_mode="nearest", seed=seed),
            layers.RandomZoom(0.15, 0.15, fill_mode="nearest", seed=seed),
            layers.RandomTranslation(0.1, 0.1, fill_mode="nearest", seed=seed),
        ],
        name="augment",
    )


# --------------------------------------------------------------------------
# the clean pipeline
# --------------------------------------------------------------------------

def _build(paths, labels, cfg: DataConfig, *, training: bool, n_classes: int,
           stats: dict):
    import tensorflow as tf

    AUTOTUNE = tf.data.AUTOTUNE
    pre = _preprocess_fn(cfg.preprocess, stats)

    def decode(path, label):
        # Decode as 1 channel and keep it uint8 until after the cache: the
        # PNGs are greyscale, and caching float32 would cost 4x the memory
        # (830 MB rather than 208 MB for the training split).
        img = tf.io.decode_png(tf.io.read_file(path), channels=1)
        if tuple(cfg.image_size) != (224, 224):
            img = tf.cast(tf.image.resize(img, cfg.image_size), tf.uint8)
        return img, tf.one_hot(label, n_classes)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(decode, num_parallel_calls=AUTOTUNE)
    if cfg.cache:
        ds = ds.cache()
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.shuffle(min(cfg.shuffle_buffer, len(paths)), seed=cfg.seed,
                        reshuffle_each_iteration=True)
    ds = ds.batch(cfg.batch_size)
    if training and cfg.augment:
        aug = _augmenter(cfg.seed)
        ds = ds.map(lambda x, y: (aug(x, training=True), y),
                    num_parallel_calls=AUTOTUNE)
    if cfg.channels == 3:
        ds = ds.map(lambda x, y: (tf.image.grayscale_to_rgb(x), y),
                    num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda x, y: (pre(x), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


def make_datasets(cfg: DataConfig | None = None) -> dict:
    """The clean pipeline: Phase 1's cleaned, de-duplicated, leak-free split.

    Returns {"train", "val", "test", "info"}. val and test are never shuffled
    and never augmented, so predictions line up with the manifest order and
    `evaluate` can trust y_true.
    """
    cfg = cfg or DataConfig()
    stats, df = load_stats(cfg.root), load_manifest(cfg.root)
    names = list(stats["classes"])

    report = verify(cfg.root)
    if not report["ok"]:
        raise RuntimeError(
            "Phase 1 outputs are not usable:\n  - " + "\n  - ".join(report["problems"])
        )

    out: dict = {}
    counts = {}
    for split in SPLITS:
        part = df[df["final_split"] == split]
        paths = [os.path.join(cfg.root, p) for p in part["processed_path"]]
        labels = part["label_idx"].astype(int).tolist()
        counts[split] = len(paths)
        out[split] = _build(paths, labels, cfg, training=(split == "train"),
                            n_classes=len(names), stats=stats)

    out["info"] = {
        "pipeline": "clean",
        "classes": names,
        "class_weights": class_weights(cfg.root),
        "counts": counts,
        "steps_per_epoch": -(-counts["train"] // cfg.batch_size),
        "validation_steps": -(-counts["val"] // cfg.batch_size),
        "input_shape": (*cfg.image_size, cfg.channels),
        "preprocess": cfg.preprocess,
        "config": cfg,
    }
    return out


def y_true(split: str, root: str | None = None) -> "list[int]":
    """Ground-truth labels for a split, in the same order the dataset yields
    them. Only valid for val/test, which are not shuffled."""
    if split == "train":
        raise ValueError("train is shuffled; y_true is meaningless for it")
    df = load_manifest(root)
    return df.loc[df["final_split"] == split, "label_idx"].astype(int).tolist()


# --------------------------------------------------------------------------
# the leaky pipeline — for the experiment only
# --------------------------------------------------------------------------

def make_leaky_datasets(cfg: DataConfig | None = None) -> dict:
    """Reproduce the ORIGINAL Phase 2 pipeline: raw dataset/Training and
    dataset/Testing, with a random 20% validation split.

    This is the contaminated baseline. It reads the uncleaned directories, so
    its training set still contains images that also appear in its test set.
    Everything else — augmentation, preprocessing, batch size, seed — is shared
    with make_datasets(), so a difference in results is attributable to the
    data cleaning and nothing else.
    """
    import tensorflow as tf

    cfg = cfg or DataConfig()
    stats = load_stats(cfg.root)
    names = list(stats["classes"])
    AUTOTUNE = tf.data.AUTOTUNE
    pre = _preprocess_fn(cfg.preprocess, stats)

    train_dir = os.path.join(cfg.root, "dataset", "Training")
    test_dir = os.path.join(cfg.root, "dataset", "Testing")
    for d in (train_dir, test_dir):
        if not os.path.isdir(d):
            raise FileNotFoundError(f"missing {d} — the raw dataset is needed for this comparison")

    common = dict(labels="inferred", label_mode="categorical", class_names=names,
                  color_mode="grayscale", image_size=cfg.image_size,
                  batch_size=cfg.batch_size, interpolation="bilinear")

    raw_train = tf.keras.utils.image_dataset_from_directory(
        train_dir, validation_split=cfg.validation_split, subset="training",
        seed=cfg.seed, shuffle=True, **common)
    raw_val = tf.keras.utils.image_dataset_from_directory(
        train_dir, validation_split=cfg.validation_split, subset="validation",
        seed=cfg.seed, shuffle=False, **common)
    raw_test = tf.keras.utils.image_dataset_from_directory(
        test_dir, shuffle=False, **common)

    # file_paths is exact; cardinality() counts batches and would round the
    # final partial batch up to a full one.
    counts = {k: len(v.file_paths)
              for k, v in (("train", raw_train), ("val", raw_val), ("test", raw_test))}

    def finish(ds, training):
        if training and cfg.augment:
            aug = _augmenter(cfg.seed)
            ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE)
        if cfg.channels == 3:
            ds = ds.map(lambda x, y: (tf.image.grayscale_to_rgb(x), y), num_parallel_calls=AUTOTUNE)
        return ds.map(lambda x, y: (pre(x), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    return {
        "train": finish(raw_train, True),
        "val": finish(raw_val, False),
        "test": finish(raw_test, False),
        "info": {
            "pipeline": "leaky (uncleaned — comparison only)",
            "classes": names,
            "class_weights": None,
            "counts": counts,
            "input_shape": (*cfg.image_size, cfg.channels),
            "preprocess": cfg.preprocess,
            "config": cfg,
        },
    }
