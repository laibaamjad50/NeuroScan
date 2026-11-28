"""Check that the web app scores images exactly as the models were evaluated.

Run from the repository root (needs dataset/ and processed_data/):

    python backend/check_backend.py                 # Track 2 model, full test set
    python backend/check_backend.py --limit 200     # quicker
    MODEL_PATH=models/resnet50_tuned.keras python backend/check_backend.py

1. Preprocessing: the app's raw-file -> 224x224 transform must reproduce every
   processed_data/test PNG pixel for pixel.
2. Scoring: the app's predictions on the raw test files must give the accuracy
   and macro-F1 recorded in results/<model>_metrics.json.

Reads only; writes nothing to data/, images/ or results/.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference import REPO_ROOT, Predictor, to_processed  # noqa: E402


def macro_f1(y, p, k):
    f1s = []
    for c in range(k):
        tp = np.sum((p == c) & (y == c))
        fp = np.sum((p == c) & (y != c))
        fn = np.sum((p != c) & (y == c))
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only the first N test images")
    args = ap.parse_args()

    import pandas as pd

    df = pd.read_csv(os.path.join(REPO_ROOT, "data", "cleaned_manifest.csv"))
    df = df[df["final_split"] == "test"].copy()
    for col in ("processed_path", "original_path"):
        df[col] = df[col].str.replace("\\", "/", regex=False)
    if args.limit:
        df = df.head(args.limit)

    # 1. preprocessing parity
    arrays, mismatched = [], []
    for row in df.itertuples(index=False):
        mine = to_processed(Image.open(os.path.join(REPO_ROOT, row.original_path)))
        ref = np.asarray(Image.open(os.path.join(REPO_ROOT, row.processed_path)).convert("L"))
        if mine.shape != ref.shape or not np.array_equal(mine, ref):
            mismatched.append(row.original_path)
        arrays.append(mine)
    print(f"preprocessing: {len(df) - len(mismatched)}/{len(df)} images identical to processed_data/")
    for p in mismatched[:5]:
        print("   differs:", p)

    # 2. scoring parity
    path = os.environ.get("MODEL_PATH", os.path.join("models", "effnet_tuned.keras"))
    path = path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)
    pred = Predictor(path, preprocess=os.environ.get("MODEL_PREPROCESS") or None)
    probs = pred.predict_arrays(np.stack(arrays))
    y, p = df["label_idx"].to_numpy(), probs.argmax(1)
    acc, f1 = float((p == y).mean()), macro_f1(y, p, len(pred.classes))
    print(f"model {pred.name} (preprocess={pred.preprocess}) on {len(df)} test images:")
    print(f"   accuracy {acc:.4f} | macro-F1 {f1:.4f}")

    ok = not mismatched
    if pred.metrics and not args.limit:
        ra, rf = pred.metrics["test_accuracy"], pred.metrics["test_macro_f1"]
        print(f"   results/{pred.name}_metrics.json: accuracy {ra:.4f} | macro-F1 {rf:.4f}")
        same = abs(acc - ra) < 1e-3 and abs(f1 - rf) < 1e-3
        print("   scoring:", "MATCHES the recorded evaluation" if same else "DOES NOT MATCH")
        ok = ok and same
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
