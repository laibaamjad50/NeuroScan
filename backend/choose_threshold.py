"""Choose the web app's "no tumour" threshold on the VALIDATION set.

Rule: the app says "no tumour" only when P(no tumour) >= t; otherwise it reports
the most likely tumour class. Missing a tumour costs more than flagging a
healthy scan for review, so we take the strictest t from a grid fixed in
advance that keeps healthy validation scans flagged as tumour within a budget
(default 5%).

The test set is NOT used to choose t. It is scored once afterwards, with and
without the rule, so the app's real behaviour can be reported.

Run from the repository root (needs processed_data/ and dataset/):

    python backend/choose_threshold.py
    MODEL_PATH=models/resnet50_tuned.keras python backend/choose_threshold.py

Writes backend/decision_rule.json, which the app reads at start-up. Touches
nothing in data/, images/ or results/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference import REPO_ROOT, RULE_PATH, Predictor, apply_rule  # noqa: E402

GRID = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)


def macro_f1(y, p, k):
    f1s = []
    for c in range(k):
        tp = np.sum((p == c) & (y == c))
        fp = np.sum((p == c) & (y != c))
        fn = np.sum((p != c) & (y == c))
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))


def score(pred, y, n512, nt, k) -> dict:
    tum = y != nt
    said_nt_other = (pred == nt) & ~n512
    return {
        "accuracy": round(float((pred == y).mean()), 4),
        "macro_f1": round(macro_f1(y, pred, k), 4),
        "tumours_missed": round(float(((pred == nt) & tum).sum() / tum.sum()), 4),
        "non512_tumours_missed": (round(float(((pred == nt) & tum & ~n512).sum() / (tum & ~n512).sum()), 4)
                                  if (tum & ~n512).any() else None),
        "healthy_flagged": round(float(((pred != nt) & ~tum).sum() / (~tum).sum()), 4),
        # of the "no tumour" answers given for non-512x512 scans, the share that were tumours
        "non512_notumor_wrong": (round(float((said_nt_other & tum).sum() / said_nt_other.sum()), 4)
                                 if said_nt_other.any() else None),
        "n": int(len(y)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=0.05,
                    help="max share of healthy validation scans flagged as tumour (default 0.05)")
    args = ap.parse_args()

    import pandas as pd

    path = os.environ.get("MODEL_PATH", os.path.join("models", "effnet_tuned.keras"))
    path = path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)
    os.environ.pop("NOTUMOR_THRESHOLD", None)
    pred = Predictor(path, preprocess=os.environ.get("MODEL_PREPROCESS") or None)
    nt, k = pred.notumor_idx, len(pred.classes)

    df = pd.read_csv(os.path.join(REPO_ROOT, "data", "cleaned_manifest.csv"))
    for col in ("processed_path", "original_path"):
        df[col] = df[col].str.replace("\\", "/", regex=False)

    data = {}
    for split in ("val", "test"):
        s = df[df["final_split"] == split]
        X = np.stack([np.asarray(Image.open(os.path.join(REPO_ROOT, p)).convert("L"))
                      for p in s["processed_path"]])
        n512 = np.array([Image.open(os.path.join(REPO_ROOT, p)).size == (512, 512)
                         for p in s["original_path"]])
        data[split] = (pred.predict_arrays(X), s["label_idx"].to_numpy(), n512)

    P, y, n512 = data["val"]
    print(f"{pred.name}: validation, budget {args.budget:.0%} healthy flagged")
    val_rows, chosen = {}, GRID[0]
    for t in GRID:
        r = score(apply_rule(P, nt, t), y, n512, nt, k)
        val_rows[str(t)] = r
        ok = r["healthy_flagged"] <= args.budget
        if ok:
            chosen = t
        print(f"  t={t:<5} tumours missed {r['tumours_missed']:.1%} | healthy flagged "
              f"{r['healthy_flagged']:.1%} | macro-F1 {r['macro_f1']:.4f} {'' if ok else '(over budget)'}")
    print(f"chosen threshold: {chosen}")

    P, y, n512 = data["test"]
    plain = score(P.argmax(1), y, n512, nt, k)
    ruled_pred = apply_rule(P, nt, chosen)
    ruled = score(ruled_pred, y, n512, nt, k)
    fired = ruled_pred != P.argmax(1)
    real = fired & (y != nt)
    ruled["rule_fired"] = {
        "n": int(fired.sum()),
        "actually_tumour": int(real.sum()),
        "type_right": int((ruled_pred[real] == y[real]).sum()),
    }
    print("test set (scored once, after choosing):")
    for name, r in (("plain argmax", plain), (f"rule t={chosen}", ruled)):
        print(f"  {name:<14} acc {r['accuracy']:.4f} | macro-F1 {r['macro_f1']:.4f} | tumours missed "
              f"{r['tumours_missed']:.1%} (non-512 {r['non512_tumours_missed']:.1%}) | healthy flagged "
              f"{r['healthy_flagged']:.1%} | non-512 'no tumour' answers wrong {r['non512_notumor_wrong']:.1%}")
    f = ruled["rule_fired"]
    print(f"  rule changed {f['n']} answers: {f['actually_tumour']} were tumours "
          f"(type right in {f['type_right']})")

    record = {
        "model": pred.name,
        "threshold": chosen,
        "rule": "say 'no tumour' only if P(no tumour) >= threshold, else the most likely tumour class",
        "selection": f"strictest threshold in {list(GRID)} with healthy validation scans flagged <= "
                     f"{args.budget:.0%}; test set not used for the choice",
        "validation": val_rows,
        "test_plain": plain,
        "test_with_rule": ruled,
    }
    with open(RULE_PATH, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
    print(f"wrote {os.path.relpath(RULE_PATH, REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
