"""Shared evaluation for NeuroScan Phase 2.

One metric set, computed one way, for every model in all three tracks. If each
of us writes our own scoring code we cannot put the numbers in the same table.

Why macro-F1 and not accuracy: the classes are only roughly balanced
(imbalance ratio 1.21), and clinically a missed tumour is worse than a false
alarm, so per-class recall matters more than the overall hit rate. Accuracy is
reported too, because it is what the original notebook reported.

The metric maths lives in `compute_metrics`, which takes plain arrays and
needs no TensorFlow — so it can be unit-tested. `evaluate_model` is the thin
wrapper that runs a Keras model over a dataset and calls it.

Typical use:

    from neuroscan import evaluate
    res = evaluate.evaluate_model(model, ds["test"], class_names,
                                  name="effnet_clean", outdir="results")
    print(res["macro_f1"])
    ...
    evaluate.comparison_table("results")      # once everyone has run
"""

from __future__ import annotations

import json
import os
from typing import Sequence

import numpy as np

__all__ = [
    "compute_metrics",
    "evaluate_model",
    "save_results",
    "plot_confusion",
    "plot_history",
    "comparison_table",
]


def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
    y_prob: np.ndarray | None = None,
) -> dict:
    """Accuracy, macro/weighted P-R-F1, per-class scores and the confusion
    matrix, as plain Python types ready for JSON.

    y_prob, if given, adds macro one-vs-rest ROC AUC and mean confidence.
    """
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 confusion_matrix, precision_recall_fscore_support,
                                 roc_auc_score)

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true {y_true.shape} and y_pred {y_pred.shape} differ in shape")
    n = len(class_names)
    labels = list(range(n))

    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    macro = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    out = {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_f1": float(weighted[2]),
        "per_class": {
            name: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(class_names)
        },
        "confusion_matrix": cm.tolist(),
        "class_names": list(class_names),
    }

    # The clinically interesting failure: a tumour called "no tumour".
    if "notumor" in class_names:
        ni = list(class_names).index("notumor")
        tumour = y_true != ni
        if tumour.any():
            out["tumour_missed_as_notumor"] = int((y_pred[tumour] == ni).sum())
            out["tumour_missed_rate"] = float((y_pred[tumour] == ni).mean())

    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=float)
        if y_prob.shape[0] == len(y_true) and y_prob.ndim == 2:
            try:
                out["macro_roc_auc"] = float(
                    roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
            except ValueError:
                out["macro_roc_auc"] = None
            out["mean_confidence"] = float(y_prob.max(axis=1).mean())
    return out


def evaluate_model(model, ds, class_names: Sequence[str], name: str,
                   outdir: str = "results", save: bool = True) -> dict:
    """Run `model` over `ds` and score it. `ds` must be unshuffled."""
    probs = model.predict(ds, verbose=0)
    truths = np.concatenate([y.numpy() for _, y in ds], axis=0)
    y_true = truths.argmax(axis=1) if truths.ndim > 1 else truths.astype(int)
    y_pred = probs.argmax(axis=1)
    if len(y_true) != len(y_pred):
        raise RuntimeError(
            f"{len(y_true)} labels vs {len(y_pred)} predictions — was the dataset shuffled?")

    metrics = compute_metrics(y_true, y_pred, class_names, y_prob=probs)
    metrics["model"] = name
    if save:
        save_results(name, metrics, outdir)
        plot_confusion(np.array(metrics["confusion_matrix"]), class_names,
                       os.path.join(outdir, f"{name}_confusion.png"), title=name)
    return metrics


def save_results(name: str, metrics: dict, outdir: str = "results") -> str:
    """Write <outdir>/<name>_metrics.json and a readable report alongside it."""
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{name}_metrics.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    lines = [f"{name}", "=" * len(name),
             f"samples            {metrics['n_samples']}",
             f"accuracy           {metrics['accuracy']:.4f}",
             f"balanced accuracy  {metrics['balanced_accuracy']:.4f}",
             f"macro F1           {metrics['macro_f1']:.4f}",
             f"macro precision    {metrics['macro_precision']:.4f}",
             f"macro recall       {metrics['macro_recall']:.4f}", "",
             f"{'class':<14}{'prec':>8}{'rec':>8}{'f1':>8}{'n':>7}"]
    for cls, m in metrics["per_class"].items():
        lines.append(f"{cls:<14}{m['precision']:>8.4f}{m['recall']:>8.4f}"
                     f"{m['f1']:>8.4f}{m['support']:>7}")
    if "tumour_missed_as_notumor" in metrics:
        lines += ["", f"tumours called 'no tumour': {metrics['tumour_missed_as_notumor']} "
                      f"({metrics['tumour_missed_rate']:.2%})"]
    with open(os.path.join(outdir, f"{name}_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def plot_confusion(cm: np.ndarray, class_names: Sequence[str], path: str,
                   title: str = "") -> str:
    """Confusion matrix, counts on the left and row-normalised on the right."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = np.asarray(cm)
    with np.errstate(invalid="ignore", divide="ignore"):
        norm = np.nan_to_num(cm / cm.sum(axis=1, keepdims=True))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, mat, fmt, lbl in ((axes[0], cm, "d", "counts"),
                              (axes[1], norm, ".2f", "row-normalised (recall)")):
        sns.heatmap(mat, annot=True, fmt=fmt, cmap="Blues", cbar=False,
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(lbl)
    fig.suptitle(f"Confusion matrix — {title}", fontweight="bold")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_history(histories, path: str, title: str = "") -> str:
    """Training curves. Pass one Keras History (or its .history dict), or a
    list of them to concatenate phase A and phase B into one plot."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    if not isinstance(histories, (list, tuple)):
        histories = [histories]
    hs = [h.history if hasattr(h, "history") else h for h in histories]

    merged: dict[str, list] = {}
    for h in hs:
        for k, v in h.items():
            merged.setdefault(k, []).extend(list(v))
    boundaries, run = [], 0
    for h in hs[:-1]:
        run += len(next(iter(h.values())))
        boundaries.append(run)

    pairs = [("accuracy", "val_accuracy"), ("loss", "val_loss")]
    fig, axes = plt.subplots(1, len(pairs), figsize=(13, 4.5))
    for ax, (tr, va) in zip(np.atleast_1d(axes), pairs):
        for key, style in ((tr, "-"), (va, "--")):
            if key in merged:
                ax.plot(range(1, len(merged[key]) + 1), merged[key], style, label=key)
        for b in boundaries:
            ax.axvline(b + 0.5, color="grey", ls=":", lw=1)
        ax.set_xlabel("epoch")
        ax.set_title(tr)
        ax.legend()
    fig.suptitle(f"Training curves — {title}" if title else "Training curves",
                 fontweight="bold")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def comparison_table(results_dir: str = "results", save: bool = True):
    """Collect every *_metrics.json into one table, best macro-F1 first."""
    import pandas as pd

    rows = []
    for fn in sorted(os.listdir(results_dir)):
        if not fn.endswith("_metrics.json"):
            continue
        with open(os.path.join(results_dir, fn), encoding="utf-8") as fh:
            m = json.load(fh)
        row = {
            "model": m.get("model", fn[: -len("_metrics.json")]),
            "n": m["n_samples"],
            "accuracy": round(m["accuracy"], 4),
            "balanced_acc": round(m["balanced_accuracy"], 4),
            "macro_f1": round(m["macro_f1"], 4),
            "macro_recall": round(m["macro_recall"], 4),
        }
        if "macro_roc_auc" in m and m["macro_roc_auc"] is not None:
            row["macro_auc"] = round(m["macro_roc_auc"], 4)
        if "tumour_missed_rate" in m:
            row["tumour_missed"] = round(m["tumour_missed_rate"], 4)
        rows.append(row)

    if not rows:
        raise FileNotFoundError(f"no *_metrics.json found in {results_dir}")
    df = pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    if save:
        df.to_csv(os.path.join(results_dir, "comparison.csv"), index=False)
        with open(os.path.join(results_dir, "comparison.md"), "w", encoding="utf-8") as fh:
            fh.write(_markdown(df))
    return df


def _markdown(df) -> str:
    """Render a DataFrame as a markdown table.

    Written by hand rather than via DataFrame.to_markdown, which needs the
    optional `tabulate` package that is not guaranteed to be installed.
    """
    cols = [str(c) for c in df.columns]
    rows = [[("" if v is None else str(v)) for v in rec] for rec in df.itertuples(index=False)]
    widths = [max(len(c), *(len(r[i]) for r in rows)) if rows else len(c)
              for i, c in enumerate(cols)]
    head = "| " + " | ".join(c.ljust(w) for c, w in zip(cols, widths)) + " |"
    rule = "|-" + "-|-".join("-" * w for w in widths) + "-|"
    body = ["| " + " | ".join(v.ljust(w) for v, w in zip(r, widths)) + " |" for r in rows]
    return "\n".join([head, rule, *body]) + "\n"
