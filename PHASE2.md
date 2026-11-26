# Phase 2 — Modelling

Phase 1 produced a clean, de-duplicated, leak-free dataset. Phase 2 trains models on it.

The work is split three ways. So that our three sets of results can go in one table, we all
use the same data pipeline and the same scoring code: `neuroscan/data.py` and
`neuroscan/evaluate.py`. **Please don't write your own loader or your own metrics** — if the
inputs or the scoring differ, the numbers aren't comparable and the comparison table is
meaningless.

---

## Setup (once, per machine or Colab session)

```bash
pip install -r requirements-phase2.txt
python scripts/build_processed_data.py      # rebuilds processed_data/ (~300 MB, gitignored)
```

`processed_data/` is not in the repository because it is too large. The script rebuilds it
from `dataset/` and `data/cleaned_manifest.csv` and reproduces the Phase 1 output exactly
(verified byte-for-byte). It takes a few minutes and is skipped for files already present.

Check everything is in place before training:

```python
from neuroscan import data
print(data.verify(full=True))      # {'ok': True, 'counts': {'train': 4156, 'val': 1039, 'test': 1567}, ...}
```

---

## The contract

| Thing | Value | Why |
|---|---|---|
| Split | Phase 1's, from `cleaned_manifest.csv` | 4,156 / 1,039 / 1,567. **Never re-roll it** — a fresh split would put near-duplicates on both sides again |
| Seed | `42` everywhere | |
| Image size | 224 x 224 | |
| Classes | `glioma, meningioma, notumor, pituitary` | this exact order, from `preprocessing_stats.json` |
| Augmentation | flip / rotate 20° / zoom 0.15 / shift 0.1 | matches the original notebook so the leakage comparison is about data, not augmentation |
| Metrics | `evaluate.compute_metrics` | macro-F1 is the headline, accuracy reported alongside |
| Artefacts | `models/<name>.keras`, `results/<name>_metrics.json` | so `evaluate.comparison_table()` can find them |

### The preprocessing trap

Keras' EfficientNet models **contain their own normalisation layer and expect inputs in
0–255**. If you feed them the standardised values from Phase 1 (mean 0.1749, std 0.1953) the
data gets normalised twice and accuracy quietly drops. Pick the mode that matches your model:

```python
data.DataConfig(preprocess="efficientnet")   # EfficientNetB0  -> leaves 0-255
data.DataConfig(preprocess="resnet50")       # ResNet50        -> caffe-style
data.DataConfig(preprocess="mobilenetv2")    # MobileNetV2     -> [-1, 1]
data.DataConfig(preprocess="rescale")        # scratch CNN     -> [0, 1]
data.DataConfig(preprocess="standardize")    # scratch CNN     -> Phase 1 constants
```

Use `channels=3` for any ImageNet backbone, `channels=1` for a from-scratch CNN.

---

## Who does what

| Track | Owner | Scope |
|---|---|---|
| 1 | **Kainat** | Shared pipeline + `evaluate()`, a from-scratch CNN baseline, and the leakage experiment |
| 2 | **Laiba** | EfficientNetB0 transfer learning on the clean pipeline — the main model |
| 3 | **Mania** | A second backbone (ResNet50 or MobileNetV2) + Grad-CAM + error analysis |

Track 1 is the foundation the other two build on, so it goes first. Tracks 2 and 3 then run
in parallel and don't block each other.

**Work in your own notebook on your own branch.** Notebooks merge terribly — two people
editing one file will conflict on every output cell. Shared code belongs in `neuroscan/`.

---

## Typical training script

```python
from neuroscan import data, evaluate

ds = data.make_datasets(data.DataConfig(preprocess="efficientnet", channels=3))
info = ds["info"]

model = build_my_model(info["input_shape"])
history = model.fit(
    ds["train"],
    validation_data=ds["val"],
    epochs=15,
    class_weight=info["class_weights"],     # mild, from Phase 1
)

res = evaluate.evaluate_model(model, ds["test"], info["classes"],
                              name="effnet_clean", outdir="results")
model.save("models/effnet_clean.keras")
print(res["macro_f1"])
```

Then, once all three of us have run:

```python
evaluate.comparison_table("results")        # writes results/comparison.{csv,md}
```

---

## The leakage experiment (Track 1)

The original Phase 2 notebook called `flow_from_directory` on the raw `dataset/` folders, so
it trained on data containing 187 exact duplicates, 90 near-duplicates, and **161 training
images that also appear in the test set**. Its reported accuracy is therefore inflated by an
unknown amount — a model can score well by recognising images it memorised in training.

`data.make_leaky_datasets()` reproduces that old pipeline deliberately, sharing everything
else with the clean one. Training the identical model under both and comparing gives us a
number for how much the leakage was worth. That measurement is the payoff of Phase 1 and
belongs in the final report.

**Never quote a headline result from the leaky pipeline.** It exists only for this comparison.
