# NeuroScan — Brain Tumor MRI Classification

**Group:** Kainat Asghar (23L-0687) | Laiba Amjad (23L-0642) | Mania Ali (23L-0876)
**Course:** DataScience · FAST-NUCES, Lahore
**Current status:** Phase 1 complete (data cleaning, EDA, insights, preprocessing). Phase 2 (modelling) in progress — Track 1 done; see [Phase 2 — Modelling](#phase-2--modelling) and [PHASE2.md](PHASE2.md).

---

## Description

NeuroScan is a machine-learning project that classifies brain MRI slices into four categories: **Glioma, Meningioma, Pituitary tumor** and **No tumor**. Phase 1 (this repository so far) prepares the data: it checks the dataset for corrupt files, duplicates and train/test leakage, explores it with statistics and plots, and produces a clean, uniformly preprocessed dataset ready for model training.

## Problem Statement

Given a single brain MRI image, automatically predict which of the four classes it belongs to. Before any model is trained, we need to know whether the data is clean, balanced, consistent and free of leakage. Otherwise reported accuracy would be inflated or the model would learn shortcuts (such as image size or source) instead of the tumour itself.

## Dataset

| Item | Details |
|------|---------|
| **Source** | [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (Masoud Nickparvar, Kaggle) |
| **Size** | 7,200 images: 5,600 in `Training/` and 1,600 in `Testing/` (1,800 per class) |
| **After cleaning** | 6,762 images (438 removed, 6.08%) |
| **Features** | This is an image dataset, so the raw input is the MRI image itself. In the notebook we extract per-image properties for analysis: file format, colour mode, width, height, aspect ratio, file size, intensity statistics (mean, std, min, max, skew, kurtosis, entropy, near-black pixel fraction), a texture proxy (`edge_density`), and fingerprints (`md5`, 256-bit `dhash`) |
| **Target variable** | `label`, one of 4 classes: `glioma`, `meningioma`, `notumor`, `pituitary` |

Class counts before and after cleaning:

| Class | Raw (train / test) | Clean (train / test) |
|-------|--------------------|----------------------|
| Glioma | 1400 / 400 | 1396 / 383 |
| Meningioma | 1400 / 400 | 1287 / 396 |
| No Tumor | 1400 / 400 | 1154 / 388 |
| Pituitary | 1400 / 400 | 1358 / 400 |

## Data Cleaning Summary

Nothing is deleted silently. Every removed file gets a `drop_reason` and is recorded in `data/cleaning_log.csv`.

- **Missing / corrupt data:** checked for zero-byte files, non-image extensions, unreadable images, missing labels and gaps in the extracted features. None found.
- **Blank images:** none found (pixel std < 2 rule).
- **Exact duplicates (md5):** 187 repeated files within a split were removed (first copy kept).
- **Near duplicates (dHash, confirmed by pixel comparison):** 90 repeats within a split removed, and **161 training images that also appeared in the test set were removed from training** (data leakage). The test set was left untouched.
- **Outliers (IQR rule on size, brightness, contrast, entropy, edge density):** 1,811 images (26.8%) were flagged but **kept**, because unusual MRI scans are usually valid scans from different scanners or slice levels.
- **Inconsistent formats:** 446 distinct resolutions and mixed colour modes (RGB, RGBA, L, P). All images are converted to greyscale, padded to a square, and resized to **224×224** (saved as PNG).
- **Scaling:** pixels ÷ 255, then standardised with the training-split mean (0.1749) and std (0.1953). Class weights were computed from the training split.
- **Splits:** a stratified 80/20 train/validation split of the cleaned Training data (4,156 train / 1,039 val). The cleaned Testing folder (1,567 images) is the untouched test set.

## Key Findings

1. **Leakage was real.** 161 training images were duplicates of test images, and 438 files (6.08%) were removed in total.
2. **Classes are roughly balanced** after cleaning (imbalance ratio 1.21; largest class Glioma, smallest No Tumor). We will still report macro-F1 and use mild class weights.
3. **Resolution is a shortcut risk.** Cramér's V between resolution and class is 0.52. About 96% of Glioma, Meningioma and Pituitary images are 512×512, while only 1% of No Tumor images are. No Tumor images come in many sizes (up to 630×630). A model could learn image size instead of the tumour, which is one reason we pad and resize everything to the same shape and plan to check with Grad-CAM in Phase 2.
4. **Simple features already separate classes.** The most informative are `edge_density` (MI = 0.54, ε² = 0.59), `height` and `width`. No Tumor images are brighter (mean intensity ≈ 62 vs ≈ 35 for Glioma) and have much higher edge density, which is partly a source artefact.
5. **Redundant features:** height/width (r = 0.97), skew/kurtosis (0.84) and black fraction/entropy (−0.81).
6. **Train/test shift:** width, height, contrast (std) and edge density differ noticeably between Training and Testing (KS ≈ 0.10–0.12), so test performance may differ for reasons unrelated to model quality.
7. **Pixel space is highly structured.** More than 50 principal components are needed for 90% of the variance, and the classes overlap in PCA, which suggests a deep model (CNN) is needed rather than a simple classifier on raw pixels.

All 16 figures behind these findings are in [`images/`](images/), and [`images/README.md`](images/README.md) explains, figure by figure, what each one shows and which notebook section produced it.

## Project Structure

Committed to this repository:

```
.
├── Brain_Tumor_MRI_Phase1.ipynb   ← Phase 1 notebook (cleaning, EDA, preprocessing), outputs included
├── Phase2_Track1_Kainat.ipynb     ← Phase 2 Track 1 (baseline CNN + leakage experiment)
├── README.md
├── PHASE2.md                      ← Phase 2 contract: split, seed, preprocessing, track owners
├── requirements.txt               ← Python dependencies for Phase 1
├── requirements-phase2.txt        ← adds TensorFlow for Phase 2
├── .gitignore
├── dataset/                       ← raw Kaggle data (Training/, Testing/), 7,200 images
├── data/
│   ├── cleaning_log.csv           ← every removed file with its drop_reason (438 rows)
│   ├── cleaned_manifest.csv       ← the clean dataset: path, label, final split (6,762 rows)
│   ├── cleaned_metadata.csv       ← all extracted per-image features for the clean set
│   ├── preprocessing_stats.json   ← image size, scaling constants, class weights, split counts
│   └── leaked_test_images.json    ← the 155 test images duplicated in the raw training set
├── neuroscan/                     ← shared Phase 2 code, used by all three tracks
│   ├── data.py                    ← the input pipeline (clean, and the leaky one for comparison)
│   └── evaluate.py                ← one metric set for every model
├── scripts/
│   └── build_processed_data.py    ← rebuilds processed_data/ from the manifest
├── images/
│   ├── 01_*.png … 16_*.png       ← the 16 EDA figures
│   └── README.md                  ← a short note on what each figure tells us
├── models/
│   └── effnet_clean.keras         ← the clean-data EfficientNetB0 (Track 3 runs Grad-CAM on it)
└── results/
    ├── comparison.{csv,md}        ← every model's scores side by side
    ├── leakage_experiment.json    ← the leakage numbers and their decomposition
    ├── *_metrics.json / *_report.txt / *_confusion.png
    └── superseded/                ← the first evaluation pass, with a README explaining why
```

Created locally and excluded from the repository via `.gitignore`:

```
├── kaggle.json                    ← your own Kaggle API token (see "How to Run" below)
├── processed_data/                ← 224×224 greyscale PNGs split into train/val/test
│                                    (~6,762 images; `python scripts/build_processed_data.py`)
└── models/baseline_cnn.keras, models/effnet_leaky.keras
                                   ← 34 MB and needed by nobody else; kept on Drive
```

`data/`, `images/` and `results/` are regenerated by the notebooks; the copies here are the
outputs of the runs whose numbers are quoted throughout this README, so you can read every
result without running anything.

## How to Run the Notebook

1. **Get a Kaggle API token:** Kaggle → Settings → *Create New Token*, which downloads `kaggle.json`.
2. **Place the token:** put `kaggle.json` next to the notebook (on Colab: left sidebar → Files → upload), or save it to `~/.kaggle/kaggle.json`.
3. **Open `Brain_Tumor_MRI_Phase1.ipynb`** in Google Colab or Jupyter (Python 3.9+).
4. **Run all cells in order** (*Runtime → Run all*). The first cell installs the dependencies (the same ones listed in `requirements.txt`, which you can instead install up front with `pip install -r requirements.txt`), and the next downloads the dataset from Kaggle: [masoudnickparvar/brain-tumor-mri-dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset). The download is skipped if `dataset/` already exists.
5. **Check the outputs** in `data/`, `images/` and `processed_data/`. Feature extraction and preprocessing go through all 7,200 images, so expect a few minutes on a CPU.

All random operations use `SEED = 42`, so results are reproducible. No models are trained in Phase 1.

## Phase 2 — Modelling

The work is split three ways (the contract, and who owns what, is in [PHASE2.md](PHASE2.md)):

| Track | Owner | Status |
|---|---|---|
| 1 | Kainat | Shared pipeline, baseline CNN, leakage experiment — **done** |
| 2 | Laiba | EfficientNetB0 transfer learning — in progress |
| 3 | Mania | Second backbone, Grad-CAM, error analysis — in progress |

All models load through `neuroscan/data.py` and are scored by `neuroscan/evaluate.py`, so
results are comparable. Full results are in [`results/`](results/); the Track 1 notebook is
[`Phase2_Track1_Kainat.ipynb`](Phase2_Track1_Kainat.ipynb).

### Results so far

| Model | Accuracy | Macro-F1 | Tumours called "no tumour" |
|-------|----------|----------|------|
| EfficientNetB0 (clean data) | 0.8551 | 0.8500 | 5.6% |
| CNN from scratch (baseline) | 0.8060 | 0.7976 | 15.0% |

Transfer learning is worth about 5 points of accuracy here — and, more importantly, cuts the
clinically expensive error (a tumour predicted as "no tumour") from 15.0% to 5.6%. Accuracy
alone understates the difference, which is why we report macro-F1 and per-class recall.

### Phase 1 paid for itself: leakage inflates results by 3 points

The original modelling notebook trained straight off the raw `dataset/` folders, skipping all
of the Phase 1 cleaning. We trained the identical EfficientNetB0, with identical
hyperparameters, on both pipelines:

| Pipeline | Accuracy | Macro-F1 |
|----------|----------|----------|
| Clean (Phase 1 split) | 0.8551 | 0.8500 |
| Uncleaned (original notebook's) | 0.8858 | 0.8823 |
| **Inflation** | **+3.07** | **+3.23** |

The uncleaned pipeline also trained on 324 more images, so the gap alone does not prove
leakage caused it. Splitting the test set settles the question. 155 of the 1,567 test images
(9.9%) are near-duplicates of training images the uncleaned model saw — Phase 1 removed the
*training*-side copies and left the test-side twins, which is listed in
[`data/leaked_test_images.json`](data/leaked_test_images.json):

| Model | Leaked (155) | Non-leaked (1,412) |
|-------|--------------|--------------------|
| Trained on clean data | 0.5419 | 0.8895 |
| Trained on uncleaned data | **0.8258** | 0.8924 |
| Difference | **+28.39** | +0.29 |

On the images neither model memorised the two are within 0.3 points, so the extra training
data accounts for almost nothing. The whole advantage sits in the leaked slice, and the
decomposition reproduces the headline gap exactly:

```
0.099 x (+0.2839)  +  0.901 x (+0.0029)  =  +0.0307
observed gap                                +0.0307
```

Two caveats. Macro-F1 is not meaningful on the leaked subset (it contains no pituitary and
only 4 glioma), so those comparisons use accuracy. And the leakage is concentrated rather than
spread: 24.7% of the meningioma test set is compromised against 0% of pituitary. The dataset
ships augmented meningioma images (`Tr-aug-me_*`), and augmenting before splitting puts
near-copies of one scan on both sides — an argument for de-duplicating *before* the split.

### Known issue: fine-tuning

Both EfficientNet runs were initially scored on damaged weights. Unfreezing the base put every
BatchNormalization layer into training mode, which collapsed validation accuracy from 0.9355 to
0.7584 in one epoch, and `EarlyStopping(restore_best_weights=True)` resets per `fit()` call, so
it restored a phase-B epoch rather than phase A's best. Reloading the `ModelCheckpoint` recovered
16 points. Section 5 of the Track 1 notebook documents it; keep BatchNorm frozen when fine-tuning.

### Still to do

Tracks 2 and 3: tune EfficientNetB0 with working fine-tuning, add a second backbone for
comparison, and run Grad-CAM to confirm the model attends to the tumour rather than to image
dimensions — Phase 1 found resolution correlates with class at Cramér's V = 0.52.

---

⚠️ **Disclaimer:** For academic/research use only. Not a certified medical device.
