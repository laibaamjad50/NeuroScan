# NeuroScan — Brain Tumor MRI Classification

**Group:** Kainat Asghar (23L-0687) | Laiba Amjad (23L-0642) | Mania Ali (23L-0876)
**Course:** Machine Learning · FAST-NUCES, Lahore
**Current status:** Phase 1 complete (data cleaning, EDA, insights, preprocessing). Modelling is Phase 2.

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

## Project Structure

```
project/
├── Brain_Tumor_MRI_Phase1.ipynb   ← Phase 1 notebook (cleaning, EDA, preprocessing)
├── kaggle.json                    ← your Kaggle API token (not committed)
├── dataset/                       ← downloaded automatically (Training/, Testing/)
├── data/                          ← cleaning_log.csv, cleaned_manifest.csv,
│                                     cleaned_metadata.csv, preprocessing_stats.json
├── images/                        ← 16 EDA figures
└── processed_data/                ← 224×224 greyscale PNGs split into train/val/test
```

`dataset/`, `data/`, `images/` and `processed_data/` are created when the notebook runs.

## How to Run the Notebook

1. **Get a Kaggle API token:** Kaggle → Settings → *Create New Token*, which downloads `kaggle.json`.
2. **Place the token:** put `kaggle.json` next to the notebook (on Colab: left sidebar → Files → upload), or save it to `~/.kaggle/kaggle.json`.
3. **Open `Brain_Tumor_MRI_Phase1.ipynb`** in Google Colab or Jupyter (Python 3.9+).
4. **Run all cells in order** (*Runtime → Run all*). The first cell installs the dependencies (`numpy pandas matplotlib seaborn scipy scikit-learn pillow tqdm kaggle`), and the next downloads the dataset from Kaggle: [masoudnickparvar/brain-tumor-mri-dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset). The download is skipped if `dataset/` already exists.
5. **Check the outputs** in `data/`, `images/` and `processed_data/`. Feature extraction and preprocessing go through all 7,200 images, so expect a few minutes on a CPU.

All random operations use `SEED = 42`, so results are reproducible. No models are trained in Phase 1.

## Next Steps (Phase 2)

Load `data/cleaned_manifest.csv`, build the input pipeline from `processed_data/` (replicating the grey channel to 3 channels for ImageNet backbones such as EfficientNetB0), train and evaluate models using accuracy, macro-F1 and a confusion matrix, and use Grad-CAM to confirm the model looks at the tumour and not at image size or borders.

---

⚠️ **Disclaimer:** For academic/research use only. Not a certified medical device.
