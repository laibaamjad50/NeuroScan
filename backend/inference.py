"""Model loading, preprocessing and Grad-CAM for the NeuroScan web app.

The preprocessing here must match what the models were trained on, step for
step, or the app will quietly score worse than the numbers in results/:

    1. greyscale, centre-pad to a square with black, resize to 224x224 with
       LANCZOS                         (scripts/build_processed_data.py)
    2. cast to float32 0-255, grey -> RGB, then the backbone's own
       preprocess function             (neuroscan/data.py, _build)

`backend/check_backend.py` verifies step 1 byte-for-byte against processed_data/
and step 2 by re-scoring the test set through this module.

Grad-CAM follows Track 3's implementation (Phase2_Track3_Mania.ipynb, Sec. 9).
"""

from __future__ import annotations

import io
import json
import os
import base64
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_SIZE = (224, 224)

# Class order is fixed by data/preprocessing_stats.json; fall back to it if the
# file is missing so the app still starts from a bare checkout of backend/.
_DEFAULT_CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

LABELS = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No Tumour",
    "pituitary": "Pituitary",
}

DESCRIPTIONS = {
    "glioma": "A tumour that starts in the glial (supporting) cells of the brain or spine. "
              "Gliomas are the most common primary brain tumours.",
    "meningioma": "A tumour that arises from the meninges, the membranes around the brain and "
                  "spinal cord. Usually benign, but it still needs medical follow-up.",
    "notumor": "The model found none of the three tumour types in this scan. This is not a "
               "clearance: a qualified radiologist must read the scan.",
    "pituitary": "A tumour of the pituitary gland at the base of the brain, which regulates "
                 "many hormones. Often treatable with surgery, medication or radiation.",
}

COLORS = {
    "glioma": "#ef4444",
    "meningioma": "#f97316",
    "notumor": "#22c55e",
    "pituitary": "#a855f7",
}


def load_classes() -> list[str]:
    path = os.path.join(REPO_ROOT, "data", "preprocessing_stats.json")
    try:
        with open(path, encoding="utf-8") as fh:
            return list(json.load(fh)["classes"])
    except (OSError, KeyError, ValueError):
        return list(_DEFAULT_CLASSES)


# ---------------------------------------------------------------------------
# preprocessing
# ---------------------------------------------------------------------------

def pad_to_square(img: Image.Image, fill: int = 0) -> Image.Image:
    """Centre the image on a square black canvas (identical to Phase 1)."""
    w, h = img.size
    s = max(w, h)
    canvas = Image.new("L", (s, s), fill)
    canvas.paste(img, ((s - w) // 2, (s - h) // 2))
    return canvas


def to_processed(img: Image.Image) -> np.ndarray:
    """Any PIL image -> the 224x224 uint8 greyscale array the models saw."""
    grey = img.convert("L")
    return np.asarray(pad_to_square(grey).resize(IMG_SIZE, Image.LANCZOS), dtype=np.uint8)


def _preprocess_fn(mode: str):
    """Mirror of neuroscan.data._preprocess_fn for the backbones we deploy."""
    from tensorflow.keras import applications

    if mode == "efficientnet":
        return lambda x: x                      # EfficientNet normalises internally
    if mode == "resnet50":
        return applications.resnet50.preprocess_input
    if mode == "mobilenetv2":
        return applications.mobilenet_v2.preprocess_input
    if mode == "rescale":
        return lambda x: x / 255.0
    raise ValueError(f"unsupported preprocess mode {mode!r}")


def guess_preprocess(model_path: str) -> str:
    name = os.path.basename(model_path).lower()
    if "resnet" in name:
        return "resnet50"
    if "mobilenet" in name:
        return "mobilenetv2"
    if "effnet" in name or "efficientnet" in name:
        return "efficientnet"
    if "baseline" in name or "cnn" in name:
        return "rescale"
    return "efficientnet"


# ---------------------------------------------------------------------------
# Grad-CAM (Track 3)
# ---------------------------------------------------------------------------

def split_model(model):
    """Split a classifier into feature_fn (input -> last conv feature map) and
    head_fn (feature map -> probabilities). Same logic as Track 3."""
    import keras

    nested = [l for l in model.layers if isinstance(l, keras.Model)]
    if nested:
        base = nested[0]
        i = model.layers.index(base)
        pre = [l for l in model.layers[:i] if not isinstance(l, keras.layers.InputLayer)]
        head = model.layers[i + 1:]

        def feature_fn(x):
            for l in pre:
                x = l(x, training=False)
            return base(x, training=False)
    else:
        conv = [l for l in model.layers if len(l.output.shape) == 4][-1]
        i = model.layers.index(conv)
        fmodel = keras.Model(model.inputs, conv.output)
        head = model.layers[i + 1:]

        def feature_fn(x):
            return fmodel(x, training=False)

    def head_fn(a):
        for l in head:
            a = l(a, training=False)
        return a

    return feature_fn, head_fn


def _jet(v: np.ndarray) -> np.ndarray:
    """Matplotlib-style 'jet' colormap for values in [0, 1] -> uint8 RGB."""
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def overlay_png(grey: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> bytes:
    """Blend a [0,1] heatmap over the greyscale input, as Track 3's figures do."""
    base = np.repeat(grey[..., None], 3, axis=-1).astype(np.float32)
    heat = _jet(cam).astype(np.float32)
    out = ((1 - alpha) * base + alpha * heat).clip(0, 255).astype(np.uint8)
    return png_bytes(out)


def png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


# ---------------------------------------------------------------------------
# decision rule: be slower to say "no tumour"
# ---------------------------------------------------------------------------
#
# Missing a tumour costs far more than flagging a healthy scan for review, and
# the models miss tumours on scans unlike the training data (see README, "Web
# App"). So "no tumour" is only reported when P(no tumour) >= threshold;
# otherwise the most likely tumour class is reported. The threshold is chosen
# on the validation set by backend/choose_threshold.py and stored in
# backend/decision_rule.json. threshold 0.5 or below = plain argmax.

RULE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "decision_rule.json")


def apply_rule(probs: np.ndarray, notumor_idx: int, threshold: float) -> np.ndarray:
    """Class decisions for an (n, k) probability array under the rule."""
    probs = np.atleast_2d(probs)
    pred = probs.argmax(1)
    low = (pred == notumor_idx) & (probs[:, notumor_idx] < threshold)
    if low.any():
        masked = probs.copy()
        masked[:, notumor_idx] = -1.0
        pred[low] = masked[low].argmax(1)
    return pred


def load_rule(model_name: str) -> dict | None:
    """The saved rule, if it was chosen for this model."""
    if not os.path.exists(RULE_PATH):
        return None
    with open(RULE_PATH, encoding="utf-8") as fh:
        rule = json.load(fh)
    return rule if rule.get("model") == model_name else None


# ---------------------------------------------------------------------------
# the predictor
# ---------------------------------------------------------------------------

@dataclass
class Predictor:
    model_path: str
    preprocess: str | None = None
    classes: list[str] = field(default_factory=load_classes)

    def __post_init__(self) -> None:
        import tensorflow as tf
        import keras

        self.preprocess = self.preprocess or guess_preprocess(self.model_path)
        self._tf = tf
        self.model = keras.models.load_model(self.model_path, compile=False)
        self.channels = int(self.model.input_shape[-1])
        self._pre = _preprocess_fn(self.preprocess)
        self._feature_fn, self._head_fn = split_model(self.model)
        self.name = os.path.splitext(os.path.basename(self.model_path))[0]
        self.metrics = self._load_metrics()
        self.notumor_idx = self.classes.index("notumor")
        self.rule = load_rule(self.name)
        override = os.environ.get("NOTUMOR_THRESHOLD")
        if override:
            self.threshold, self.rule = float(override), None   # saved metrics no longer apply
        else:
            self.threshold = float(self.rule["threshold"]) if self.rule else 0.5

        # Fail at start-up, not on the first request, if the split is wrong.
        x = self.to_input(np.zeros((1, *IMG_SIZE), np.uint8))
        full = self.model(x, training=False).numpy()
        split = self._head_fn(self._feature_fn(x)).numpy()
        if not np.allclose(full, split, atol=1e-4):
            raise RuntimeError("Grad-CAM model split does not reproduce the model output")

    def _load_metrics(self) -> dict | None:
        path = os.path.join(REPO_ROOT, "results", f"{self.name}_metrics.json")
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as fh:
            m = json.load(fh)
        return {
            "test_accuracy": m.get("accuracy"),
            "test_macro_f1": m.get("macro_f1"),
            "tumour_missed_rate": m.get("tumour_missed_rate"),
            "n_test": m.get("n_samples"),
        }

    def to_input(self, grey_batch: np.ndarray):
        """uint8 (n,224,224) -> model input, exactly as neuroscan/data.py builds it."""
        tf = self._tf
        x = tf.cast(tf.convert_to_tensor(grey_batch), tf.float32)[..., None]
        if self.channels == 3:
            x = tf.image.grayscale_to_rgb(x)
        return self._pre(x)

    def predict_arrays(self, grey_batch: np.ndarray, batch_size: int = 64) -> np.ndarray:
        out = [self.model.predict(self.to_input(grey_batch[i:i + batch_size]), verbose=0)
               for i in range(0, len(grey_batch), batch_size)]
        return np.concatenate(out)

    def decide(self, probs: np.ndarray) -> np.ndarray:
        return apply_rule(probs, self.notumor_idx, self.threshold)

    def gradcam(self, grey: np.ndarray, class_idx: int | None = None):
        """Grad-CAM for one image and class (default: the top class).
        Returns (cam 224x224 in [0,1], probabilities)."""
        tf = self._tf
        x = self.to_input(grey[None])
        with tf.GradientTape() as tape:
            A = self._feature_fn(x)
            tape.watch(A)
            p = self._head_fn(A)
            score = tf.reduce_max(p, axis=1) if class_idx is None else p[:, class_idx]
        g = tape.gradient(score, A)
        w = tf.reduce_mean(g, axis=(1, 2), keepdims=True)
        cam = tf.nn.relu(tf.reduce_sum(w * A, axis=-1))
        cam = cam / (tf.reduce_max(cam, axis=(1, 2), keepdims=True) + 1e-8)
        cam = tf.image.resize(cam[..., None], IMG_SIZE, method="bilinear")[0, ..., 0]
        return cam.numpy(), p.numpy()[0]

    def analyse(self, img: Image.Image, with_gradcam: bool = True) -> dict:
        w, h = img.size
        grey = to_processed(img)

        probs = self.model.predict(self.to_input(grey[None]), verbose=0)[0]
        top = int(np.argmax(probs))
        idx = int(self.decide(probs)[0])
        cls = self.classes[idx]
        rule_applied = idx != top
        # Explain the class we report, which under the rule may not be the top one.
        cam = self.gradcam(grey, idx)[0] if with_gradcam else None

        def entry(i: int) -> dict:
            c = self.classes[i]
            return {
                "class": c,
                "label": LABELS.get(c, c),
                "probability": float(probs[i]),
                "percentage": round(float(probs[i]) * 100, 2),
                "color": COLORS.get(c, "#00d2c8"),
            }

        all_probs = sorted((entry(i) for i in range(len(self.classes))),
                           key=lambda d: d["probability"], reverse=True)

        warnings = []
        native_512 = (w, h) == (512, 512)
        prediction = {**entry(idx), "confidence": float(probs[idx]),
                      "description": DESCRIPTIONS.get(cls, ""),
                      "is_tumour": cls != "notumor",
                      "rule_applied": rule_applied,
                      "top_class": self.classes[top]}
        if rule_applied:
            # The model's own answer was a weak "no tumour". Report a possible
            # tumour, and only hint at the type: it is unreliable here.
            fired = (self.rule or {}).get("test_with_rule", {}).get("rule_fired")
            evidence = ""
            if fired and fired.get("actually_tumour"):
                evidence = (f" In our test set, {fired['actually_tumour']} of {fired['n']} scans like this "
                            f"were tumours, but the type was right for only {fired['type_right']} of them.")
            prediction.update(
                label="Possible Tumour",
                color="#f59e0b",
                likely_type=LABELS.get(cls, cls),
                notumor_probability=float(probs[top]),
                description=(f"The model leaned towards 'No Tumour' ({probs[top]:.0%}), but not enough to "
                             f"rule a tumour out (it needs {self.threshold:.0%}). If there is a tumour, the "
                             f"most likely type is {LABELS.get(cls, cls)} ({probs[idx]:.0%})." + evidence),
            )
            warnings.append("A tumour could not be ruled out. Have this scan reviewed by a radiologist.")
        elif cls == "notumor" and not native_512:
            # Scans in sizes other than 512x512 come from a source the models saw
            # almost no tumours from (README, "Web App"); the rate is measured on
            # the test set by choose_threshold.py, with the rule applied.
            rate = (self.rule or {}).get("test_with_rule", {}).get("non512_notumor_wrong")
            share = f"about 1 in {round(1 / rate)}" if rate else "a noticeable share of"
            warnings.append(
                f"This image is {w}x{h}, not 512x512. In our test set, {share} 'No Tumour' results "
                "on scans that weren't 512x512 were actually tumours. Have this scan reviewed by a "
                "radiologist."
            )
        if float(probs[idx]) < 0.6 and not rule_applied:
            warnings.append("Low confidence: the model is unsure between classes.")

        result = {
            "prediction": prediction,
            "all_probabilities": all_probs,
            "input": {"width": w, "height": h, "padded": w != h, "native_512": native_512},
            "model_input": data_url(png_bytes(grey)),
            "warnings": warnings,
        }
        if cam is not None:
            result["gradcam"] = data_url(overlay_png(grey, cam))
        return result

    def info(self) -> dict:
        return {
            "name": self.name,
            "path": os.path.relpath(self.model_path, REPO_ROOT),
            "preprocess": self.preprocess,
            "input_shape": list(self.model.input_shape[1:]),
            "classes": [LABELS.get(c, c) for c in self.classes],
            "metrics": self.metrics,                       # plain argmax, as in results/
            "notumor_threshold": self.threshold,
            "metrics_with_rule": (self.rule or {}).get("test_with_rule"),
        }
