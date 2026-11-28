"""NeuroScan, Flask backend.

Group: Kainat Asghar (23L-0687) | Laiba Amjad (23L-0642) | Mania Ali (23L-0876)

Run from the repository root:

    pip install -r backend/requirements.txt
    python backend/app.py

then open http://localhost:5000 (the backend also serves the frontend).

Which model is served:
    MODEL_PATH        default models/effnet_tuned.keras (Track 2, committed)
    MODEL_PREPROCESS  efficientnet | resnet50 | mobilenetv2 | rescale
                      (default: guessed from the file name)

To serve Track 3's ResNet50, download models/resnet50_tuned.keras from Drive and
set MODEL_PATH=models/resnet50_tuned.keras.
"""

from __future__ import annotations

import io
import os
import sys

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from PIL import Image, UnidentifiedImageError

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference import REPO_ROOT, Predictor  # noqa: E402

FRONTEND_DIR = os.path.join(REPO_ROOT, "frontend")
ALLOWED = {"png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"}
MAX_UPLOAD_MB = 10

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
CORS(app)  # lets frontend/index.html work when opened straight from disk

predictor: Predictor | None = None
load_error: str | None = None


def resolve_model_path() -> str:
    path = os.environ.get("MODEL_PATH", os.path.join("models", "effnet_tuned.keras"))
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def load_model() -> None:
    global predictor, load_error
    path = resolve_model_path()
    if not os.path.exists(path):
        load_error = f"model file not found: {path}"
        print(f"[!] {load_error}")
        return
    print(f"Loading {path} ...")
    try:
        predictor = Predictor(path, preprocess=os.environ.get("MODEL_PREPROCESS") or None)
    except Exception as exc:  # noqa: BLE001 - report it on /health instead of crashing
        load_error = f"{type(exc).__name__}: {exc}"
        print(f"[!] could not load model: {load_error}")
        return
    print(f"Model ready: {predictor.name} (preprocess={predictor.preprocess})")


@app.get("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.get("/health")
def health():
    return jsonify({
        "status": "ok" if predictor else "error",
        "model_loaded": predictor is not None,
        "model": predictor.info() if predictor else None,
        "error": load_error,
    }), 200 if predictor else 503


@app.post("/predict")
def predict():
    if predictor is None:
        return jsonify({"error": f"Model not loaded ({load_error})."}), 503

    file = request.files.get("file")
    if file is None:
        return jsonify({"error": 'Send multipart/form-data with the image in field "file".'}), 400
    if not file.filename:
        return jsonify({"error": "Empty filename."}), 400
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED:
        return jsonify({"error": f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED))}."}), 415

    try:
        img = Image.open(io.BytesIO(file.read()))
        img.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        return jsonify({"error": "The file could not be read as an image."}), 400

    with_gradcam = request.args.get("gradcam", "1") != "0"
    try:
        result = predictor.analyse(img, with_gradcam=with_gradcam)
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("prediction failed")
        return jsonify({"error": f"Prediction failed: {type(exc).__name__}"}), 500

    result["model"] = predictor.name
    return jsonify(result)


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": f"File too large (limit {MAX_UPLOAD_MB} MB)."}), 413


if __name__ == "__main__":
    load_model()
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 5000)),
            debug=False)
