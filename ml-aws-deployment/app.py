"""Flask REST API + web interface for the Iris ML model."""
import math
import os
import urllib.request
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", BASE_DIR / "model.pkl"))
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
MAX_VALUE = 20.0
BAD_INPUT = "Please enter valid numerical values for all four features."

app = Flask(__name__)
CORS(app)


def load_bundle(path=MODEL_PATH):
    """Load the trained model bundle; return None if it is missing or broken."""
    try:
        return joblib.load(path)
    except Exception as exc:  # missing file, version mismatch, corrupt pickle
        app.logger.error("Could not load model from %s: %s", path, exc)
        return None


bundle = load_bundle()  # loaded once at startup, reused for every request
_cloud_cache = {}


def detect_ec2():
    """True only if the AWS EC2 instance metadata service (IMDSv2) answers."""
    if "ec2" not in _cloud_cache:
        try:
            req = urllib.request.Request(
                "http://169.254.169.254/latest/api/token", method="PUT",
                headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"})
            _cloud_cache["ec2"] = bool(urllib.request.urlopen(req, timeout=0.5).read())
        except Exception:
            _cloud_cache["ec2"] = False
    return _cloud_cache["ec2"]


def parse_features(data):
    """Return (list_of_floats, None) or (None, error_message)."""
    if not isinstance(data, dict):
        return None, "Request body must be a JSON object."
    values = []
    for name in FEATURES:
        raw = data.get(name)
        if raw is None or raw == "" or isinstance(raw, bool):
            return None, BAD_INPUT
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None, BAD_INPUT
        if not math.isfinite(value):
            return None, BAD_INPUT
        if value < 0:
            return None, "Negative values are not allowed."
        if value == 0 or value > MAX_VALUE:
            return None, f"Each value must be greater than 0 and at most {MAX_VALUE:g} cm."
        values.append(value)
    return values, None


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/predict")
def predict():
    if bundle is None:
        return jsonify(success=False, error="Model is not loaded. Run train_model.py first."), 503
    values, error = parse_features(request.get_json(silent=True))
    if error:
        return jsonify(success=False, error=error), 400
    try:
        model, names = bundle["model"], bundle["class_names"]
        probs = model.predict_proba(np.array([values]))[0]
        best = int(np.argmax(probs))
        return jsonify(
            success=True,
            prediction=names[best],
            confidence=round(float(probs[best]), 4),
            probabilities={n: round(float(p), 4) for n, p in zip(names, probs)},
        )
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify(success=False, error="Prediction failed on the server."), 500


@app.get("/health")
def health():
    file_exists = MODEL_PATH.exists()
    loaded = bundle is not None
    ok = file_exists and loaded
    return jsonify(
        status="healthy" if ok else "unhealthy",
        model_loaded=loaded,
        model_file_exists=file_exists,
        in_container=Path("/.dockerenv").exists(),
    ), (200 if ok else 503)


@app.get("/api/info")
def info():
    data = {
        "project": "Iris ML Prediction System",
        "model": "Logistic Regression",
        "framework": "Flask",
        "deployment": "Docker + AWS EC2",
        "dataset": "Iris (scikit-learn)",
        "features": len(FEATURES),
        "classes": 3,
        "in_container": Path("/.dockerenv").exists(),
        "running_on_ec2": detect_ec2(),
    }
    if bundle:
        data.update(accuracy=round(bundle["accuracy"], 4), train_samples=bundle["n_train"],
                    test_samples=bundle["n_test"], sklearn_version=bundle["sklearn_version"])
    return jsonify(data)


@app.errorhandler(404)
def not_found(_):
    return jsonify(success=False, error="Endpoint not found."), 404


@app.errorhandler(405)
def bad_method(_):
    return jsonify(success=False, error="Method not allowed."), 405


@app.errorhandler(500)
def server_error(_):
    return jsonify(success=False, error="Internal server error."), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
