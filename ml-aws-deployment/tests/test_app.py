import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_model  # noqa: E402


@pytest.fixture(scope="session")
def module():
    if not train_model.MODEL_PATH.exists():
        train_model.main()
    import app as app_module
    app_module.bundle = app_module.load_bundle()
    return app_module


@pytest.fixture()
def client(module):
    return module.app.test_client()


SAMPLES = {
    "Iris Setosa": [5.1, 3.5, 1.4, 0.2],
    "Iris Versicolor": [6.0, 2.9, 4.5, 1.5],
    "Iris Virginica": [6.5, 3.0, 5.5, 1.8],
}


def payload(v):
    return dict(zip(["sepal_length", "sepal_width", "petal_length", "petal_width"], v))


def test_model_loads(module):
    assert module.bundle is not None
    assert module.bundle["accuracy"] > 0.8


def test_home_page(client):
    r = client.get("/")
    assert r.status_code == 200 and b"Cloud ML Predictor" in r.data


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.get_json()["status"] == "healthy" and r.get_json()["model_loaded"] is True


def test_info(client):
    d = client.get("/api/info").get_json()
    assert d["model"] == "Logistic Regression" and 0 < d["accuracy"] <= 1


@pytest.mark.parametrize("label,values", SAMPLES.items())
def test_predict(client, label, values):
    d = client.post("/predict", json=payload(values)).get_json()
    assert d["success"] is True and d["prediction"] == label
    assert 0 < d["confidence"] <= 1


@pytest.mark.parametrize("bad", [
    {}, {"sepal_length": 5.1},
    payload(["abc", 3.5, 1.4, 0.2]),
    payload([-1, 3.5, 1.4, 0.2]),
    payload([0, 3.5, 1.4, 0.2]),
    payload([5.1, 3.5, 1.4, 999]),
    payload([None, 3.5, 1.4, 0.2]),
])
def test_invalid_input_rejected(client, bad):
    r = client.post("/predict", json=bad)
    assert r.status_code == 400 and r.get_json()["success"] is False


def test_non_json_rejected(client):
    assert client.post("/predict", data="nope").status_code == 400
