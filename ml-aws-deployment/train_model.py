"""Train the Iris classifier once and save it to model.pkl."""
from pathlib import Path

import joblib
import sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"


def main(path=MODEL_PATH):
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
    )
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    accuracy = float(accuracy_score(y_test, model.predict(X_test)))

    bundle = {
        "model": model,
        "accuracy": accuracy,
        "class_names": ["Iris " + n.capitalize() for n in iris.target_names],
        "n_train": len(X_train),
        "n_test": len(X_test),
        "sklearn_version": sklearn.__version__,
    }
    joblib.dump(bundle, path)
    print(f"Test accuracy: {accuracy:.4f} ({len(X_test)} test samples)")
    print(f"Model saved to {path}")
    return bundle


if __name__ == "__main__":
    main()
