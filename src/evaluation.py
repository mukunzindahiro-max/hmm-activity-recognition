import numpy as np
import pandas as pd

from src.features import PROCESSED_DIR
from src.hmm import GaussianHMM
from src.preprocessing import ACTIVITIES
from src.train import MODEL_PATH, to_sequences

METRICS_PATH = PROCESSED_DIR / "test_metrics.csv"
PREDICTIONS_PATH = PROCESSED_DIR / "test_predictions.csv"


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: list = ACTIVITIES) -> pd.DataFrame:
    index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[index[t], index[p]] += 1
    return pd.DataFrame(matrix, index=pd.Index(labels, name="true"),
                        columns=pd.Index(labels, name="predicted"))


def per_state_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list = ACTIVITIES) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels).to_numpy()
    total = cm.sum()
    rows = []
    for k, label in enumerate(labels):
        tp = cm[k, k]
        fn = cm[k].sum() - tp
        fp = cm[:, k].sum() - tp
        tn = total - tp - fn - fp
        rows.append({
            "State (Activity)": label,
            "Number of Samples": int(cm[k].sum()),
            "Sensitivity": tp / (tp + fn) if tp + fn else np.nan,
            "Specificity": tn / (tn + fp) if tn + fp else np.nan,
            "Precision": tp / (tp + fp) if tp + fp else np.nan,
            "Accuracy (one-vs-rest)": (tp + tn) / total,
        })
    table = pd.DataFrame(rows)
    table["Overall Accuracy"] = np.trace(cm) / total
    return table


def predict_independent(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    return np.array(model.states)[np.argmax(model._log_emission(X), axis=1)]


def decode_test_set(model: GaussianHMM, test_features: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for file, df in test_features.groupby("file", sort=False):
        df = df.sort_values("window").copy()
        X = df.drop(columns=["file", "window", "start_s", "activity"]).to_numpy()
        posterior = model.predict_proba(X)
        df["viterbi"] = model.predict(X)
        df["independent"] = predict_independent(model, X)
        for k, state in enumerate(model.states):
            df[f"p_{state}"] = posterior[:, k]
        frames.append(df[["file", "window", "start_s", "activity", "viterbi", "independent"]
                         + [f"p_{s}" for s in model.states]])
    return pd.concat(frames, ignore_index=True)


def evaluate(model: GaussianHMM, test_features: pd.DataFrame, save: bool = True) -> tuple:
    predictions = decode_test_set(model, test_features)
    metrics = per_state_metrics(predictions["activity"].to_numpy(), predictions["viterbi"].to_numpy())
    if save:
        predictions.to_csv(PREDICTIONS_PATH, index=False)
        metrics.to_csv(METRICS_PATH, index=False)
    return predictions, metrics


if __name__ == "__main__":
    model = GaussianHMM.load(MODEL_PATH)
    test = pd.read_csv(PROCESSED_DIR / "features_test.csv")
    predictions, metrics = evaluate(model, test)

    print(metrics.round(3).to_string(index=False))
    print("\nConfusion matrix (Viterbi, both test files):")
    print(confusion_matrix(predictions["activity"], predictions["viterbi"]).to_string())
    for file, df in predictions.groupby("file"):
        print(f"\n{file}: Viterbi accuracy {np.mean(df['activity'] == df['viterbi']):.3f}, "
              f"independent accuracy {np.mean(df['activity'] == df['independent']):.3f}")
