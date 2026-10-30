import numpy as np
import pandas as pd

from src.features import FEATURE_NAMES, PROCESSED_DIR
from src.hmm import GaussianHMM
from src.preprocessing import ACTIVITIES

MODEL_PATH = PROCESSED_DIR / "hmm_model.json"

HMM_CONFIG = {
    "covariance_type": "diag",
    "tol": 1e-4,
    "max_iter": 200,
    "min_covar": 1e-2,
    "transition_prior": 1.0,
}
CLIPS_PER_SESSION = 4
CHAIN_SEED = 0


def to_sequences(features: pd.DataFrame) -> tuple:
    sequences, labels, files = [], [], []
    for file, df in features.groupby("file", sort=False):
        df = df.sort_values("window")
        sequences.append(df[FEATURE_NAMES].to_numpy())
        labels.append(df["activity"].to_numpy())
        files.append(file)
    return sequences, labels, files


def chain_clips(sequences: list, labels: list, clips_per_session: int = CLIPS_PER_SESSION,
                seed: int = CHAIN_SEED) -> tuple:
    order = np.random.default_rng(seed).permutation(len(sequences))
    sessions = [order[i:i + clips_per_session] for i in range(0, len(order), clips_per_session)]
    chained_X = [np.vstack([sequences[i] for i in s]) for s in sessions]
    chained_y = [np.concatenate([labels[i] for i in s]) for s in sessions]
    return chained_X, chained_y


def initialise(model: GaussianHMM, sequences: list, labels: list) -> GaussianHMM:
    model.fit_supervised(sequences, labels)
    n = model.n_states
    model.startprob_ = np.full(n, 1.0 / n)
    model.transmat_ = np.full((n, n), 1.0 / n)
    pooled = np.cov(np.vstack(sequences), rowvar=False, bias=True)
    model.covars_ = np.array([model._regularise(pooled) for _ in range(n)])
    return model


def train_hmm(train_features: pd.DataFrame, verbose: bool = False, save: bool = True) -> GaussianHMM:
    sequences, labels = chain_clips(*to_sequences(train_features)[:2])
    model = GaussianHMM(ACTIVITIES, verbose=verbose, **HMM_CONFIG)
    initialise(model, sequences, labels)
    model.fit(sequences)
    if save:
        model.save(MODEL_PATH)
    return model


if __name__ == "__main__":
    train = pd.read_csv(PROCESSED_DIR / "features_train.csv")
    model = train_hmm(train, verbose=True)
    print(f"\nConverged: {model.converged_} after {model.n_iter_} iterations (tol = {model.tol})")
    print("\nInitial probabilities:")
    print(pd.Series(model.startprob_, index=ACTIVITIES).round(3).to_string())
    print("\nTransition matrix:")
    print(pd.DataFrame(model.transmat_, index=ACTIVITIES, columns=ACTIVITIES).round(3).to_string())
    print(f"\nSaved to {MODEL_PATH.relative_to(PROCESSED_DIR.parents[1])}")
