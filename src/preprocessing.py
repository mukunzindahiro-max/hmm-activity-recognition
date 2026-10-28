from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
TEST_DIR = ROOT / "data" / "test"

ACTIVITIES = ["standing", "walking", "jumping", "still"]
SIGNAL_COLUMNS = [f"{p}_{a}" for p in ("acc", "gyr", "grav") for a in "xyz"]

TARGET_FS = 100
CUTOFF_HZ = 20.0
FILTER_ORDER = 4
EDGE_TRIM_S = 0.5
WINDOW_S = 2.0
OVERLAP = 0.5

WINDOW_SIZE = int(WINDOW_S * TARGET_FS)
STEP_SIZE = int(WINDOW_SIZE * (1 - OVERLAP))


def load_recording(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.sort_values("time").drop_duplicates(subset="time").reset_index(drop=True)
    df["seconds_elapsed"] = (df["time"] - df["time"].iloc[0]) / 1e9
    return df


def resample_uniform(df: pd.DataFrame, target_fs: int = TARGET_FS) -> pd.DataFrame:
    t = df["seconds_elapsed"].to_numpy()
    grid = np.arange(0.0, t[-1], 1.0 / target_fs)
    out = pd.DataFrame({"seconds_elapsed": grid})
    for col in SIGNAL_COLUMNS:
        out[col] = np.interp(grid, t, df[col].to_numpy())

    nearest = np.clip(np.searchsorted(t, grid), 0, len(t) - 1)
    out["activity"] = df["activity"].to_numpy()[nearest]
    out["fs"] = target_fs
    return out


def lowpass_filter(df: pd.DataFrame, fs: int = TARGET_FS,
                   cutoff: float = CUTOFF_HZ, order: int = FILTER_ORDER) -> pd.DataFrame:
    b, a = butter(order, cutoff / (fs / 2), btype="low")
    out = df.copy()
    for col in SIGNAL_COLUMNS:
        out[col] = filtfilt(b, a, df[col].to_numpy())
    return out


def trim_edges(df: pd.DataFrame, trim_s: float = EDGE_TRIM_S) -> pd.DataFrame:
    t = df["seconds_elapsed"]
    keep = (t >= trim_s) & (t <= t.iloc[-1] - trim_s)
    out = df[keep].reset_index(drop=True)
    out["seconds_elapsed"] = out["seconds_elapsed"] - out["seconds_elapsed"].iloc[0]
    return out


def preprocess(path: Path, trim: bool = True) -> pd.DataFrame:
    df = resample_uniform(load_recording(path))
    df = lowpass_filter(df)
    return trim_edges(df) if trim else df


def make_windows(df: pd.DataFrame, window_size: int = WINDOW_SIZE,
                 step_size: int = STEP_SIZE) -> tuple:
    signals = df[SIGNAL_COLUMNS].to_numpy()
    labels = df["activity"].to_numpy()
    seconds = df["seconds_elapsed"].to_numpy()

    windows, window_labels, starts = [], [], []
    for start in range(0, len(df) - window_size + 1, step_size):
        segment = labels[start:start + window_size]
        values, counts = np.unique(segment, return_counts=True)
        windows.append(signals[start:start + window_size])
        window_labels.append(values[np.argmax(counts)])
        starts.append(seconds[start])

    if not windows:
        return np.empty((0, window_size, len(SIGNAL_COLUMNS))), np.array([]), np.array([])
    return np.stack(windows), np.array(window_labels), np.array(starts)


def load_windows(directory: Path, pattern: str, trim: bool) -> list:
    sequences = []
    for path in sorted(directory.glob(pattern)):
        df = preprocess(path, trim=trim)
        windows, labels, starts = make_windows(df)
        sequences.append({
            "file": path.name,
            "signal": df,
            "windows": windows,
            "labels": labels,
            "start_s": starts,
        })
    return sequences


def load_training_windows() -> list:
    return load_windows(RAW_DIR, "*.csv", trim=True)


def load_test_windows() -> list:
    return load_windows(TEST_DIR, "mixed_*.csv", trim=False)


if __name__ == "__main__":
    train = load_training_windows()
    test = load_test_windows()

    print(f"Sampling rate: {TARGET_FS} Hz | window: {WINDOW_S} s = {WINDOW_SIZE} samples "
          f"| step: {STEP_SIZE} samples ({int(OVERLAP * 100)}% overlap)")

    counts = pd.Series(np.concatenate([s["labels"] for s in train])).value_counts()
    print(f"\nTraining: {len(train)} clips, {counts.sum()} windows")
    print(counts.reindex(ACTIVITIES).to_string())

    for seq in test:
        per_label = pd.Series(seq["labels"]).value_counts().reindex(ACTIVITIES).fillna(0).astype(int)
        print(f"\nTest {seq['file']}: {len(seq['labels'])} windows")
        print(per_label.to_string())
