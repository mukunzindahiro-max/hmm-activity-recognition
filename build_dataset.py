from pathlib import Path
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
EXPORT_DIR = ROOT / "data" / "sensor_logger_exports"
RAW_DIR = ROOT / "data" / "raw"
TEST_DIR = ROOT / "data" / "test"

ACTIVITIES = ["standing", "walking", "jumping", "still"]
PARTICIPANT = "james"
MAX_CLIP_S = 10.0

SENSORS = {
    "Accelerometer.csv": "acc",
    "Gyroscope.csv": "gyr",
    "Gravity.csv": "grav",
}

TEST_SESSIONS = {
    "test-1-data-2026-10-28_08-54-48": (
        "mixed_01",
        [
            (0.00, 10.00, "standing"),
            (10.00, 21.25, "walking"),
            (21.25, 32.40, "jumping"),
            (32.40, np.inf, "still"),
        ],
    ),
    "testing-2-data-2026-10-28_08-56-40": (
        "mixed_02",
        [
            (0.00, 11.75, "walking"),
            (11.75, 21.00, "still"),
            (21.00, 31.00, "jumping"),
            (31.00, np.inf, "standing"),
        ],
    ),
}

FOLDER_PATTERN = re.compile(r"^(?P<activity>[a-z]+)-(?P<index>\d+)-data-")


def read_metadata(folder: Path) -> dict:
    meta = pd.read_csv(folder / "Metadata.csv").iloc[0]
    rates_ms = [float(r) for r in str(meta["sampleRateMs"]).split("|") if r]
    return {"device": meta["device name"], "fs": round(1000.0 / min(rates_ms))}


def merge_sensors(folder: Path) -> pd.DataFrame:
    merged = None
    for filename, prefix in SENSORS.items():
        df = pd.read_csv(folder / filename)
        df = df.rename(columns={axis: f"{prefix}_{axis}" for axis in "xyz"})
        cols = ["time", "seconds_elapsed"] + [f"{prefix}_{a}" for a in "xyz"]
        df = df[cols]
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df.drop(columns="seconds_elapsed"), on="time", how="inner")

    merged = merged.dropna().drop_duplicates(subset="time")
    merged = merged.sort_values("time").reset_index(drop=True)
    return merged


def trim_to_max_duration(df: pd.DataFrame, max_s: float = MAX_CLIP_S) -> pd.DataFrame:
    seconds = (df["time"] - df["time"].iloc[0]) / 1e9
    excess = seconds.iloc[-1] - max_s
    if excess <= 0:
        return df
    start = excess / 2
    keep = (seconds >= start) & (seconds <= start + max_s)
    return df[keep].reset_index(drop=True)


def finalise(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    df = df.copy()
    df["seconds_elapsed"] = (df["time"] - df["time"].iloc[0]) / 1e9
    df["fs"] = meta["fs"]
    df["participant"] = PARTICIPANT
    df["device"] = meta["device"]
    ordered = (
        ["time", "seconds_elapsed"]
        + [f"{p}_{a}" for p in SENSORS.values() for a in "xyz"]
        + ["fs", "activity", "participant", "device"]
    )
    return df[ordered]


def build_training_clips() -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    for folder in sorted((EXPORT_DIR / "train").iterdir()):
        match = FOLDER_PATTERN.match(folder.name)
        if not folder.is_dir() or match is None:
            continue
        activity = match["activity"]
        if activity not in ACTIVITIES:
            raise ValueError(f"Unknown activity in folder name: {folder.name}")

        meta = read_metadata(folder)
        df = trim_to_max_duration(merge_sensors(folder))
        df["activity"] = activity
        df = finalise(df, meta)

        name = f"{activity}_{int(match['index']):02d}.csv"
        df.to_csv(RAW_DIR / name, index=False)
        summary.append({
            "file": name,
            "activity": activity,
            "samples": len(df),
            "duration_s": round(df["seconds_elapsed"].iloc[-1], 2),
            "fs": meta["fs"],
            "device": meta["device"],
            "source_folder": folder.name,
        })
    return pd.DataFrame(summary).sort_values("file").reset_index(drop=True)


def label_segments(seconds: pd.Series, segments: list) -> pd.Series:
    labels = pd.Series(index=seconds.index, dtype="object")
    for start, end, activity in segments:
        labels[(seconds >= start) & (seconds < end)] = activity
    return labels


def build_test_sessions() -> pd.DataFrame:
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    for folder_name, (out_name, segments) in TEST_SESSIONS.items():
        folder = EXPORT_DIR / "test" / folder_name
        meta = read_metadata(folder)
        df = merge_sensors(folder)
        df["activity"] = label_segments(df["seconds_elapsed"], segments)
        df = df.dropna(subset=["activity"])
        df = finalise(df, meta)

        df.to_csv(TEST_DIR / f"{out_name}.csv", index=False)
        summary.append({
            "file": f"{out_name}.csv",
            "activity": " -> ".join(s[2] for s in segments),
            "samples": len(df),
            "duration_s": round(df["seconds_elapsed"].iloc[-1], 2),
            "fs": meta["fs"],
            "device": meta["device"],
            "source_folder": folder_name,
        })
    return pd.DataFrame(summary)


def main() -> None:
    train = build_training_clips()
    test = build_test_sessions()

    print(f"Training clips written to {RAW_DIR.relative_to(ROOT)}: {len(train)}")
    totals = train.groupby("activity")["duration_s"].agg(["count", "sum", "min", "max"])
    print(totals.round(2).to_string())
    print(f"\nTest sessions written to {TEST_DIR.relative_to(ROOT)}:")
    print(test[["file", "activity", "samples", "duration_s"]].to_string(index=False))


if __name__ == "__main__":
    main()
