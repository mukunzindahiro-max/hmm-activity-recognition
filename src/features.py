import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import preprocessing as pp

PROCESSED_DIR = pp.ROOT / "data" / "processed"
EPS = 1e-4
LOCOMOTION_BAND = (0.5, 3.5)

FEATURE_DESCRIPTIONS = {
    "acc_mag_mean": ("time", "average intensity of body acceleration"),
    "acc_mag_std": ("time", "variability of acceleration; near zero when still, very high when jumping"),
    "acc_rms": ("time", "energy of the acceleration signal in the time domain"),
    "acc_sma": ("time", "signal magnitude area, total movement across the three axes"),
    "acc_ptp": ("time", "peak-to-peak range; captures landing impacts of jumps"),
    "gyr_mag_std": ("time", "variability of rotation; separates a hand-held phone (standing) from a phone on a table (still)"),
    "gyr_sma": ("time", "total rotation across the three axes; walking and jumping swing the phone"),
    "acc_corr_yz": ("time", "correlation between vertical and forward axes; coordinated during gait, random when idle"),
    "tilt_deg": ("time", "phone tilt from horizontal computed from gravity; flat on a table when still, upright when held"),
    "vert_dom_freq": ("frequency", "dominant FFT frequency of vertical acceleration; step rate for walking, jump rate for jumping"),
    "vert_spectral_energy": ("frequency", "total FFT energy of vertical acceleration; orders of magnitude apart between activities"),
    "vert_spectral_entropy": ("frequency", "flatness of the spectrum; low for periodic motion, high for noise"),
    "vert_band_ratio": ("frequency", "share of FFT energy in the 0.5-3.5 Hz locomotion band"),
    "gyr_dom_freq": ("frequency", "dominant FFT frequency of rotation; confirms the movement rhythm"),
}
FEATURE_NAMES = list(FEATURE_DESCRIPTIONS)

LOG_FEATURES = [
    "acc_mag_mean", "acc_mag_std", "acc_rms", "acc_sma", "acc_ptp",
    "gyr_mag_std", "gyr_sma", "vert_spectral_energy",
]

ACC = slice(0, 3)
GYR = slice(3, 6)
GRAV = slice(6, 9)


def magnitude(xyz: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(xyz ** 2, axis=1))


def spectrum(signal: np.ndarray, fs: int) -> tuple:
    centred = (signal - signal.mean()) * np.hanning(len(signal))
    power = np.abs(np.fft.rfft(centred)) ** 2 / len(signal)
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / fs)
    return freqs[1:], power[1:]


def dominant_frequency(freqs: np.ndarray, power: np.ndarray) -> float:
    return float(freqs[np.argmax(power)])


def spectral_entropy(power: np.ndarray) -> float:
    p = power / (power.sum() + EPS)
    return float(-np.sum(p * np.log2(p + 1e-12)) / np.log2(len(p)))


def band_ratio(freqs: np.ndarray, power: np.ndarray, band: tuple = LOCOMOTION_BAND) -> float:
    in_band = (freqs >= band[0]) & (freqs <= band[1])
    return float(power[in_band].sum() / (power.sum() + EPS))


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def window_features(window: np.ndarray, fs: int = pp.TARGET_FS) -> dict:
    acc, gyr, grav = window[:, ACC], window[:, GYR], window[:, GRAV]
    acc_mag, gyr_mag = magnitude(acc), magnitude(gyr)

    g = grav.mean(axis=0)
    g_unit = g / (np.linalg.norm(g) + EPS)
    vertical = acc @ g_unit
    tilt = np.degrees(np.arccos(np.clip(abs(g_unit[2]), 0, 1)))

    vert_freqs, vert_power = spectrum(vertical, fs)
    gyr_freqs, gyr_power = spectrum(gyr_mag, fs)

    return {
        "acc_mag_mean": acc_mag.mean(),
        "acc_mag_std": acc_mag.std(),
        "acc_rms": np.sqrt(np.mean(acc_mag ** 2)),
        "acc_sma": np.mean(np.sum(np.abs(acc), axis=1)),
        "acc_ptp": np.ptp(acc_mag),
        "gyr_mag_std": gyr_mag.std(),
        "gyr_sma": np.mean(np.sum(np.abs(gyr), axis=1)),
        "acc_corr_yz": safe_corr(acc[:, 1], acc[:, 2]),
        "tilt_deg": tilt,
        "vert_dom_freq": dominant_frequency(vert_freqs, vert_power),
        "vert_spectral_energy": vert_power.sum(),
        "vert_spectral_entropy": spectral_entropy(vert_power),
        "vert_band_ratio": band_ratio(vert_freqs, vert_power),
        "gyr_dom_freq": dominant_frequency(gyr_freqs, gyr_power),
    }


def extract_features(sequences: list) -> pd.DataFrame:
    rows = []
    for seq in sequences:
        for i, (window, label, start) in enumerate(zip(seq["windows"], seq["labels"], seq["start_s"])):
            rows.append({"file": seq["file"], "window": i, "start_s": start,
                         "activity": label, **window_features(window)})
    return pd.DataFrame(rows)


def log_transform(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in LOG_FEATURES:
        out[col] = np.log10(out[col] + EPS)
    return out


class ZScoreNormalizer:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, df: pd.DataFrame) -> "ZScoreNormalizer":
        self.mean_ = df[FEATURE_NAMES].mean()
        self.std_ = df[FEATURE_NAMES].std(ddof=0).replace(0, 1.0)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[FEATURE_NAMES] = (df[FEATURE_NAMES] - self.mean_) / self.std_
        return out

    def to_dict(self) -> dict:
        return {"features": FEATURE_NAMES, "log_features": LOG_FEATURES,
                "mean": self.mean_.to_dict(), "std": self.std_.to_dict()}

    @classmethod
    def from_dict(cls, params: dict) -> "ZScoreNormalizer":
        norm = cls()
        norm.mean_ = pd.Series(params["mean"])[FEATURE_NAMES]
        norm.std_ = pd.Series(params["std"])[FEATURE_NAMES]
        return norm


def build_feature_sets(save: bool = True) -> tuple:
    train_raw = extract_features(pp.load_training_windows())
    test_raw = extract_features(pp.load_test_windows())

    train_log, test_log = log_transform(train_raw), log_transform(test_raw)
    normalizer = ZScoreNormalizer().fit(train_log)
    train, test = normalizer.transform(train_log), normalizer.transform(test_log)

    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        train.to_csv(PROCESSED_DIR / "features_train.csv", index=False)
        test.to_csv(PROCESSED_DIR / "features_test.csv", index=False)
        with open(PROCESSED_DIR / "normalization_stats.json", "w") as f:
            json.dump(normalizer.to_dict(), f, indent=2)

    return train, test, train_raw, test_raw, normalizer


if __name__ == "__main__":
    train, test, train_raw, _, _ = build_feature_sets()
    print(f"Features: {len(FEATURE_NAMES)} "
          f"({sum(v[0] == 'time' for v in FEATURE_DESCRIPTIONS.values())} time-domain, "
          f"{sum(v[0] == 'frequency' for v in FEATURE_DESCRIPTIONS.values())} frequency-domain)")
    print(f"Train windows: {len(train)} | test windows: {len(test)}")
    print("\nMean raw feature value per activity:")
    print(train_raw.groupby("activity")[FEATURE_NAMES].mean().reindex(pp.ACTIVITIES).T.round(3).to_string())
