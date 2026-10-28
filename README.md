# Human Activity Recognition with Hidden Markov Models

Formative 2: recognising **standing, walking, jumping and still** from smartphone
accelerometer and gyroscope data, using a Hidden Markov Model implemented from
scratch in NumPy.

## Data collection

| Item | Value |
|---|---|
| App | Sensor Logger 1.68 (iOS) |
| Device | iPhone 13 (iOS 26.5) |
| Sensors | Accelerometer (x, y, z), Gyroscope (x, y, z), Gravity (x, y, z) |
| Sampling rate | 100 Hz (10 ms interval) |
| Training clips | 80 (20 per activity, 8.7 – 10.0 s each) |
| Test sessions | 2 continuous sessions of about 40 s (all four activities) |

Recording time per activity (training):

| Activity | Clips | Total duration |
|---|---|---|
| Standing | 20 | 193.7 s |
| Walking | 20 | 199.7 s |
| Jumping | 20 | 186.8 s |
| Still | 20 | 199.7 s |

Clips longer than 10 s are trimmed equally from both ends, which also removes the
phone-tap at the start and end of each recording.

## Repository structure

```
data/
  raw/                     cleaned, labelled training clips (<activity>_<NN>.csv)
  test/                    unseen continuous test sessions (mixed_<NN>.csv)
  dataset_summary.csv      samples, duration and sampling rate of every file
build_dataset.py           converts the Sensor Logger exports into the clean CSV files
requirements.txt
```

## CSV format

| Column | Description |
|---|---|
| `time` | Unix timestamp (ns) from the phone |
| `seconds_elapsed` | seconds since the first sample |
| `acc_x`, `acc_y`, `acc_z` | user acceleration, gravity removed (m/s²) |
| `gyr_x`, `gyr_y`, `gyr_z` | angular velocity (rad/s) |
| `grav_x`, `grav_y`, `grav_z` | gravity vector (m/s²) |
| `fs` | device sampling rate (Hz) |
| `activity` | ground-truth label |
| `participant`, `device` | who recorded the clip and on which phone |

The test sessions follow these activity orders:

- `mixed_01.csv`: standing → walking → jumping → still
- `mixed_02.csv`: walking → still → jumping → standing

## Setup

```bash
pip install -r requirements.txt
```
