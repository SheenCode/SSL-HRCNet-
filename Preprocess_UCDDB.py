import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

import biosppy.signals.tools as st
import numpy as np
from biosppy.signals.ecg import correct_rpeaks, hamilton_segmenter
from joblib import cpu_count
from scipy.signal import medfilt
from tqdm import tqdm


# =========================
# Configuration
# =========================
FS = 128
SAMPLES_PER_MIN = FS * 60

BEFORE = 2   # minutes before the target minute
AFTER = 2    # minutes after the target minute
HR_MIN = 20
HR_MAX = 300
MIN_OVERLAP_SECONDS = 5

DATA_DIR = "UCDDB"
LABEL_SUFFIX = "_respevt.txt"

FILENAME_SET = [
    "ucddb002", "ucddb003", "ucddb005", "ucddb007", "ucddb008", "ucddb009",
    "ucddb010", "ucddb011", "ucddb012", "ucddb013", "ucddb014", "ucddb015",
    "ucddb017", "ucddb018", "ucddb019", "ucddb020", "ucddb021", "ucddb022",
    "ucddb028", "ucddb023", "ucddb024", "ucddb025", "ucddb026", "ucddb006",
    "ucddb027"
]

def time_to_seconds(time_str: str) -> int:
    """Convert time string in HH:MM:SS format to seconds."""
    hours, minutes, seconds = map(int, time_str.split(":"))
    return hours * 3600 + minutes * 60 + seconds


def load_labels(subject_id: str):
    """
    Load respiratory event annotations for one subject.

    Returns:
        event_starts: np.ndarray of event start times in seconds
        event_durations: np.ndarray of event durations in seconds
    """
    label_path = os.path.join(DATA_DIR, "files", f"{subject_id}{LABEL_SUFFIX}")

    event_starts = []
    event_durations = []

    with open(label_path, "r") as f:
        lines = f.readlines()

    # Skip the header lines
    for line in lines[3:]:
        columns = line.split()
        if len(columns) < 3:
            continue

        # columns[0]: start time, columns[2]: duration or tag
        if columns[2] == "EVENT":
            continue

        event_starts.append(time_to_seconds(columns[0]))
        event_durations.append(int(columns[2]))

    return np.array(event_starts), np.array(event_durations)


def intervals_overlap(interval1, interval2, min_overlap=5):
    """
    Check whether two intervals overlap for at least `min_overlap` seconds.

    Args:
        interval1: [start1, end1]
        interval2: [start2, end2]
        min_overlap: minimum overlap duration in seconds

    Returns:
        bool
    """
    start_max = max(interval1[0], interval2[0])
    end_min = min(interval1[1], interval2[1])
    return end_min > start_max and (end_min - start_max) >= min_overlap


def extract_rri_and_rpeak(signal: np.ndarray, fs: int):
    """
    Extract RRI and R-peak amplitude signals from ECG.

    Returns:
        ((rri_tm, rri_signal), (ampl_tm, ampl_signal)) or None
    """
    signal, _, _ = st.filter_signal(
        signal,
        ftype="FIR",
        band="bandpass",
        order=int(0.3 * fs),
        frequency=[3, 45],
        sampling_rate=fs,
    )

    # Detect and correct R-peaks
    rpeaks, = hamilton_segmenter(signal, sampling_rate=fs)
    rpeaks, = correct_rpeaks(signal, rpeaks=rpeaks, sampling_rate=fs, tol=0.1)

    avg_beats_per_min = len(rpeaks) / (1 + BEFORE + AFTER)
    if avg_beats_per_min < 40 or avg_beats_per_min > 200:
        return None

    # Extract RRI and R-peak amplitude
    rri_tm = rpeaks[1:] / float(fs)
    rri_signal = np.diff(rpeaks) / float(fs)
    rri_signal = medfilt(rri_signal, kernel_size=3)

    ampl_tm = rpeaks / float(fs)
    ampl_signal = signal[rpeaks]

    # Physiological HR constraint
    hr = 60 / rri_signal
    if not np.all((hr >= HR_MIN) & (hr <= HR_MAX)):
        return None

    return (rri_tm, rri_signal), (ampl_tm, ampl_signal)


def assign_label(window_start_sec, window_end_sec, event_starts, event_durations, min_overlap=5):
    """
    Assign binary label to one window:
    1 if overlap with any respiratory event is >= min_overlap seconds, else 0.
    """
    sample_interval = [window_start_sec, window_end_sec]

    for start, duration in zip(event_starts, event_durations):
        event_interval = [start, start + duration]
        if intervals_overlap(event_interval, sample_interval, min_overlap):
            return 1
    return 0


# =========================
# Worker
# =========================
def worker(subject_id: str):
    """
    Process one UCDDB subject.

    Returns:
        X: list of samples
        y: list of binary labels
    """
    signal_path = os.path.join(DATA_DIR, f"{subject_id}.txt")
    ecg = np.loadtxt(signal_path)

    event_starts, event_durations = load_labels(subject_id)

    X = []
    y = []

    total_minutes = int(len(ecg) / SAMPLES_PER_MIN)

    for minute_idx in tqdm(range(total_minutes), desc=subject_id):
        if minute_idx < BEFORE or (minute_idx + 1 + AFTER) > total_minutes:
            continue

        start_idx = int((minute_idx - BEFORE) * SAMPLES_PER_MIN)
        end_idx = int((minute_idx + 1 + AFTER) * SAMPLES_PER_MIN)
        segment = ecg[start_idx:end_idx]

        features = extract_rri_and_rpeak(segment, FS)
        if features is None:
            continue

        window_start_sec = int((minute_idx - BEFORE) * 60)
        window_end_sec = int((minute_idx + 1 + AFTER) * 60)

        label = assign_label(
            window_start_sec,
            window_end_sec,
            event_starts,
            event_durations,
            min_overlap=MIN_OVERLAP_SECONDS,
        )

        X.append(features)
        y.append(label)

    return X, y

if __name__ == "__main__":
    num_worker = min(30, max(cpu_count() - 1, 1))

    all_X = []
    all_y = []

    with ProcessPoolExecutor(max_workers=num_worker) as executor:
        futures = [executor.submit(worker, subject_id) for subject_id in FILENAME_SET]

        for future in as_completed(futures):
            X, y = future.result()
            all_X.extend(X)
            all_y.extend(y)

    ucddb_data = {
        "X_train": all_X,
        "y_train": all_y,
    }

    with open("UCDDB.pkl", "wb") as f:
        pickle.dump(ucddb_data, f, protocol=2)

    print("\nok!")
