import os
import sys
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Any

import numpy as np
import wfdb
import biosppy.signals.tools as st
from biosppy.signals.ecg import hamilton_segmenter, correct_rpeaks
from scipy.signal import medfilt
from joblib import cpu_count
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================
BASE_DIR = "apnea-ecg-database-1.0.0"

FS = 100
SAMPLES_PER_MIN = FS * 60

BEFORE = 2   # minutes before the target segment
AFTER = 2    # minutes after the target segment

HR_MIN = 20
HR_MAX = 300

MAX_WORKERS = 35
NUM_WORKERS = min(MAX_WORKERS, max(1, cpu_count() - 1))

def load_ecg_signal(record_name: str) -> np.ndarray:
    """
    Load the single-lead ECG signal from a record.
    """
    record_path = os.path.join(BASE_DIR, record_name)
    signal = wfdb.rdrecord(record_path, channels=[0]).p_signal[:, 0]
    return signal


def bandpass_filter(signal: np.ndarray, fs: int) -> np.ndarray:
    """
    Apply FIR bandpass filtering to the ECG signal.
    """
    filtered_signal, _, _ = st.filter_signal(
        signal,
        ftype="FIR",
        band="bandpass",
        order=int(0.3 * fs),
        frequency=[3, 45],
        sampling_rate=fs
    )
    return filtered_signal


def detect_rpeaks(signal: np.ndarray, fs: int) -> np.ndarray:
    """
    Detect and correct R peaks in the ECG signal.
    """
    rpeaks, = hamilton_segmenter(signal, sampling_rate=fs)
    rpeaks, = correct_rpeaks(signal, rpeaks=rpeaks, sampling_rate=fs, tol=0.1)
    return rpeaks


def extract_rri_and_amplitude(
    signal: np.ndarray,
    rpeaks: np.ndarray,
    fs: int
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Extract R-R interval (RRI) sequence and R-peak amplitude sequence.
    """
    rri_time = rpeaks[1:] / float(fs)
    rri_signal = np.diff(rpeaks) / float(fs)
    rri_signal = medfilt(rri_signal, kernel_size=3)

    ampl_time = rpeaks / float(fs)
    ampl_signal = signal[rpeaks]

    return (rri_time, rri_signal), (ampl_time, ampl_signal)


def is_valid_rpeak_count(rpeaks: np.ndarray, duration_min: int) -> bool:
    """
    Check whether the detected number of R peaks is within a reasonable range.
    """
    beats_per_min = len(rpeaks) / duration_min
    return 40 <= beats_per_min <= 200


def is_valid_hr(rri_signal: np.ndarray, hr_min: float, hr_max: float) -> bool:
    """
    Check whether all heart rates derived from RRI are physiologically plausible.
    """
    if len(rri_signal) == 0:
        return False

    hr = 60.0 / rri_signal
    return np.all((hr >= hr_min) & (hr <= hr_max))


def process_record(
    record_name: str,
    labels: List[str]
) -> Tuple[List[Any], List[float], List[str]]:
    """
    Process a single ECG record and extract samples.

    Returns:
        X      : list of samples, each sample contains [(rri_time, rri_signal), (ampl_time, ampl_signal)]
        y      : list of binary labels
        groups : list of record names corresponding to each sample
    """
    X, y, groups = [], [], []

    try:
        signals = load_ecg_signal(record_name)
    except Exception as e:
        print(f"[ERROR] Failed to load record {record_name}: {e}", file=sys.stderr)
        return X, y, groups

    total_minutes = len(signals) / float(SAMPLES_PER_MIN)

    for idx in tqdm(range(len(labels)), desc=record_name, file=sys.stdout):
        if idx < BEFORE or (idx + 1 + AFTER) > total_minutes:
            continue

        start = int((idx - BEFORE) * SAMPLES_PER_MIN)
        end = int((idx + 1 + AFTER) * SAMPLES_PER_MIN)
        segment = signals[start:end]

        try:
            segment = bandpass_filter(segment, FS)
            rpeaks = detect_rpeaks(segment, FS)

            if not is_valid_rpeak_count(rpeaks, 1 + BEFORE + AFTER):
                continue

            (rri_time, rri_signal), (ampl_time, ampl_signal) = extract_rri_and_amplitude(
                segment, rpeaks, FS
            )

            if not is_valid_hr(rri_signal, HR_MIN, HR_MAX):
                continue

            X.append([(rri_time, rri_signal), (ampl_time, ampl_signal)])
            y.append(0.0 if labels[idx] == "N" else 1.0)
            groups.append(record_name)

        except Exception as e:
            print(
                f"[WARNING] Failed to process record {record_name}, segment {idx}: {e}",
                file=sys.stderr
            )
            continue

    return X, y, groups


def collect_dataset(
    record_names: List[str],
    label_dict: Dict[str, List[str]],
    desc: str
) -> Tuple[List[Any], List[float], List[str]]:
    """
    Parallel processing for a group of ECG records.
    """
    all_X, all_y, all_groups = [], [], []

    print(f"{desc}...")
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [
            executor.submit(process_record, record_name, label_dict[record_name])
            for record_name in record_names
        ]

        for future in as_completed(futures):
            X, y, groups = future.result()
            all_X.extend(X)
            all_y.extend(y)
            all_groups.extend(groups)

    print()
    return all_X, all_y, all_groups


def load_training_labels(record_names: List[str]) -> Dict[str, List[str]]:
    """
    Load apnea annotations for training records from WFDB annotation files.
    """
    label_dict = {}
    for record_name in record_names:
        ann = wfdb.rdann(os.path.join(BASE_DIR, record_name), extension="apn")
        label_dict[record_name] = ann.symbol
    return label_dict


def load_test_labels(annotation_file: str) -> Dict[str, List[str]]:
    """
    Load test annotations from the provided text file.
    """
    label_dict = {}

    with open(annotation_file, "r") as f:
        content = f.read().strip()

    for block in content.split("\n\n"):
        if not block.strip():
            continue
        record_name = block[:3]
        label_dict[record_name] = list("".join(block.split()[2::2]))

    return label_dict

if __name__ == "__main__":
    train_names = [
        "a01", "a02", "a03", "a04", "a05", "a06", "a07", "a08", "a09", "a10",
        "a11", "a12", "a13", "a14", "a15", "a16", "a17", "a18", "a19", "a20",
        "b01", "b02", "b03", "b04", "b05",
        "c01", "c02", "c03", "c04", "c05", "c06", "c07", "c08", "c09", "c10"
    ]

    test_names = [
        "x01", "x02", "x03", "x04", "x05", "x06", "x07", "x08", "x09", "x10",
        "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19", "x20",
        "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "x29", "x30",
        "x31", "x32", "x33", "x34", "x35"
    ]

    train_label_dict = load_training_labels(train_names)
    test_label_dict = load_test_labels(os.path.join(BASE_DIR, "test-dataset-annos.txt"))

    o_train, y_train, groups_train = collect_dataset(train_names, train_label_dict, desc="Training")
    o_test, y_test, groups_test = collect_dataset(test_names, test_label_dict, desc="Testing")

    apnea_ecg = {
        "o_train": o_train,
        "y_train": y_train,
        "groups_train": groups_train,
        "o_test": o_test,
        "y_test": y_test,
        "groups_test": groups_test
    }

    output_path = os.path.join(BASE_DIR, "apnea-ecg.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(apnea_ecg, f, protocol=2)

    print(f"\nSaved processed dataset to: {output_path}")
    print("ok!")
