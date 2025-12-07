import os
import numpy as np
import pandas as pd
import wfdb
import neurokit2 as nk
from scipy import signal as scipy_signal
from sklearn.model_selection import KFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    accuracy_score
)
from sklearn.metrics import classification_report, roc_curve, confusion_matrix, precision_score, recall_score, f1_score
import zipfile
from datetime import datetime


def load_emg_data(data_root='data', gesture=1, include_indices=None):
    """Load single gesture EMG data from PhysioNet."""
    if include_indices is None:
        include_indices = [i for i in range(32) if i not in [16, 23, 24, 31]]

    metadata_records = []
    signal_arrays = []
    channel_names = None

    for root, _, files in os.walk(data_root):
        for file in files:
            if file.endswith(".hea"):
                record_name = os.path.splitext(file)[0]
                full_path = os.path.join(root, record_name)

                try:
                    parts = record_name.split('_')
                    gesture_num = int(parts[2].replace('gesture', ''))

                    if gesture_num != gesture:
                        continue

                    record = wfdb.rdrecord(full_path)
                    emg_signal = record.p_signal

                    if channel_names is None:
                        channel_names = [record.sig_name[i].strip()
                                       for i in range(len(record.sig_name))
                                       if i in include_indices]

                    kept_channels = [i for i in range(emg_signal.shape[1])
                                   if i in include_indices]
                    filtered_signal = emg_signal[:, kept_channels]

                    metadata_records.append({
                        'id': record_name,
                        'session': int(parts[0].replace('session', '')),
                        'subject': int(parts[1].replace('participant', '')),
                        'gesture': gesture_num,
                        'trial': int(parts[3].replace('trial', '')),
                        'signal_idx': len(signal_arrays),
                        'n_samples': filtered_signal.shape[0],
                        'fs': record.fs
                    })

                    signal_arrays.append(filtered_signal)

                except Exception as e:
                    print(f"Skipping {record_name}: {e}")
                    continue

    metadata_df = pd.DataFrame(metadata_records)
    metadata_df = metadata_df.sort_values(
        by=['session', 'subject', 'gesture', 'trial']
    ).reset_index(drop=True)
    metadata_df['signal_idx'] = range(len(metadata_df))

    signals = np.stack(signal_arrays, axis=0)

    print(f"Loaded {len(metadata_df)} trials for gesture {gesture}")
    print(f"Shape: {signals.shape}")
    print(f"Subjects: {sorted(metadata_df['subject'].unique())}")

    return metadata_df, signals, channel_names


def clean_emg_signal(signal, fs=2000, lowcut=20, highcut=450, notch_freq=60):
    """Clean single trial EMG signal using NeuroKit2."""
    cleaned_channels = []

    for ch_idx in range(signal.shape[1]):
        ch_signal = signal[:, ch_idx]
        cleaned = nk.emg_clean(ch_signal, sampling_rate=fs)
        cleaned_channels.append(cleaned)

    return np.column_stack(cleaned_channels)


def clean_all_trials(signals, fs=2000):
    """Clean all trials in dataset."""
    cleaned_signals = []

    print(f"Cleaning {signals.shape[0]} trials...")
    for i in range(signals.shape[0]):
        cleaned = clean_emg_signal(signals[i], fs=fs)
        cleaned_signals.append(cleaned)
        if (i + 1) % 50 == 0:
            print(f"  Cleaned {i+1}/{signals.shape[0]}")

    return np.stack(cleaned_signals, axis=0)


def extract_time_domain_features(signal):
    """Extract time-domain features from single trial."""
    features = []

    for ch_idx in range(signal.shape[1]):
        ch_signal = signal[:, ch_idx]

        mav = np.mean(np.abs(ch_signal))
        rms = np.sqrt(np.mean(ch_signal**2))
        var = np.var(ch_signal)
        wl = np.sum(np.abs(np.diff(ch_signal)))

        threshold = 0.01
        zc = np.sum(np.diff(np.sign(ch_signal)) != 0)
        ssc = np.sum(np.diff(np.sign(np.diff(ch_signal))) != 0)

        features.extend([mav, rms, var, wl, zc, ssc])

    return np.array(features)


def extract_frequency_domain_features(signal, fs=2000):
    """Extract frequency-domain features from single trial."""
    features = []

    for ch_idx in range(signal.shape[1]):
        ch_signal = signal[:, ch_idx]

        freqs, psd = scipy_signal.welch(ch_signal, fs=fs, nperseg=min(256, len(ch_signal)))

        total_power = np.sum(psd)
        if total_power > 0:
            mnf = np.sum(freqs * psd) / total_power
            cumsum_psd = np.cumsum(psd)
            mdf_idx = np.where(cumsum_psd >= total_power / 2)[0]
            mdf = freqs[mdf_idx[0]] if len(mdf_idx) > 0 else 0
        else:
            mnf = 0
            mdf = 0

        features.extend([mnf, mdf])

    return np.array(features)


def extract_all_discrete_features(signal, fs=2000):
    """Extract all discrete features from single trial."""
    td_features = extract_time_domain_features(signal)
    fd_features = extract_frequency_domain_features(signal, fs)
    return np.concatenate([td_features, fd_features])


def prepare_timeseries_features(signal, target_length=500):
    """Downsample time-series for VAE input."""
    original_length = signal.shape[0]
    if original_length == target_length:
        return signal

    indices = np.linspace(0, original_length - 1, target_length, dtype=int)
    return signal[indices, :]


def extract_features_for_dataset(signals, metadata, fs=2000, ts_length=500):
    """Extract both discrete and time-series features for entire dataset."""
    X_discrete = []
    X_ts = []
    y = []

    print(f"Extracting features from {signals.shape[0]} trials...")
    for i in range(signals.shape[0]):
        discrete_feats = extract_all_discrete_features(signals[i], fs=fs)
        ts_feats = prepare_timeseries_features(signals[i], target_length=ts_length)

        X_discrete.append(discrete_feats)
        X_ts.append(ts_feats)
        y.append(metadata.iloc[i]['subject'])

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{signals.shape[0]}")

    X_discrete = np.array(X_discrete)
    X_ts = np.array(X_ts)
    y = np.array(y)

    print(f"Discrete features shape: {X_discrete.shape}")
    print(f"Time-series features shape: {X_ts.shape}")
    print(f"Labels shape: {y.shape}")

    return X_ts, X_discrete, y


def split_subject_trials_by_metadata(X_ts, X_discrete, y, metadata, subject_list, n_enrollment=3):
    """Split specified subjects' trials into enrollment and test sets."""
    enrollment_data = {'X_ts': [], 'X_discrete': [], 'y': [], 'subjects': []}
    test_data = {'X_ts': [], 'X_discrete': [], 'y': [], 'subjects': []}

    for subject in subject_list:
        subject_mask = metadata['subject'] == subject
        subject_indices = metadata[subject_mask].index.tolist()

        if len(subject_indices) < n_enrollment + 1:
            n_enroll = max(1, len(subject_indices) // 2)
        else:
            n_enroll = n_enrollment

        enroll_idx = subject_indices[:n_enroll]
        test_idx = subject_indices[n_enroll:]

        enrollment_data['X_ts'].append(X_ts[enroll_idx])
        enrollment_data['X_discrete'].append(X_discrete[enroll_idx])
        enrollment_data['y'].append(y[enroll_idx])
        enrollment_data['subjects'].extend([subject] * len(enroll_idx))

        test_data['X_ts'].append(X_ts[test_idx])
        test_data['X_discrete'].append(X_discrete[test_idx])
        test_data['y'].append(y[test_idx])
        test_data['subjects'].extend([subject] * len(test_idx))

    enrollment_combined = (
        np.vstack(enrollment_data['X_ts']),
        np.vstack(enrollment_data['X_discrete']),
        np.concatenate(enrollment_data['y'])
    )

    test_combined = (
        np.vstack(test_data['X_ts']),
        np.vstack(test_data['X_discrete']),
        np.concatenate(test_data['y'])
    )

    return enrollment_combined, test_combined


def create_all_fold_splits(metadata, n_folds=5, random_state=42):
    """Generate subject assignments for all folds."""
    subjects = sorted(metadata['subject'].unique())
    n_subjects = len(subjects)

    n_eval = n_subjects // n_folds
    n_adapt = n_subjects // n_folds

    fold_configs = []

    for fold_idx in range(n_folds):
        eval_start = fold_idx * n_eval
        eval_end = eval_start + n_eval

        adapt_start = eval_end % n_subjects
        adapt_end = (adapt_start + n_adapt) % n_subjects

        eval_subjects = subjects[eval_start:eval_end]

        if adapt_end > adapt_start:
            adapt_subjects = subjects[adapt_start:adapt_end]
        else:
            adapt_subjects = subjects[adapt_start:] + subjects[:adapt_end]

        all_eval_adapt = set(eval_subjects + adapt_subjects)
        train_subjects = [s for s in subjects if s not in all_eval_adapt]

        fold_configs.append({
            'fold': fold_idx,
            'train_subjects': train_subjects,
            'adapt_subjects': adapt_subjects,
            'eval_subjects': eval_subjects
        })

        print(f"Fold {fold_idx}: Train={len(train_subjects)}, Adapt={len(adapt_subjects)}, Eval={len(eval_subjects)}")

    return fold_configs


def get_subject_data(X_ts, X_discrete, y, metadata, subject_list):
    """Get data for specified subjects."""
    mask = metadata['subject'].isin(subject_list)
    indices = metadata[mask].index.tolist()

    return X_ts[indices], X_discrete[indices], y[indices], metadata.iloc[indices].reset_index(drop=True)


def augment_clean_timeseries(signal, noise_level=0.02, time_shift_range=10, amp_scale_range=(0.95, 1.05)):
    """Augment cleaned EMG time-series."""
    augmented = signal.copy()

    noise = np.random.normal(0, noise_level * np.std(signal), signal.shape)
    augmented += noise

    if time_shift_range > 0:
        shift = np.random.randint(-time_shift_range, time_shift_range + 1)
        augmented = np.roll(augmented, shift, axis=0)

    scale = np.random.uniform(amp_scale_range[0], amp_scale_range[1])
    augmented *= scale

    return augmented



def classification_report_biometric(y_true, y_pred, y_scores=None, positive_label=1):
    """
    Generate classification report with standard metrics, Specificity, and Biometric stats.
    """

    # 1. Convert to binary integers (Genuine=1, Impostor=0)
    y_true_binary = (y_true == positive_label).astype(int)
    y_pred_binary = (y_pred == positive_label).astype(int)

    # 2. Calculate Basic Components using Confusion Matrix
    cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # 3. Calculate The Requested Metrics
    precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_true_binary, y_pred_binary, zero_division=0) # Same as TAR
    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)

    # Specificity = TN / (TN + FP) -> Ability to correctly reject impostors
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # 4. Biometric Specific Metrics
    tar = recall

    n_impostor = tn + fp
    far = fp / n_impostor if n_impostor > 0 else 0.0

    n_genuine = tp + fn
    frr = fn / n_genuine if n_genuine > 0 else 0.0

    print("\n" + "="*50)
    print("      PERFORMANCE METRICS (Genuine Class 1)")
    print("="*50)
    print(f"PRECISION:   {precision * 100:.2f}%")
    print(f"RECALL:      {recall * 100:.2f}%  (Same as TAR)")
    print(f"SPECIFICITY: {specificity * 100:.2f}%  (1 - FAR)")
    print(f"F1 SCORE:    {f1 * 100:.2f}%")
    print("-" * 50)

    # Print Biometric Stats
    print("\nBiometric Authentication Details:")
    print(f"True Acceptance Rate (TAR):  {tar:.4f}")
    print(f"False Acceptance Rate (FAR): {far:.4f}")
    print(f"False Rejection Rate (FRR):  {frr:.4f}")

    biometric_metrics = {
        'Precision': precision,
        'Recall': recall,
        'Specificity': specificity,
        'F1': f1,
        'TAR': tar,
        'FAR': far,
        'FRR': frr,
        'n_genuine': n_genuine,
        'n_impostor': n_impostor
    }

    # EER Calculation
    if y_scores is not None:
        fpr, tpr, thresholds = roc_curve(y_true_binary, y_scores)
        frr_curve = 1 - tpr

        eer_idx = np.argmin(np.abs(fpr - frr_curve))
        eer = (fpr[eer_idx] + frr_curve[eer_idx]) / 2

        biometric_metrics['EER'] = eer
        biometric_metrics['EER_threshold'] = thresholds[eer_idx]

        print(f"Equal Error Rate (EER):      {eer:.4f}")
        print(f"EER Threshold:               {thresholds[eer_idx]:.4f}")

    print("-" * 50)
    print(f"Genuine Samples:  {n_genuine}")
    print(f"Impostor Samples: {n_impostor}")
    print("=" * 50 + "\n")

    return biometric_metrics


def download_grabmyo_dataset(save_dir='grabmyo_data', drive_zip_path='/content/drive/MyDrive/gesture-recognition-and-biometrics-electromyogram-grabmyo-1.1.0.zip'):
    os.makedirs(save_dir, exist_ok=True)

    # Define marker file path
    marker_file = os.path.join(save_dir, '.extraction_complete')

    # Check if extraction was already completed
    if os.path.exists(marker_file):
        print(f"✓ Dataset already extracted at: {save_dir}")
        print("Skipping extraction to save time.")
        return save_dir

    # Check if Google Drive is mounted
    if not os.path.exists('/content/drive'):
        print("Google Drive is not mounted. Please mount Google Drive to access the dataset.")
        print("You can do this by clicking the folder icon on the left, then the Google Drive icon.")
        return None

    print(f"Using dataset from Google Drive: {drive_zip_path}")

    if not os.path.exists(drive_zip_path):
        print(f"Error: Zip file not found at {drive_zip_path}")
        return None

    print("\nExtracting files...")
    try:
        with zipfile.ZipFile(drive_zip_path, 'r') as zip_ref:
            zip_ref.extractall(save_dir)

        print(f"Dataset extracted to: {save_dir}")

        # Create marker file to indicate successful extraction
        with open(marker_file, 'w') as f:
            f.write(f"Extraction completed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        print("✓ Extraction complete! Marker file created.")

    except Exception as e:
        print(f"Error during extraction: {e}")
        return None

    return save_dir