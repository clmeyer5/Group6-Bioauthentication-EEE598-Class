import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from data import extract_all_discrete_features, augment_clean_timeseries


def balance_with_augmentation(X_ts, X_discrete, y, target_per_class=None, fs=2000):
    """Balance dataset by augmenting minority classes."""
    unique_classes, class_counts = np.unique(y, return_counts=True)
    
    if target_per_class is None:
        target_per_class = class_counts.max()
    
    X_ts_balanced = []
    X_discrete_balanced = []
    y_balanced = []
    
    for cls in unique_classes:
        cls_mask = y == cls
        cls_X_ts = X_ts[cls_mask]
        cls_X_discrete = X_discrete[cls_mask]
        cls_y = y[cls_mask]
        
        n_original = len(cls_y)
        n_to_generate = target_per_class - n_original
        
        X_ts_balanced.append(cls_X_ts)
        X_discrete_balanced.append(cls_X_discrete)
        y_balanced.append(cls_y)
        
        if n_to_generate > 0:
            aug_X_ts = []
            aug_X_discrete = []
            
            for _ in range(n_to_generate):
                idx = np.random.randint(0, n_original)
                aug_signal = augment_clean_timeseries(cls_X_ts[idx])
                aug_discrete = extract_all_discrete_features(aug_signal, fs=fs)
                
                aug_X_ts.append(aug_signal)
                aug_X_discrete.append(aug_discrete)
            
            aug_X_ts = np.array(aug_X_ts)
            aug_X_discrete = np.array(aug_X_discrete)
            aug_y = np.full(n_to_generate, cls)
            
            X_ts_balanced.append(aug_X_ts)
            X_discrete_balanced.append(aug_X_discrete)
            y_balanced.append(aug_y)
    
    X_ts_balanced = np.concatenate(X_ts_balanced, axis=0)
    X_discrete_balanced = np.concatenate(X_discrete_balanced, axis=0)
    y_balanced = np.concatenate(y_balanced, axis=0)
    
    shuffle_idx = np.random.permutation(len(y_balanced))
    X_ts_balanced = X_ts_balanced[shuffle_idx]
    X_discrete_balanced = X_discrete_balanced[shuffle_idx]
    y_balanced = y_balanced[shuffle_idx]
    
    print(f"Balanced from {len(y)} to {len(y_balanced)} samples")
    
    return X_ts_balanced, X_discrete_balanced, y_balanced


class DataPreprocessor:
    """Handles scaling and encoding with proper train/test separation."""
    
    @staticmethod
    def one_hot_encode_labels(y_train, y_test):
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)
        if y_test.ndim == 1:
            y_test = y_test.reshape(-1, 1)
        
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoder.fit(y_train)
        
        y_train_ohe = encoder.transform(y_train)
        y_test_ohe = encoder.transform(y_test)
        
        return y_train_ohe, y_test_ohe, encoder
    
    
    @staticmethod
    def split_data(ts_data, test_size=0.2, random_state=42):
        X_train_raw, X_test_raw = train_test_split(
        ts_data, test_size=test_size, random_state=random_state
        )
    
        print(f"Split Sizes -> Train: {X_train_raw.shape}, Test: {X_test_raw.shape}")

        return X_train_raw, X_test_raw
    
    
    @staticmethod
    def scale_ts_features(X_train_raw, X_test_raw):
        num_train, length, channels = X_train_raw.shape
        num_test = X_test_raw.shape[0]
        
        # 1. Flatten for Scaling: (Batch * Length, Channels)
        # We still scale per-channel to preserve specific sensor ranges
        X_train_flat = X_train_raw.reshape(-1, channels)
        X_test_flat = X_test_raw.reshape(-1, channels)

        # 2. Fit Standard Scaler on TRAIN ONLY
        scaler = StandardScaler()
        X_train_scaled_flat = scaler.fit_transform(X_train_flat)
        X_test_scaled_flat = scaler.transform(X_test_flat)

        # 3. Reshape back to 3D (Batch, Length, Channels)
        X_train_scaled = X_train_scaled_flat.reshape(num_train, length, channels)
        X_test_scaled = X_test_scaled_flat.reshape(num_test, length, channels)

        # 4. Transpose for PyTorch Conv1d: (Batch, Channels, Length)
        X_train_final = np.transpose(X_train_scaled, (0, 2, 1))
        X_test_final = np.transpose(X_test_scaled, (0, 2, 1))
        
        # --- NEW STEP: Flatten Batch and Channels ---
        # Current: (N_Subjects, 28, 10240)
        # Target:  (N_Subjects * 28, 1, 10240)
        X_train_final = X_train_final.reshape(-1, 1, length)
        X_test_final = X_test_final.reshape(-1, 1, length)

        print(f"Reshaped for Per-Channel Training: {X_train_final.shape}")
        
        return X_train_final, X_test_final, scaler
    
    @staticmethod
    def scale_discrete_features(X_train_discrete, X_test_discrete):
        scaler = StandardScaler()
        scaler.fit(X_train_discrete)
        
        X_train_scaled = scaler.transform(X_train_discrete)
        X_test_scaled = scaler.transform(X_test_discrete)
        
        return X_train_scaled, X_test_scaled, scaler


class MixedFeatureDataset(Dataset):
    """PyTorch dataset for combined time-series and discrete features."""
    
    def __init__(self, ts_features, discrete_features, labels_ohe, device):
        self.ts_tensors = torch.tensor(ts_features, dtype=torch.float32, device=device)
        self.labels_tensors = torch.tensor(labels_ohe, dtype=torch.float32, device=device)
        
        if discrete_features is not None:
            self.discrete_tensors = torch.tensor(discrete_features, dtype=torch.float32, device=device)
        else:
            self.discrete_tensors = None
        
        self.n_samples = self.ts_tensors.shape[0]
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        ts = self.ts_tensors[idx]
        discrete = self.discrete_tensors[idx] if self.discrete_tensors is not None else torch.zeros(0, device=self.ts_tensors.device)
        label = self.labels_tensors[idx]
        
        return ts, discrete, label


def create_fold_loaders(train_data, adapt_enroll, adapt_test, eval_enroll, eval_test, 
                       device, batch_size=64, augment=True, validation_split=0.15):
    """Create all dataloaders for one fold with proper enrollment/test splits."""
    
    X_ts_train, X_discrete_train, y_train = train_data
    X_ts_adapt_enroll, X_discrete_adapt_enroll, y_adapt_enroll = adapt_enroll
    X_ts_adapt_test, X_discrete_adapt_test, y_adapt_test = adapt_test
    X_ts_eval_enroll, X_discrete_eval_enroll, y_eval_enroll = eval_enroll
    X_ts_eval_test, X_discrete_eval_test, y_eval_test = eval_test
    
    # Augment training data
    if augment:
        X_ts_train, X_discrete_train, y_train = balance_with_augmentation(
            X_ts_train, X_discrete_train, y_train
        )
    
    # Split training into train/val
    n_train = len(y_train)
    n_val = int(n_train * validation_split)
    indices = np.random.permutation(n_train)
    
    train_idx = indices[n_val:]
    val_idx = indices[:n_val]
    
    X_ts_tr = X_ts_train[train_idx]
    X_discrete_tr = X_discrete_train[train_idx]
    y_tr = y_train[train_idx]
    
    X_ts_val = X_ts_train[val_idx]
    X_discrete_val = X_discrete_train[val_idx]
    y_val = y_train[val_idx]
    
    # One-hot encode (fit on training only)
    y_tr_ohe, y_val_ohe, encoder = DataPreprocessor.one_hot_encode_labels(y_tr, y_val)
    y_adapt_enroll_ohe = encoder.transform(y_adapt_enroll.reshape(-1, 1))
    y_adapt_test_ohe = encoder.transform(y_adapt_test.reshape(-1, 1))
    y_eval_enroll_ohe = encoder.transform(y_eval_enroll.reshape(-1, 1))
    y_eval_test_ohe = encoder.transform(y_eval_test.reshape(-1, 1))
    
    # Scale features (fit on training only)
    X_ts_tr_scaled, X_ts_val_scaled, ts_scaler = DataPreprocessor.scale_ts_features(X_ts_tr, X_ts_val)
    X_discrete_tr_scaled, X_discrete_val_scaled, discrete_scaler = DataPreprocessor.scale_discrete_features(
        X_discrete_tr, X_discrete_val
    )
    
    # Scale adapt/eval data
    X_ts_adapt_enroll_scaled = ts_scaler.transform(
        X_ts_adapt_enroll.reshape(-1, X_ts_adapt_enroll.shape[-1])
    ).reshape(X_ts_adapt_enroll.shape)
    X_discrete_adapt_enroll_scaled = discrete_scaler.transform(X_discrete_adapt_enroll)
    
    X_ts_adapt_test_scaled = ts_scaler.transform(
        X_ts_adapt_test.reshape(-1, X_ts_adapt_test.shape[-1])
    ).reshape(X_ts_adapt_test.shape)
    X_discrete_adapt_test_scaled = discrete_scaler.transform(X_discrete_adapt_test)
    
    X_ts_eval_enroll_scaled = ts_scaler.transform(
        X_ts_eval_enroll.reshape(-1, X_ts_eval_enroll.shape[-1])
    ).reshape(X_ts_eval_enroll.shape)
    X_discrete_eval_enroll_scaled = discrete_scaler.transform(X_discrete_eval_enroll)
    
    X_ts_eval_test_scaled = ts_scaler.transform(
        X_ts_eval_test.reshape(-1, X_ts_eval_test.shape[-1])
    ).reshape(X_ts_eval_test.shape)
    X_discrete_eval_test_scaled = discrete_scaler.transform(X_discrete_eval_test)
    
    # Create datasets
    train_dataset = MixedFeatureDataset(X_ts_tr_scaled, X_discrete_tr_scaled, y_tr_ohe, device)
    val_dataset = MixedFeatureDataset(X_ts_val_scaled, X_discrete_val_scaled, y_val_ohe, device)
    adapt_enroll_dataset = MixedFeatureDataset(
        X_ts_adapt_enroll_scaled, X_discrete_adapt_enroll_scaled, y_adapt_enroll_ohe, device
    )
    adapt_test_dataset = MixedFeatureDataset(
        X_ts_adapt_test_scaled, X_discrete_adapt_test_scaled, y_adapt_test_ohe, device
    )
    eval_enroll_dataset = MixedFeatureDataset(
        X_ts_eval_enroll_scaled, X_discrete_eval_enroll_scaled, y_eval_enroll_ohe, device
    )
    eval_test_dataset = MixedFeatureDataset(
        X_ts_eval_test_scaled, X_discrete_eval_test_scaled, y_eval_test_ohe, device
    )
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    adapt_enroll_loader = DataLoader(adapt_enroll_dataset, batch_size=batch_size, shuffle=False)
    adapt_test_loader = DataLoader(adapt_test_dataset, batch_size=batch_size, shuffle=False)
    eval_enroll_loader = DataLoader(eval_enroll_dataset, batch_size=batch_size, shuffle=False)
    eval_test_loader = DataLoader(eval_test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train={len(train_loader.dataset)}, Val={len(val_loader.dataset)}")
    print(f"Adapt: Enroll={len(adapt_enroll_loader.dataset)}, Test={len(adapt_test_loader.dataset)}")
    print(f"Eval: Enroll={len(eval_enroll_loader.dataset)}, Test={len(eval_test_loader.dataset)}")
    
    return {
        'train': train_loader,
        'val': val_loader,
        'adapt_enroll': adapt_enroll_loader,
        'adapt_test': adapt_test_loader,
        'eval_enroll': eval_enroll_loader,
        'eval_test': eval_test_loader,
        'encoder': encoder,
        'ts_scaler': ts_scaler,
        'discrete_scaler': discrete_scaler
    }


def print_loader_stats(loaders):
    """Print basic statistics about dataloaders."""
    for name, loader in loaders.items():
        if isinstance(loader, DataLoader):
            print(f"{name}: {len(loader.dataset)} samples, {len(loader)} batches")
