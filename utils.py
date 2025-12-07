import torch
import numpy as np
import random
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def set_random_seeds(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_classification_metrics(y_true, y_pred):
    """Compute classification metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'confusion_matrix': cm
    }


def aggregate_fold_results(fold_results):
    """Aggregate results across all folds."""
    accuracies = [r['accuracy'] for r in fold_results]
    f1_scores = [r.get('f1_score', 0) for r in fold_results]
    
    return {
        'mean_accuracy': np.mean(accuracies),
        'std_accuracy': np.std(accuracies),
        'mean_f1': np.mean(f1_scores),
        'std_f1': np.std(f1_scores),
        'all_accuracies': accuracies,
        'all_f1_scores': f1_scores
    }


def print_results_summary(aggregated_results):
    """Print formatted results summary."""
    print("\n" + "="*50)
    print("CROSS-VALIDATION RESULTS")
    print("="*50)
    print(f"Mean Accuracy: {aggregated_results['mean_accuracy']:.4f} ± {aggregated_results['std_accuracy']:.4f}")
    print(f"Mean F1-Score: {aggregated_results['mean_f1']:.4f} ± {aggregated_results['std_f1']:.4f}")
    print("\nPer-Fold Accuracies:")
    for i, acc in enumerate(aggregated_results['all_accuracies']):
        print(f"  Fold {i}: {acc:.4f}")
    print("="*50 + "\n")
