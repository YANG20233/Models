import random
import numpy as np
from sklearn.utils import resample

def bootstrap_evaluation(model, graphs, n_iterations=1000, sample_size=None, random_state=42):
    # Set seed for reproducibility
    random.seed(random_state)
    np.random.seed(random_state)

    if sample_size is None:
        sample_size = len(graphs)  # Default to full dataset size for each sample

    roc_auc_scores, f1_scores, balanced_acc_scores, sensitivity_scores, specificity_scores = [], [], [], [], []

    for i in range(n_iterations):
        # Resample the dataset with replacement
        sampled_graphs = resample(graphs, replace=True, n_samples=sample_size, random_state=random_state + i)

        # Create a loader from the resampled dataset
        loader = DataLoader(sampled_graphs, batch_size=32, shuffle=False)

        # Evaluate model on the resampled dataset
        roc_auc, f1, balanced_acc, sensitivity, specificity, _, _ = evaluate_model(model, loader)

        # Store the scores
        roc_auc_scores.append(roc_auc)
        f1_scores.append(f1)
        balanced_acc_scores.append(balanced_acc)
        sensitivity_scores.append(sensitivity)
        specificity_scores.append(specificity)

    # Calculate mean and 95% confidence intervals for each metric
    metrics = {
        'roc_auc': (np.mean(roc_auc_scores), np.percentile(roc_auc_scores, 2.5), np.percentile(roc_auc_scores, 97.5)),
        'f1': (np.mean(f1_scores), np.percentile(f1_scores, 2.5), np.percentile(f1_scores, 97.5)),
        'balanced_acc': (np.mean(balanced_acc_scores), np.percentile(balanced_acc_scores, 2.5), np.percentile(balanced_acc_scores, 97.5)),
        'sensitivity': (np.mean(sensitivity_scores), np.percentile(sensitivity_scores, 2.5), np.percentile(sensitivity_scores, 97.5)),
        'specificity': (np.mean(specificity_scores), np.percentile(specificity_scores, 2.5), np.percentile(specificity_scores, 97.5))
    }

    return metrics

bootstrap_evaluation(model, graphs)
