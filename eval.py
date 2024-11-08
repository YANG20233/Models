import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F 
import torch.nn as nn
import matplotlib.pyplot as plt

from rdkit import Chem, DataStructs
from rdkit.Chem import rdchem, Descriptors, rdMolDescriptors, AllChem, rdFingerprintGenerator
from rdkit.Chem import rdFingerprintGenerator as fpg

from torch.utils.data import random_split, Subset 
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool, global_add_pool

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score, classification_report, confusion_matrix, roc_curve, recall_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import resample

          
def evaluate_model(loader, model):
    model.eval()  # Set the model to evaluation mode
    all_probs = []
    all_labels = []
    
    with torch.no_grad():  # Disable gradient calculation
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)  # Forward pass
            probs = torch.sigmoid(out).cpu().numpy()  # Apply sigmoid for binary classification
            
            all_probs.extend(probs)
            all_labels.extend(batch.y.cpu().numpy())
    
    # Use a fixed decision boundary of 0.7
    threshold = 0.7
    preds = (np.array(all_probs) >= threshold).astype(int)
    
    # Calculate binary metrics using the fixed threshold
    roc_auc = roc_auc_score(all_labels, all_probs)  # ROC-AUC for binary classification
    f1 = f1_score(all_labels, preds)  # F1 score for binary classification
    balanced_acc = balanced_accuracy_score(all_labels, preds)  # Balanced accuracy for binary classification

    # Calculate confusion matrix for sensitivity and specificity
    tn, fp, fn, tp = confusion_matrix(all_labels, preds).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity (recall)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # Specificity

    return roc_auc, f1, balanced_acc, sensitivity, specificity, all_labels, all_probs, threshold
