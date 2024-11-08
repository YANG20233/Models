import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F 
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import random_split, Subset 
from torch_geometric.data import DataLoader 
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool, global_add_pool

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import KFold

def evaluate_model(loader, model):
    model.eval()  # Set the model to evaluation mode
    all_probs = []
    all_labels = []
    
    with torch.no_grad():  # Disable gradient calculation
        for batch in loader:
            batch = batch.to(device)  # Use global `device`
            out = model(batch.x, batch.edge_index, batch.batch)  # Forward pass
            probs = torch.sigmoid(out).cpu().numpy()  # Apply sigmoid for binary classification
            
            all_probs.extend(probs)
            all_labels.extend(batch.y.cpu().numpy())

    # Sweep through thresholds to find the one that maximizes balanced accuracy
    thresholds = np.linspace(0.1, 0.9, 100)
    best_balanced_acc = 0.0
    best_preds = None
    optimal_threshold = 0.5  # Default threshold, updated if a better one is found
    
    for threshold in thresholds:
        preds = (np.array(all_probs) >= threshold).astype(int)
        balanced_acc = balanced_accuracy_score(all_labels, preds)
        if balanced_acc > best_balanced_acc:
            best_balanced_acc = balanced_acc
            best_preds = preds
            optimal_threshold = threshold  # Update optimal threshold
    
    # Calculate binary metrics using the best predictions
    roc_auc = roc_auc_score(all_labels, all_probs)  # ROC-AUC for binary classification
    f1 = f1_score(all_labels, best_preds)  # F1 score for binary classification
    balanced_acc = balanced_accuracy_score(all_labels, best_preds)  # Balanced accuracy for binary classification

    # Return metrics along with the optimal threshold
    return {
        "roc_auc": roc_auc,
        "f1": f1,
        "balanced_acc": balanced_acc,
        "optimal_threshold": optimal_threshold,
        "all_labels": all_labels,
        "all_probs": all_probs
    }

def train():
   model.train()
   total_loss = 0
   for batch in train_loader:
       batch = batch.to(device)
       optimizer.zero_grad()
       
       # Forward pass
       out = model(batch.x, batch.edge_index, batch.batch)
       
       # Squeeze the output to remove the extra dimension
       out = out.squeeze(1)  # Change shape from [32, 1] to [32]
       
       # Ensure target labels are of type float
       loss = loss_func(out, batch.y.float())  # Convert batch.y to float for BCEWithLogitsLoss
       
       loss.backward()
       optimizer.step()
       total_loss += loss.item() * batch.num_graphs

   scheduler.step()  # Update the learning rate if using a scheduler
   return total_loss / len(train_loader.dataset)

for epoch in range(50):
    train_loss = train()
    # Get metrics from evaluate_model
    metrics = evaluate_model(val_loader, model)
    roc_auc = metrics["roc_auc"]
    f1 = metrics["f1"]
    balanced_acc = metrics["balanced_acc"]
    optimal_threshold = metrics["optimal_threshold"]
    
    # Early stopping based on ROC-AUC
    early_stopping(roc_auc)
    print(f'Epoch {epoch + 1}, Loss: {train_loss:.4f}, ROC-AUC: {roc_auc:.4f}, F1-Score: {f1:.4f}, Balanced Accuracy: {balanced_acc:.4f}, Optimal Threshold: {optimal_threshold:.2f}')
    
    if early_stopping.early_stop:
        print("Early stopping triggered")
        break
        
# Define cross-validation function
def cross_validate(model_class, dataset, batch_size, k_folds=5):
    # KFold splits the dataset into k folds
    kfold = KFold(n_splits=k_folds, shuffle=True)

    results = {
        'roc_auc': [],
        'f1': [],
        'balanced_acc': []
    }

    # Define the model arguments based on your dataset's input size
    in_channels = dataset[0].x.size(1)  # Number of input features per node
    hidden_channels = 64  # Hidden layer size
    out_channels = 1  # Output size (binary classification)
    num_heads = 4  # Number of attention heads
    num_layers = 3  # Number of layers in GATv2
    dropout = 0.3  # Dropout rate

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f'Fold {fold + 1}/{k_folds}')

        # Subset the dataset for training and validation
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        # Use PyTorch Geometric DataLoader for batching graphs
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        # Instantiate a new model for each fold (to reset parameters)
        model = model_class(in_channels=in_channels,
                            hidden_channels=hidden_channels,
                            out_channels=out_channels,
                            num_heads=num_heads,
                            num_layers=num_layers,
                            dropout=dropout).to(device)

        # Reset optimizer and scheduler for each fold
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

        # Training loop for a certain number of epochs (e.g., 10)
        for epoch in range(10):
            train_loss = train_model_cross_val(train_loader, model, optimizer, scheduler)
            roc_auc, f1, balanced_acc = evaluate_model(val_loader, model)
            print(f'Epoch {epoch + 1}: Loss: {train_loss:.4f}, ROC-AUC: {roc_auc:.4f}, F1: {f1:.4f}, Balanced Accuracy: {balanced_acc:.4f}')
        
        # Store the results for this fold
        results['roc_auc'].append(roc_auc)
        results['f1'].append(f1)
        results['balanced_acc'].append(balanced_acc)

    # Calculate the average metrics across all folds
    avg_roc_auc = sum(results['roc_auc']) / k_folds
    avg_f1 = sum(results['f1']) / k_folds
    avg_balanced_acc = sum(results['balanced_acc']) / k_folds

    print(f'\nCross-Validation Results:')
    print(f'Average ROC-AUC: {avg_roc_auc:.4f}')
    print(f'Average F1-Score: {avg_f1:.4f}')
    print(f'Average Balanced Accuracy: {avg_balanced_acc:.4f}')

    return results

# Define the training function
def train_model_cross_val(train_loader, model, optimizer, scheduler):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        out = out.squeeze(1)
        loss = loss_func(out, batch.y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    scheduler.step()
    return total_loss / len(train_loader.dataset)

results = cross_validate(GATv2Model, graphs, batch_size=32, k_folds=5)

