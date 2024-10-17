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

random_seed = 42
torch.manual_seed(random_seed)
random.seed(random_seed)

# Extract labels to use for stratified splitting
labels = np.array([graph.y.item() for graph in graphs])

# Stratified splitting using sklearn's train_test_split
train_indices, temp_indices = train_test_split(
    np.arange(len(labels)), test_size=0.2, stratify=labels, random_state=42
)

val_indices, test_indices = train_test_split(
    temp_indices, test_size=0.5, stratify=labels[temp_indices], random_state=42
)

# Create subsets based on the indices
train_graphs = Subset(graphs, train_indices)
val_graphs = Subset(graphs, val_indices)
test_graphs = Subset(graphs, test_indices)

# Create DataLoaders for each split
train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)
val_loader = DataLoader(val_graphs, batch_size=32, shuffle=True)

class EarlyStopping:
   def __init__(self, patience=10, min_delta=0):
       self.patience = patience
       self.min_delta = min_delta
       self.counter = 0
       self.best_score = None 
       self.early_stop = False


   def __call__(self, score):  
       if self.best_score is None:
           self.best_score = score
       elif score < self.best_score + self.min_delta:  
           self.counter += 1
           if self.counter >= self.patience:
               self.early_stop = True
       else:
           self.best_score = score
           self.counter = 0

# Define the GATv2 Layer with Leaky ReLU and BatchNorm
class GATv2Layer(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, concat=True, dropout=0.3):
        super(GATv2Layer, self).__init__()
        self.gatv2_conv = GATv2Conv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=num_heads,
            concat=concat,
            dropout=dropout
        )
        self.batch_norm = nn.BatchNorm1d(out_channels * num_heads if concat else out_channels)

    def forward(self, x, edge_index):
        x = self.gatv2_conv(x, edge_index)  # GATv2 convolution
        x = self.batch_norm(x)  # Apply batch normalization
        x = F.leaky_relu(x, negative_slope=0.02)  # Adjusted Leaky ReLU
        return x

# Define the GATv2 Model for classification
class GATv2Model(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads, num_layers, dropout=0.3):
        super(GATv2Model, self).__init__()
        self.layers = nn.ModuleList()

        # First GATv2 layer
        self.layers.append(GATv2Layer(in_channels, hidden_channels, num_heads, dropout=dropout))

        # Intermediate GATv2 layers
        for _ in range(num_layers - 2):
            self.layers.append(GATv2Layer(hidden_channels * num_heads, hidden_channels, num_heads, dropout=dropout))

        # Final GATv2 layer without concatenation
        self.layers.append(GATv2Layer(hidden_channels * num_heads, hidden_channels, num_heads=1, concat=False, dropout=dropout))

        # Fully connected layer for classification
        self.fc = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        for layer in self.layers[:-1]:
            x = layer(x, edge_index)  # Pass through intermediate GATv2 layers

        x = self.layers[-1](x, edge_index)  # Last GATv2 layer
        x = global_mean_pool(x, batch)  # Aggregate node features to graph level
        x = self.fc(x)  # Fully connected classification layer
        return torch.sigmoid(x)  # Sigmoid activation for binary classification

# Model, optimizer, and loss function
model = GATv2Model(in_channels=graphs[0].x.size(1),
                   hidden_channels=64, 
                   out_channels=1, 
                   num_heads=4,
                   num_layers=3, 
                   dropout=0.3
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
early_stopping = EarlyStopping(patience=10, min_delta=1e-4)
loss_func = torch.nn.BCEWithLogitsLoss()
