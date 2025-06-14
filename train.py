# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 16:39:41 2025

@author: giopo
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from DrugInteractionDataset import DrugInteractionDataset
from DNN import DrugInteractionDNN
import pandas as pd
from sklearn.model_selection import train_test_split
import os

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for features, labels in dataloader:
        features, labels = features.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * features.size(0)
    
    return running_loss / len(dataloader.dataset)


def evaluate(model, dataloader, device, threshold=0.5, category_names=None):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            outputs = model(features)
            preds = (outputs.cpu().numpy() > threshold).astype(int)
            all_preds.append(preds)
            all_labels.append(labels.numpy())

    y_true = np.vstack(all_labels)
    y_pred = np.vstack(all_preds)

    report = classification_report(
        y_true, y_pred, 
        target_names=category_names if category_names else None,
        output_dict=True,
        zero_division=0
    )

    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'report': report
    }

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Φόρτωση αρχικού CSV
df = pd.read_csv('C:/Users/giopo/OneDrive/Έγγραφα/Πρόβλεψη αλληλεπιδράσεων φαρμάκων/code/main/data preparation/train_in_distribution.csv')

# Split σε 80% train και 20% validation
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

# Αποθήκευση σε νέα CSV
train_df.to_csv('C:/Users/giopo/OneDrive/Έγγραφα/Πρόβλεψη αλληλεπιδράσεων φαρμάκων/code/main/data preparation/train_split.csv', index=False)
val_df.to_csv('C:/Users/giopo/OneDrive/Έγγραφα/Πρόβλεψη αλληλεπιδράσεων φαρμάκων/code/main/data preparation/val_split.csv', index=False)

print("✅ Έγινε split: train_split.csv & val_split.csv")

# Φόρτωση του training dataset με oversampling (ή undersampling)
train_dataset = DrugInteractionDataset(
    'C:/Users/giopo/OneDrive/Έγγραφα/Πρόβλεψη αλληλεπιδράσεων φαρμάκων/code/main/data preparation/train_split.csv',
    'C:/Users/giopo/OneDrive/Έγγραφα/Πρόβλεψη αλληλεπιδράσεων φαρμάκων/code/main/data preparation/categories.csv',
    balance_mode="oversample"  # ή "undersample"
)

# Φόρτωση validation και test χωρίς balancing
val_dataset = DrugInteractionDataset(
    'C:/Users/giopo/OneDrive/Έγγραφα/Πρόβλεψη αλληλεπιδράσεων φαρμάκων/code/main/data preparation/val_split.csv',
    'C:/Users/giopo/OneDrive/Έγγραφα/Πρόβλεψη αλληλεπιδράσεων φαρμάκων/code/main/data preparation/categories.csv',
    balance_mode='none'
)

test_dataset = DrugInteractionDataset(
    'C:/Users/giopo/OneDrive/Έγγραφα/Πρόβλεψη αλληλεπιδράσεων φαρμάκων/code/main/data preparation/test_in_distribution.csv',
    'C:/Users/giopo/OneDrive/Έγγραφα/Πρόβλεψη αλληλεπιδράσεων φαρμάκων/code/main/data preparation/categories.csv',
    balance_mode="none"
)

# DataLoaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)


# Μοντέλο
input_size = len(train_dataset[1][0])
output_size = len(train_dataset[1][1])
model = DrugInteractionDNN(input_dim=input_size, output_dim=output_size).to(device)

# Loss & Optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training parameters
best_acc = 0.0
patience = 12  
patience_counter = 0

# Training Loop
for epoch in range(1, 100):  # 100 epochs
    train_loss = train(model, train_loader, criterion, optimizer, device)
    val_metrics = evaluate(model, val_loader, device)
    
    print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | "
          f"Val Accuracy: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['f1']:.4f}")
    
    # Check overfiting
    if val_metrics['accuracy'] > best_acc:
        print(f"Validation F1 improved from {best_acc:.4f} to {val_metrics['accuracy']:.4f}")
        best_acc = val_metrics['accuracy']
      
        patience_counter = 0  # Reset patience counter
    else:
        patience_counter += 1
        print(f"No improvement in validation F1 for {patience_counter}/{patience} epochs")
        
        # Early stopping
        if patience_counter >= patience:
            print("Early stopping triggered")
            break


# Τελική αξιολόγηση στο test set

# Λίστα με τα ονόματα των κατηγοριών (στήλη 'category_name' στο categories.csv)
df = pd.read_csv("C:/Users/giopo/OneDrive/Έγγραφα/Πρόβλεψη αλληλεπιδράσεων φαρμάκων/code/main/data preparation/categories.csv")
category_names = df.iloc[:, 0].tolist()  # επιλέγει την πρώτη στήλη
category_names.append('No-Iteraction')


# Ορισμός OOD κατηγοριών (Out-of-Distribution)
ood_categories = [
    'psoriatic arthritis', 'monogenic disease', 'hypospadias',
    'chromosomal disease', 'polycystic ovary syndrome',
    'orofacial cleft', 'cryptorchidism'
]

category_names = [cat for cat in category_names if cat not in ood_categories]

# Τελική αξιολόγηση στο test set
test_metrics = evaluate(model, test_loader, device, category_names=category_names)

# Βασικά αποτελέσματα
print("\n Test Set Evaluation:")
print(f"Accuracy: {test_metrics['accuracy']:.4f} | "
      f"Precision: {test_metrics['precision']:.4f} | "
      f"Recall: {test_metrics['recall']:.4f} | "
      f"F1: {test_metrics['f1']:.4f}")

# Αναλυτικό report ανά κατηγορία
print("\n Detailed Classification Report:")
for label, metrics in test_metrics['report'].items():
    if isinstance(metrics, dict):
        print(f"{label:20s} | Precision: {metrics['precision']:.3f} | Recall: {metrics['recall']:.3f} | F1-score: {metrics['f1-score']:.3f}")

