# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 16:39:41 2025
Train MLP Multilabel Classification 
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
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


def train(model, dataloader, criterion, optimizer, device, scheduler=None):
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

    epoch_loss = running_loss / len(dataloader.dataset)
    if scheduler:
        scheduler.step(epoch_loss)
    return epoch_loss

    return epoch_loss

def evaluate(model, dataloader, device, threshold=0.5, category_names=None):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            outputs = model(features)
            probs = outputs.cpu().numpy()
            preds = (probs > threshold).astype(int)
            labels = labels.cpu().numpy()

            all_preds.append(preds)
            all_probs.append(probs)
            all_labels.append(labels)

    y_true = np.vstack(all_labels)
    y_pred = np.vstack(all_preds)
    y_probs = np.vstack(all_probs)

    report = classification_report(
        y_true, y_pred,
        target_names=category_names if category_names else None,
        output_dict=True,
        zero_division=0
    )

    binary_accuracy = (y_true == y_pred).mean()

    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'binary_accuracy': binary_accuracy,
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'report': report,
        'probs': y_probs
    }


def save_checkpoint(model, optimizer, epoch, metrics, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_f1': metrics['f1'],
        'val_accuracy': metrics['accuracy']
    }, path)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data preparation
df = pd.read_csv('data preparation/small_train_in_distribution.csv')

def fast_pair_aware_split(df, test_size=0.01, random_state=42, enforce_all_labels=True):
    df = df.copy()

    # Δημιουργία pair_key χωρίς apply
    df['pair_key'] = list(zip(df['STITCH1'], df['STITCH2']))

    # Όλες οι στήλες κατηγοριών
    label_cols = [col for col in df.columns if col.startswith("cat_")]

    # Ομαδοποίηση ανά pair_key
    grouped = df.groupby("pair_key")[label_cols].max()

    # Εύρεση όλων των κατηγοριών που υπάρχουν
    all_labels = set()
    for col in label_cols:
        if grouped[col].sum() > 0:
            all_labels.add(col)

    # Κάλυψη τουλάχιστον ενός δείγματος ανά κατηγορία
    selected_test_pairs = set()
    covered_labels = set()

    for label in all_labels:
        candidates = grouped[grouped[label] == 1].index.difference(pd.Index(selected_test_pairs))
        if len(candidates) > 0:
            selected_test_pairs.add(candidates[0])
            covered_labels.add(label)
        else:
            print(f"[Warning] No pair found for label: {label}")

    # Συμπλήρωσε test set μέχρι να φτάσει το επιθυμητό μέγεθος
    remaining_pairs = list(set(grouped.index) - selected_test_pairs)
    np.random.seed(random_state)
    np.random.shuffle(remaining_pairs)
    desired_test_size = int(len(grouped) * test_size)
    remaining_needed = max(0, desired_test_size - len(selected_test_pairs))
    selected_test_pairs.update(remaining_pairs[:remaining_needed])

    # Δημιουργία τελικών συνόλων
    is_test = df['pair_key'].isin(selected_test_pairs)
    test_df = df[is_test].drop(columns=['pair_key']).reset_index(drop=True)
    train_df = df[~is_test].drop(columns=['pair_key']).reset_index(drop=True)

    # Έλεγχος κάλυψης όλων των κατηγοριών
    test_label_coverage = (test_df[label_cols].sum() > 0)
    missing_labels = [col for col in label_cols if not test_label_coverage[col]]

    if missing_labels:
        msg = f"[Error] The following labels are missing from the val set: {missing_labels}"
        if enforce_all_labels:
            raise ValueError(msg)
        else:
            print(msg)
    else:
        print("[Info] All labels are covered in the validation set.")

    return train_df, test_df



train_df, val_df = fast_pair_aware_split(df, test_size=0.2)



# Save splits
train_df.to_csv('train_split.csv', index=False)
val_df.to_csv('val_split.csv', index=False)
print("✅ Created train and validation splits")

# Load datasets
train_dataset = DrugInteractionDataset(
    'train_split.csv',
    balance_mode="undersample"
)

val_dataset = DrugInteractionDataset(
    'val_split.csv',
    balance_mode='undersample'
)

test_dataset = DrugInteractionDataset(
    'data preparation/small_test_in_distribution.csv',
    balance_mode='undersample'  
)

#DataLoader
batch_size = 64
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model initialization
input_size = len(train_dataset[0][0])
output_size = len(train_dataset[0][1])
model = DrugInteractionDNN(input_dim=input_size, output_dim=output_size).to(device)


criterion = nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)


# Training parameters
best_f1 = 0.0
best_acc = 0.0
patience = 20
patience_counter = 0
best_model_path = 'best_model.pt'  # Single file for best model

# Training loop
for epoch in range(1, 101):  # 50 epochs
    train_loss = train(model, train_loader, criterion, optimizer, device)
    val_metrics = evaluate(model, val_loader, device)
    
    print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | "
          f"Val bin Accuracy: {val_metrics['binary_accuracy']:.4f} | Val F1: {val_metrics['f1']:.4f}")
    
    # Check for improvement
    if val_metrics['f1'] > best_f1  :
        print(f"Validation F1 improved from {best_f1:.4f} to {val_metrics['f1']:.4f}")
        best_f1 = val_metrics['f1']
        patience_counter = 0
        
        # Save ONLY the best model (overwrites previous)
        save_checkpoint(model, optimizer, epoch, val_metrics, best_model_path)
        print(f"Saved best model to {best_model_path}") 
    else:
        patience_counter += 1
        print(f"No improvement in validation F1 for {patience_counter}/{patience} epochs")
        
        # Early stopping
        if patience_counter >= patience:
            print("Early stopping triggered")
            break
    
 

# Load the best model for final evaluation
print("\nLoading best model for evaluation...")
checkpoint = torch.load(best_model_path)
model.load_state_dict(checkpoint['model_state_dict'])
# Get category names (excluding OOD categories)

"Change this line according to the 'seen' categories that you choose in Id_OoD.py or seven_class_only.py."
category_names =  ['No-Interaction', 'cardiovascular system disease', 'gastrointestinal system disease', 'hematopoietic system disease', 'integumentary system disease', 'nervous system disease', 'respiratory system disease']
# Final evaluation on test set
print("\nEvaluating on test set...")
test_metrics = evaluate(model, test_loader, device, category_names=category_names)

# Basic results
print("\nTest Set Evaluation:")
print(f"Accuracy: {test_metrics['accuracy']:.4f} | "
      f"Precision: {test_metrics['precision']:.4f} | "
      f"Recall: {test_metrics['recall']:.4f} | "
      f"F1: {test_metrics['f1']:.4f}")

# Detailed report per category
print("\nDetailed Classification Report:")
for label, metrics in test_metrics['report'].items():
    if isinstance(metrics, dict):
        print(f"{label:20s} | Precision: {metrics['precision']:.3f} | Recall: {metrics['recall']:.3f} | F1-score: {metrics['f1-score']:.3f}")