# -*- coding: utf-8 -*-
"""
Enhanced Contrastive Learning with KNN-based OOD Detection
multilabel clasiffication MLP as Feature extractor -size 64 - MLP Last hidden layer
Created on Mon Jun 23 2025
"""

# Standard libraries
from sklearn.datasets import make_blobs
import numpy as np
import pandas as pd
import ast

# PyTorch for neural networks
import torch
import torch.nn as nn
import torch.nn.functional as F

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Evaluation metrics
from sklearn.metrics import (
    roc_curve, auc, f1_score, precision_score, recall_score,
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay
)

# Utilities
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Custom modules
from DrugInteractionDataset import DrugInteractionDataset
from DNN import DrugInteractionDNN
from out_of_Distribution import KNNOoD

# ----------- Utility Functions and Classes ------------

# Extract and concatenate multiple embedding columns
def concat_embeddings(df):
    emb1 = np.stack(df['embedding1_1'].apply(ast.literal_eval).values)
    emb2 = np.stack(df['embedding1_2'].apply(ast.literal_eval).values)
    emb3 = np.stack(df['embedding2_1'].apply(ast.literal_eval).values)
    emb4 = np.stack(df['embedding2_2'].apply(ast.literal_eval).values)
    return np.concatenate([emb1, emb2, emb3, emb4], axis=1)

# Custom Dataset class combining seen and unseen samples for contrastive training
class SeenUnseenDataset(Dataset):
    def __init__(self, seen_data, unseen_data):
        self.unseen_data = unseen_data
        self.data = torch.cat([seen_data, unseen_data], dim=0)
        self.labels = torch.cat([
            torch.ones(len(seen_data)),       # Seen = 1
            torch.zeros(len(unseen_data))     # Unseen (OOD) = 0
        ], dim=0)

        # Shuffle dataset
        perm = torch.randperm(len(self.data))
        self.data = self.data[perm]
        self.labels = self.labels[perm]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Projection head used in contrastive learning
class ProjectionHead(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=256, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# Contrastive loss function (similar to Siamese loss)
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_pos = label * torch.pow(euclidean_distance, 2)
        loss_neg = (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        loss = torch.mean(loss_pos + loss_neg)
        return loss

# Evaluation for OOD scores using ROC and optimal threshold (Youden’s index)
def evaluate_ood_detection(scores_in, scores_ood, reverse=False):
    scores = np.concatenate([scores_in, scores_ood])
    labels = np.array([0] * len(scores_in) + [1] * len(scores_ood))

    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    youden_idx = np.argmax(tpr - fpr)
    cutoff = thresholds[youden_idx]
    youden_j = tpr[youden_idx] - fpr[youden_idx]

    preds = (scores <= cutoff) if reverse else (scores >= cutoff)

    return {
        "f1_score": f1_score(labels, preds),
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "specificity": 1 - fpr[youden_idx],
        "roc_auc": roc_auc,
        "cutoff": cutoff,
        "youden_j": youden_j,
        "threshold_direction": "<=" if reverse else ">="
    }

# KDE plot for score distributions
def plot_density(scores_in, scores_ood, k_values):
    plt.figure(figsize=(15, len(k_values) * 4))
    for idx, k in enumerate(k_values):
        plt.subplot(len(k_values), 1, idx + 1)
        sns.kdeplot(scores_in[k], label='In-Distribution', fill=True)
        sns.kdeplot(scores_ood[k], label='Out-of-Distribution', fill=True, color='red')
        plt.title(f'Density Plot for k = {k}')
        plt.legend()
    plt.tight_layout()
    plt.show()

# ROC curve plot for all k values
def plot_roc_curve(scores_in, scores_ood, k_values, similarity=True):
    y_true = np.concatenate([np.zeros(len(scores_in)), np.ones(len(scores_ood))])
    plt.figure(figsize=(10, 7))
    for k in k_values:
        scores = np.concatenate([scores_in[k], scores_ood[k]])
        y_scores = scores if similarity else -scores
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'k={k} (AUC={roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for different k')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

# -------------------- Main Script --------------------

if __name__ == "__main__":
    # 1. Load datasets
    seen_ds = DrugInteractionDataset('data preparation/small_train_in_distribution.csv')
    unseen_ds = DrugInteractionDataset('data preparation/small_out_of_distribution.csv')
    test_seen_ds = DrugInteractionDataset('data preparation/small_test_in_distribution.csv')

    seen_input_df = seen_ds.get_data()
    unseen_input_df = unseen_ds.get_data()
    test_input_df = test_seen_ds.get_data()

    # Remove zero-labeled rows
    seen_input_df = seen_input_df[seen_input_df["num_labels"] != 0]

    # Extract labels and embeddings
    cols = [f'cat_{i}' for i in range(7)]
    seen_labels = np.stack(seen_input_df[cols].values)
    test_seen_labels = np.stack(test_input_df[cols].values)

    seen_concat = concat_embeddings(seen_input_df)
    unseen_concat = concat_embeddings(unseen_input_df)
    test_concat = concat_embeddings(test_input_df)

    # 2. Load pretrained DNN
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DrugInteractionDNN(input_dim=512, output_dim=7).to(device)
    checkpoint = torch.load('best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 3. Extract intermediate features from the DNN
    @torch.no_grad()
    def extract_features(x):
        x = torch.tensor(x, dtype=torch.float32).to(device)
        x = model.hidden1(x)
        x = model.hidden2(x)
        x = model.hidden3(x)
        x = model.hidden4(x)
        return model.hidden5(x).cpu()

    seen_data = extract_features(seen_concat)
    unseen_data = extract_features(unseen_concat)
    test_data = extract_features(test_concat)

    # Split unseen data into two parts
    half = unseen_data.size(0) - 2000
    first_half = unseen_data[:4000]
    second_half = unseen_data[half:]

    # 4. Train contrastive projection head
    projection = ProjectionHead().to(device)
    dataset = SeenUnseenDataset(seen_data, first_half)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    print("Training projection head...")
    criterion = ContrastiveLoss(margin=2)
    optimizer = torch.optim.Adam(projection.parameters(), lr=0.001, weight_decay=1e-6)

    # Training with early stopping
    epochs = 300
    patience = 10
    best_loss = float('inf')
    patience_counter = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_data, batch_labels in loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)

            # Create random pairs
            idx1 = torch.randperm(len(batch_data))
            idx2 = torch.randperm(len(batch_data))

            out1 = projection(batch_data[idx1])
            out2 = projection(batch_data[idx2])
            labels = (batch_labels[idx1] == batch_labels[idx2]).float()

            loss = criterion(out1, out2, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(projection.state_dict(), 'best_projection.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # 5. OOD Detection using KNN on projected space
    @torch.no_grad()
    def get_projections(data):
        return projection(data.to(device)).cpu().numpy()

    z_base = torch.tensor(get_projections(seen_data)).numpy()
    z_test = torch.tensor(get_projections(test_data)).numpy()
    z_ood = torch.tensor(get_projections(second_half)).numpy()

    k_values = [10, 50, 100, 200, 500]
    print("\nRunning KNN OOD detection...")
    detector = KNNOoD(z_base, k_values=k_values, similarity_metric='dot')
    detector.fit(z_base)

    scores_in = detector.compute_ood_scores(z_test)
    scores_ood = detector.compute_ood_scores(z_ood)

    # Evaluate performance for each k
    results = {}
    for k in k_values:
        results[k] = evaluate_ood_detection(scores_in[k].values, scores_ood[k].values)

    # Print summarized metrics
    print("\n=== OOD Detection Results ===")
    summary = []
    for k, res in results.items():
        summary.append({
            'k': k,
            'AUC': f"{res['roc_auc']:.4f}",
            'F1': f"{res['f1_score']:.4f}",
            'Recall': f"{res['recall']:.4f}",
            'Precision': f"{res['precision']:.4f}",
            'Specificity': f"{res['specificity']:.4f}",
            'Cutoff': f"{res['cutoff']:.4f}",
            'Youden J': f"{res['youden_j']:.4f}"
        })
    print(pd.DataFrame(summary).to_string(index=False))

    # Plotting
    reverse = detector.similarity_metric in ['norm', 'euclidean', 'cityblock', 'chebyshev', 'minkowski','mahalanobis']
    plot_density(scores_in, scores_ood, k_values)
    plot_roc_curve(scores_in, scores_ood, k_values, similarity=True)

    # Confusion matrices and analysis of false positives
    for k in k_values:
        print(f"\nConfusion Matrix for k={k}:")
        s_in = scores_in[k].values
        s_ood = scores_ood[k].values
        y_true = np.concatenate([np.zeros_like(s_in), np.ones_like(s_ood)])
        cutoff = results[k]['cutoff']
        scores_all = np.concatenate([s_in, s_ood])
        y_pred = (scores_all <= cutoff) if reverse else (scores_all >= cutoff)

        cm = confusion_matrix(y_true, y_pred)
        print(cm)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['InD', 'OoD'])
        disp.plot(cmap=plt.cm.Blues, values_format='d')
        plt.title(f'Confusion Matrix (k={k})')
        plt.grid(False)
        plt.show()

        # Analyze false positives (InD samples predicted as OoD)
        false_positives = []
        for i in range(len(s_in)):
            if y_true[i] == 0 and y_pred[i] == 1:
                false_positives.append(i)

        print(f"\n[False Positives - InD predicted as OoD] for k={k}")
        print(f"Total: {len(false_positives)} samples")

        if false_positives:
            fp_labels = test_seen_labels[false_positives]
            count_negative_class = np.sum(fp_labels[:, 0] == 1)
            print(f"False Positives that were truly negative samples (class 0): {count_negative_class}")
        else:
            print("No false positives for this k.")
