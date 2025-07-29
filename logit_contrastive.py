# -*- coding: utf-8 -*-
"""
Enhanced Contrastive Learning with KNN-based OOD Detection 
multilabel clasiffication MLP as Feature extractor -logit - MLP Sigmoid Output
Created on Tue Jul 15 15:42:07 2025
@author: User
"""

from sklearn.datasets import make_blobs
import numpy as np
import pandas as pd
import ast
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, f1_score, precision_score, recall_score,
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay
)
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

from DrugInteractionDataset import DrugInteractionDataset
from DNN import DrugInteractionDNN
from out_of_Distribution import KNNOoD

# Dataset Embedding Extraction
def concat_embeddings(df):
    emb1 = np.stack(df['embedding1_1'].apply(ast.literal_eval).values)
    emb2 = np.stack(df['embedding1_2'].apply(ast.literal_eval).values)
    emb3 = np.stack(df['embedding2_1'].apply(ast.literal_eval).values)
    emb4 = np.stack(df['embedding2_2'].apply(ast.literal_eval).values)
    return np.concatenate([emb1, emb2, emb3, emb4], axis=1)

class SeenUnseenDataset(Dataset):
    def __init__(self, seen_data, unseen_data):
        
        self.unseen_data = unseen_data

        self.data = torch.cat([seen_data, self.unseen_data], dim=0)
        self.labels = torch.cat([
            torch.ones(len(seen_data)),     # Seen = 1
            torch.zeros(len(self.unseen_data))  # OoD = 0
        ], dim=0)

        perm = torch.randperm(len(self.data))
        self.data = self.data[perm]
        self.labels = self.labels[perm]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
class ProjectionHead(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=64, output_dim=128):
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

def evaluate_ood_detection(scores_in, scores_ood, reverse=False):
    """Enhanced evaluation with KNN scores"""
    scores = np.concatenate([scores_in, scores_ood])
    labels = np.array([0] * len(scores_in) + [1] * len(scores_ood))
    
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    # Optimal threshold (Youden's J statistic)
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

if __name__ == "__main__":
    # 1. Load data
    seen_ds = DrugInteractionDataset('data preparation/small_train_in_distribution.csv')
    unseen_ds = DrugInteractionDataset('data preparation/small_out_of_distribution.csv')
    test_seen_ds = DrugInteractionDataset('data preparation/small_test_in_distribution.csv')

    seen_input_df = seen_ds.get_data()
    unseen_input_df = unseen_ds.get_data()
    test_input_df = test_seen_ds.get_data()

    seen_input_df = seen_input_df[seen_input_df["num_labels"] != 0]
    cols = [f'cat_{i}' for i in range(7)]
    seen_labels = np.stack(seen_input_df[cols].values)
    test_seen_labels = np.stack(test_input_df[[f'cat_{i}' for i in range(7)]].values)

    seen_concat = concat_embeddings(seen_input_df)
    unseen_concat = concat_embeddings(unseen_input_df)
    test_concat = concat_embeddings(test_input_df)

    # 2. Load DNN model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DrugInteractionDNN(input_dim=512, output_dim=7).to(device)
    checkpoint = torch.load('best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 3. Extract features
    @torch.no_grad()
    def extract_features(x):
           x = torch.tensor(x, dtype=torch.float32).to(device)
          
           return model(x).cpu()
      
    seen_data = extract_features(seen_concat)
    unseen_data = extract_features(unseen_concat)
    test_data = extract_features(test_concat)
    half = unseen_data.size(0)-2000


    first_half = unseen_data[:4000]
    second_half = unseen_data[half:]

    
    # 4. Enhanced Contrastive Learning
    projection = ProjectionHead().to(device)


    dataset = SeenUnseenDataset(seen_data, first_half)
    loader = DataLoader(dataset, batch_size=126, shuffle=True)

    print("Training projection head...")
    # 2. Loss function
    criterion = ContrastiveLoss(margin=1)
 
    
    # 3. Optimizer & Scheduler
    optimizer = torch.optim.Adam(projection.parameters(), lr=0.0001, weight_decay=1e-4)

    epochs=300
    patience = 10
    best_loss = float('inf')
    patience_counter = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
    
        for batch_data, batch_labels in loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
    
            
            idx1 = torch.randperm(len(batch_data))
            idx2 = torch.randperm(len(batch_data))
           # For contrastive loss, we use the same batch twice as dummy pairs (replace with real pairs in practice)
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
    
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(projection.state_dict(), 'best_projection.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # 5. KNN-based OOD Detection
    @torch.no_grad()
    def get_projections(data):
        return projection(data.to(device)).cpu().numpy()
    """
    # Normalize projections
    z_base = F.normalize(torch.tensor(get_projections(seen_data)), dim=1).numpy()
    z_test = F.normalize(torch.tensor(get_projections(test_data)), dim=1).numpy()
    z_ood = F.normalize(torch.tensor(get_projections(unseen_data)), dim=1).numpy()
    """
    
    z_base = torch.tensor(get_projections(seen_data)).numpy()
    z_test = torch.tensor(get_projections(test_data)).numpy()
    z_ood = torch.tensor(get_projections(second_half)).numpy()
    
    
    k_values = [10, 50, 100, 200, 500]
    
    print("\nRunning KNN OOD detection...")
    # Initialize the KNN-based OOD detector with different k values and dot product similarity
    detector = KNNOoD(z_base, k_values=k_values, similarity_metric='dot')
    detector.fit(z_base)  # Fit the detector on in-distribution embeddings (training/base set)
    
    # Compute OOD scores for in-distribution test samples and out-of-distribution samples
    scores_in = detector.compute_ood_scores(z_test)
    scores_ood = detector.compute_ood_scores(z_ood)
    
    # Evaluate the OOD detection performance for each k
    results = {}
    for k in k_values:
        # The evaluation function returns metrics such as ROC AUC, F1 score, recall, precision, specificity, cutoff, Youden's J
        results[k] = evaluate_ood_detection(scores_in[k].values, scores_ood[k].values)
    
    # Print a summary table with the evaluation results
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
    
    # Depending on the similarity metric, determine whether lower or higher scores indicate OOD
    reverse = detector.similarity_metric in ['norm', 'euclidean', 'cityblock', 'chebyshev', 'minkowski', 'mahalanobis']
    
    # Plot density distributions of OOD scores for in-distribution and out-of-distribution samples
    plot_density(scores_in, scores_ood, k_values)
    # Plot ROC curves for each k with respect to similarity scores
    plot_roc_curve(scores_in, scores_ood, k_values, similarity=True)
    plt.show()
    
    # For each k, print confusion matrix and identify false positives
    for k in k_values:
        print(f"\nConfusion Matrix for k={k}:")
        s_in = scores_in[k].values
        s_ood = scores_ood[k].values
        # Construct true labels: 0 for in-distribution, 1 for out-of-distribution
        y_true = np.concatenate([np.zeros_like(s_in), np.ones_like(s_ood)])
        
        cutoff = results[k]['cutoff']  # Threshold chosen from evaluation step
        scores_all = np.concatenate([s_in, s_ood])
        # Decide predictions based on whether lower or higher scores indicate OOD
        y_pred = (scores_all <= cutoff) if reverse else (scores_all >= cutoff)
        
        # Compute and print confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        print(cm)
        
        # Visualize the confusion matrix with labels
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['InD', 'OoD'])
        disp.plot(cmap=plt.cm.Blues, values_format='d')
        plt.title(f'Confusion Matrix (k={k})')
        plt.grid(False)
        plt.show()
        
        # === Detect False Positives ===
        # False positives: in-distribution samples misclassified as out-of-distribution
        false_positives = []
        for i in range(len(s_in)):
            true_idx = i  # InD samples are at the start
            if y_true[true_idx] == 0 and y_pred[true_idx] == 1:
                false_positives.append(i)
        
        print(f"\n[False Positives - InD predicted as OoD] for k={k}")
        print(f"Total: {len(false_positives)} samples")
        
        # === Check how many false positives belong to the negative class (logit_0 == 1) ===
        if false_positives:
            fp_labels = test_seen_labels[false_positives]
            count_negative_class = np.sum(fp_labels[:, 0] == 1)
            print(f"False Positives that were negative samples (class 0): {count_negative_class}")
        else:
            print("No false positive predictions for this k.")
        