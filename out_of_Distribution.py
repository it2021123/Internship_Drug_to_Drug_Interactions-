# -*- coding: utf-8 -*-
"""
Complete kNN-based OOD Detection with Visualization - 1)Logit sigmoid Output -2) Last Layer output
"""

# === Imports ===
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, classification_report, f1_score,
    accuracy_score, precision_score, recall_score, roc_auc_score
)
import ast
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cdist
from DrugInteractionDataset import DrugInteractionDataset
from DNN import DrugInteractionDNN
from sklearn.metrics import ConfusionMatrixDisplay

# === kNN-based OOD Detection Class ===
class KNNOoD:
    def __init__(self, seen_data, k_values=[5, 10, 20, 50], similarity_metric='dot'):
        self.k_values = k_values
        self.similarity_metric = similarity_metric
        self.seen_data = seen_data.values if isinstance(seen_data, pd.DataFrame) else seen_data

    def fit(self, seen_data):
        self.seen_data = seen_data

    def cosine_similarity(self, K, z):
        # Compute cosine similarity between seen embeddings K and a single vector z
        K_norm = np.linalg.norm(K, axis=1, keepdims=True)
        z_norm = np.linalg.norm(z)
        denominator = np.clip(K_norm.flatten() * z_norm, a_min=1e-10, a_max=None)
        similarity = (K @ z.T) / denominator
        return np.nan_to_num(similarity, nan=0.0, posinf=0.0, neginf=0.0)

    def compute_similarities(self, data):
        # Compute similarity/distance based on selected metric
        if self.similarity_metric == 'dot':
            sims = data @ self.seen_data.T
            sorted_sims = np.sort(sims, axis=1)[:, ::-1]
            return {k: sorted_sims[:, k - 1] for k in self.k_values}

        elif self.similarity_metric == 'norm':
            dists = np.linalg.norm(data[:, None, :] - self.seen_data[None, :, :], axis=2)
            sorted_dists = np.sort(dists, axis=1)
            return {k: sorted_dists[:, k - 1] for k in self.k_values}

        elif self.similarity_metric == 'cosine':
            sims = np.array([self.cosine_similarity(self.seen_data, z) for z in data])
            sorted_sims = np.sort(sims, axis=1)[:, ::-1]
            return {k: sorted_sims[:, k - 1] for k in self.k_values}

        elif self.similarity_metric in ['euclidean', 'manhattan', 'chebyshev', 'minkowski','mahalanobis']:
            dists = cdist(data, self.seen_data, metric=self.similarity_metric)
            sorted_dists = np.sort(dists, axis=1)
            return {k: sorted_dists[:, k - 1] for k in self.k_values}

        else:
            raise ValueError(f"Unsupported similarity metric: {self.similarity_metric}")

    def compute_ood_scores(self, unseen_data):
        unseen_array = unseen_data.values if isinstance(unseen_data, pd.DataFrame) else unseen_data
        results = self.compute_similarities(unseen_array)
        return pd.DataFrame(results)

# === Helper Function to Concatenate Embeddings ===
def concat_embeddings(df):
    emb1 = np.stack(df['embedding1_1'].apply(ast.literal_eval).values)
    emb2 = np.stack(df['embedding1_2'].apply(ast.literal_eval).values)
    emb3 = np.stack(df['embedding2_1'].apply(ast.literal_eval).values)
    emb4 = np.stack(df['embedding2_2'].apply(ast.literal_eval).values)
    return np.concatenate([emb1, emb2, emb3, emb4], axis=1)

# === OOD Evaluation Metrics ===
def evaluate_ood_detection(scores_in, scores_ood, reverse=False):
    scores = np.concatenate([scores_in, scores_ood])
    labels = np.array([0] * len(scores_in) + [1] * len(scores_ood))  # 0=InD, 1=OoD

    threshold_direction = '<=' if reverse else '>='
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    best_idx = np.argmax(tpr - fpr)
    cutoff = thresholds[best_idx]
    youden_j = tpr[best_idx] - fpr[best_idx]

    preds = scores <= cutoff if reverse else scores >= cutoff

    f1 = f1_score(labels, preds)
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    specificity = tn / (tn + fp + 1e-15)

    return {
        "f1_score": f1,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "roc_auc": roc_auc,
        "cutoff": cutoff,
        "threshold_direction": threshold_direction,
        "youden_j": youden_j
    }

# === Density Plot for InD vs OoD Scores ===
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

# === ROC Curve Plot ===
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

# === Main Execution Block ===
if __name__ == "__main__":

    # Load datasets
    train_dataset = DrugInteractionDataset('data preparation/small_train_in_distribution.csv')
    test_seen_ds = DrugInteractionDataset('data preparation/small_test_in_distribution.csv')
    ood_dataset = DrugInteractionDataset('data preparation/small_out_of_distribution.csv')

    # Load pre-trained model
    model = DrugInteractionDNN(input_dim=512, output_dim=7)
    checkpoint = torch.load('best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    
    # Preprocess data and extract embeddings
    seen_input_df = train_dataset.get_data()
    unseen_input_df = ood_dataset.get_data()
    test_input_df = test_seen_ds.get_data()

    unseen_input_df = unseen_input_df.sample(n=min(2000, len(unseen_input_df)), random_state=42).reset_index(drop=True)
    test_seen_concat = concat_embeddings(test_input_df)
    seen_concat = concat_embeddings(seen_input_df)
    unseen_concat = concat_embeddings(unseen_input_df)

    # Combine training and test InD samples
    seen_concat = np.vstack([seen_concat, test_seen_concat])
    seen_input_df = seen_input_df[seen_input_df["num_labels"] != 0]
    test_seen_labels = np.stack(test_input_df[[f'cat_{i}' for i in range(7)]].values)

    # Pass embeddings through the model
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seen_tensor = torch.tensor(seen_concat, dtype=torch.float32).to(device)
        test_seen_tensor = torch.tensor(test_seen_concat, dtype=torch.float32).to(device)
        unseen_tensor = torch.tensor(unseen_concat, dtype=torch.float32).to(device)
        model.to(device)
        """
        if you want last layer embedding-output size 64 
        seen_tensor =model.hidden1(seen_tensor).cpu()
        test_seen_tensor =model.hidden1(test_seen_tensor)
        unseen_tensor = model.hidden1(unseen_tensor)
        
        test_seen_tensor = model.hidden2(test_seen_tensor)
        seen_tensor = model.hidden2(seen_tensor)
        unseen_tensor = model.hidden2(unseen_tensor)
        
        test_seen_tensor = model.hidden3(test_seen_tensor)
        seen_tensor = model.hidden3(seen_tensor)
        unseen_tensor = model.hidden3(unseen_tensor)
        
        seen_tensor = model.hidden4(seen_tensor)
        unseen_tensor = model.hidden4(unseen_tensor)
        test_seen_tensor = model.hidden4(test_seen_tensor)
        
        seen_data = model.hidden5(seen_tensor)
        unseen_data =model.hidden5(unseen_tensor)
        test_seen_data = model.hidden5(test_seen_tensor)

        """
        seen_data = model(seen_tensor).cpu().numpy()
        unseen_data = model(unseen_tensor).cpu().numpy()
        test_seen_data =model(test_seen_tensor).cpu().numpy()
        
        #unseen_logits = unseen_data.copy() 


    # === Initialize kNN-based Out-of-Distribution Detector ===
    k_values = [10, 50, 100, 200, 500]  # Different values of k for kNN
    detector = KNNOoD(seen_data, k_values=k_values, similarity_metric='dot')  # Use dot-product similarity
    detector.fit(seen_data)  # Store seen data embeddings
    
    # === Compute similarity scores for test seen and unseen (OOD) data ===
    # These scores represent the similarity or distance from each test point to its k-th nearest neighbor in the training data
    scores_in = detector.compute_ood_scores(test_seen_data)  # Scores for in-distribution data
    scores_ood = detector.compute_ood_scores(unseen_data)    # Scores for out-of-distribution data
    
    # === Evaluate the OOD detection performance for each value of k ===
    # Reverse threshold comparison for distance-based metrics (low = similar)
    reverse = detector.similarity_metric in ['norm', 'euclidean', 'cityblock', 'chebyshev', 'minkowski', 'mahalanobis']
    results = {}
    for k in k_values:
        s_in = scores_in[k].values
        s_ood = scores_ood[k].values
        results[k] = evaluate_ood_detection(s_in, s_ood, reverse=reverse)  # Compute F1, precision, recall, etc.
    
    # === Prepare a summary table with performance metrics for each k ===
    summary = []
    for k, metrics in results.items():
        summary.append({
            "k": k,
            "Cutoff": round(metrics["cutoff"], 4),                # Best threshold separating InD vs OoD
            "F1-score": round(metrics["f1_score"], 4),
            "Accuracy": round(metrics["accuracy"], 4),
            "Precision": round(metrics["precision"], 4),
            "Recall": round(metrics["recall"], 4),
            "Specificity": round(metrics["specificity"], 4),
            "AUC": round(metrics["roc_auc"], 4),
            "Youden's J": round(metrics["youden_j"], 4)            # TPR - FPR
        })
    df_summary = pd.DataFrame(summary)
    
    # === Print the evaluation results in a formatted table ===
    print("\n=== Summary of Results ===")
    print(df_summary.to_string(index=False))
    
    # === Plot density plots showing score distributions ===
    # Helps visualize how well-separated InD and OoD scores are
    plot_density(scores_in, scores_ood, k_values)
    
    # === Plot ROC curves for all k values ===
    # AUC shows overall discrimination capability between InD and OoD
    plot_roc_curve(scores_in, scores_ood, k_values, similarity=True)
    
    # === Analyze Confusion Matrix and False Positives for each k ===
    for k in k_values:
        print(f"\nConfusion Matrix for k={k}:")
        s_in = scores_in[k].values
        s_ood = scores_ood[k].values
        y_true = np.concatenate([np.zeros_like(s_in), np.ones_like(s_ood)])  # 0: InD, 1: OoD
    
        # Apply the optimal threshold determined earlier
        cutoff = results[k]['cutoff']
        scores_all = np.concatenate([s_in, s_ood])
        y_pred = (scores_all <= cutoff) if reverse else (scores_all >= cutoff)
    
        # Compute and display confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        print(cm)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['InD', 'OoD'])
        disp.plot(cmap=plt.cm.Blues, values_format='d')
        plt.title(f'Confusion Matrix (k={k})')
        plt.grid(False)
        plt.show()
    
        # === Identify false positives (InD incorrectly classified as OoD) ===
        false_positives = [i for i in range(len(s_in)) if y_true[i] == 0 and y_pred[i] == 1]
        print(f"\n[False Positives - InD predicted as OoD] for k={k}")
        print(f"Total: {len(false_positives)} samples")
    
        # === Check if false positives mostly belong to the negative class (logit_0 = 1) ===
        if false_positives:
            fp_labels = test_seen_labels[false_positives]
            count_negative_class = np.sum(fp_labels[:, 0] == 1)  # class 0 is considered negative
            print(f"False Positives with class 0 (negative): {count_negative_class}")
        else:
            print("No false positives for this k.")   
        