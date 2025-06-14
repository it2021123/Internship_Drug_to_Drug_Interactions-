# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 14:14:29 2025

@author: giopo
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

# 1. Φόρτωσε το αρχείο CSV (αντικατάστησε με το πραγματικό σου όνομα αρχείου)
df = pd.read_csv('C:/Users/giopo/OneDrive/Έγγραφα/Πρόβλεψη αλληλεπιδράσεων φαρμάκων/code/main/data preparation/known_effects.csv')

# 2. Πάρε τη στήλη Disease Class, αφαίρεσε NaN και διπλότυπα
categories = df['Disease Class'].dropna().unique()

# 3. Ταξινόμησε αλφαβητικά αν θέλεις
categories_sorted = sorted(categories)

# 4. Αποθήκευση στο νέο αρχείο categories.csv
pd.Series(categories_sorted).to_csv('C:/Users/giopo/OneDrive/Έγγραφα/Πρόβλεψη αλληλεπιδράσεων φαρμάκων/code/main/data preparation/categories.csv', index=False, header=['Disease Class'])


class DrugInteractionDataset(Dataset):
    def __init__(self, csv_file, categories_file=None, balance_mode="none"):
        self.data = pd.read_csv(csv_file)

        # Δηλώνεις OOD labels με το όνομά τους
        ood_categories = [
            'psoriatic arthritis', 'monogenic disease', 'hypospadias',
            'chromosomal disease', 'polycystic ovary syndrome',
            'orofacial cleft', 'cryptorchidism'
        ]

        # Αν δοθεί αρχείο με τα labels
        if categories_file:
            self.categories = pd.read_csv(categories_file, header=None)[0].tolist()
            self.in_distribution_categories = [cat for cat in self.categories if cat not in ood_categories]
            self.num_categories = len(self.in_distribution_categories)
            self.cat_cols = [f'cat_{i}' for i in range(self.num_categories)]
        else:
            self.cat_cols = [col for col in self.data.columns if col.startswith('cat_')]
            self.num_categories = len(self.cat_cols)

        self.embedding_cols = ['embedding1', 'embedding2']

        # Εφαρμογή oversampling ή undersampling
        if balance_mode == "oversample":
            self.data = self._oversample(self.data, self.cat_cols)
        elif balance_mode == "undersample":
            self.data = self._undersample(self.data, self.cat_cols)

    def _oversample(self, df, label_cols):
        cat_counts = df[label_cols].sum()
        target_count = cat_counts.max()

        new_rows = []
        for cat in label_cols:
            current_count = cat_counts[cat]
            if current_count < target_count:
                need = int(target_count - current_count)
                cat_df = df[df[cat] == 1]
                if not cat_df.empty:
                    sampled = cat_df.sample(need, replace=True, random_state=42)
                    new_rows.append(sampled)
        if new_rows:
            df = pd.concat([df] + new_rows).sample(frac=1, random_state=42).reset_index(drop=True)
        return df

    def _undersample(self, df, label_cols):
        cat_counts = df[label_cols].sum()
        target_count = cat_counts.min()

        dfs = []
        for cat in label_cols:
            cat_df = df[df[cat] == 1]
            sampled = cat_df.sample(int(target_count), random_state=42)
            dfs.append(sampled)
        df = pd.concat(dfs).sample(frac=1, random_state=42).reset_index(drop=True)
        return df

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        embedding1 = np.array(eval(self.data.iloc[idx]['embedding1']))
        embedding2 = np.array(eval(self.data.iloc[idx]['embedding2']))

        features = np.concatenate((embedding1, embedding2), axis=None)
        labels = self.data.iloc[idx][self.cat_cols].values.astype(np.float32)

        features = torch.tensor(features, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)
        
        return features, labels

# Παράδειγμα χρήσης:
if __name__ == "__main__":
    # Φόρτωση του dataset
    train_dataset = DrugInteractionDataset('C:/Users/giopo/OneDrive/Έγγραφα/Πρόβλεψη αλληλεπιδράσεων φαρμάκων/code/main/data preparation/train_in_distribution.csv', 'C:/Users/giopo/OneDrive/Έγγραφα/Πρόβλεψη αλληλεπιδράσεων φαρμάκων/code/main/data preparation/categories.csv')
    test_dataset = DrugInteractionDataset('C:/Users/giopo/OneDrive/Έγγραφα/Πρόβλεψη αλληλεπιδράσεων φαρμάκων/code/main/data preparation/test_in_distribution.csv')
    ood_dataset = DrugInteractionDataset('C:/Users/giopo/OneDrive/Έγγραφα/Πρόβλεψη αλληλεπιδράσεων φαρμάκων/code/main/data preparation/out_of_distribution.csv')
    
    # Παράδειγμα πρόσβασης
    features, labels = train_dataset[0]
    print(f"Features shape: {features.shape}")  # Θα είναι (embedding_size1 + embedding_size2,)
    print(f"Labels shape: {labels.shape}")     # Θα είναι (num_categories,)
    
    # Δημιουργία DataLoader
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Παράδειγμα batch
    batch_features, batch_labels = next(iter(train_loader))
    print(f"Batch features shape: {batch_features.shape}")  # (batch_size, embedding_size1 + embedding_size2)
    print(f"Batch labels shape: {batch_labels.shape}")      # (batch_size, num_categories)