# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 16:01:54 2025
@author: giopo
"""

import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from collections import defaultdict
import math

# 1. Load datasets
df_pos = pd.read_csv('dataset_with_both_AB_and_BA.csv')
df_neg = pd.read_csv('negative_dataset_with_both_AB_and_BA.csv')
#If you want to minimaze the population of negative samples
#df_neg = df_neg.sample(frac=0.1, random_state=42)

# 2. Load embeddings
print("Loading embeddings...")
with open('drug_embeddings.json', 'r') as f:
    embeddings1 = json.load(f)
with open('drug_embeddings_smiles.json', 'r') as f:
    embeddings2 = json.load(f)

# 3. Out-of-distribution (OOD) disease classes
ood_categories = [
    'psoriatic arthritis', 'monogenic disease', 'hypospadias',
    'chromosomal disease', 'polycystic ovary syndrome',
    'orofacial cleft', 'cryptorchidism', 'hematopoietic system diseases'
]

# 4. Create pair_key for merging
df_pos['pair_key'] = df_pos.apply(lambda row: (row['STITCH 1'], row['STITCH 2']), axis=1)
df_neg['pair_key'] = df_neg.apply(lambda row: (row['STITCH 1'], row['STITCH 2']), axis=1)

# Remove negatives that exist in positives
positive_keys = set(df_pos['pair_key'])
df_neg = df_neg[~df_neg['pair_key'].isin(positive_keys)].copy()

# 5. Group positive interactions by pair
pos_interactions = defaultdict(list)
for _, row in df_pos.iterrows():
    pos_interactions[row['pair_key']].append(row['Disease Class'])

# 6. Categories (excluding OOD)
freq_per_class = df_pos['Disease Class'].value_counts(dropna=True)
print(freq_per_class)

all_categories = set(df_pos['Disease Class'].dropna().unique())
all_categories.add("No-Interaction")
categories = sorted(list(all_categories - set(ood_categories)))

cat_to_idx = {cat: idx for idx, cat in enumerate(categories)}
num_categories = len(categories)

print("Categories:", categories)
print("Number of categories:", num_categories)

pd.Series(categories).to_csv('categories.csv', index=False, header=False)

# 7. Multi-hot encoding per drug pair
all_pairs = {}

# Encode positives
for pair, interactions in pos_interactions.items():
    encoding = np.zeros(num_categories, dtype=np.int8)
    has_ood = False
    valid_interactions = []
    all_labels = []

    for interaction in interactions:
        if interaction in ood_categories:
            has_ood = True
        else:
            idx = cat_to_idx.get(interaction)
            if idx is not None:
                encoding[idx] = 1
                valid_interactions.append(interaction)
        all_labels.append(interaction)

    all_pairs[pair] = {
        'encoding': encoding,
        'stitch1': pair[0],
        'stitch2': pair[1],
        'Disease Class': all_labels,
        'has_ood': has_ood
    }

# Encode negatives
for pair in df_neg['pair_key'].unique():
    if pair not in all_pairs:
        encoding = np.zeros(num_categories, dtype=np.int8)
        encoding[cat_to_idx['No-Interaction']] = 1
        all_pairs[pair] = {
            'encoding': encoding,
            'stitch1': pair[0],
            'stitch2': pair[1],
            'Disease Class': ['No-Interaction'],
            'has_ood': False
        }

# 8. Split into In-Distribution and OOD
in_distribution_pairs = {}
out_of_distribution_pairs = {}

for pair, data in all_pairs.items():
    if data['has_ood']:
        out_of_distribution_pairs[pair] = data
    else:
        in_distribution_pairs[pair] = data

# 9. Create dataset with embeddings
def create_dataset(pairs_dict):
    data = []
    missing = []
    
    for pair, data_dict in pairs_dict.items():
        s1, s2 = pair
        if s1 in embeddings1 and s2 in embeddings1 and s1 in embeddings2 and s2 in embeddings2:
            row = {
                'STITCH1': s1,
                'STITCH2': s2,
                'embedding1_1': embeddings1[s1]['embedding'],
                'embedding2_1': embeddings1[s2]['embedding'],
                'embedding1_2': embeddings2[s1]['embedding'],
                'embedding2_2': embeddings2[s2]['embedding'],
                **{f'cat_{i}': data_dict['encoding'][i] for i in range(num_categories)}
            }
            data.append(row)
        else:
            missing.append(pair)
    return pd.DataFrame(data), missing

in_dist_df, missing_in = create_dataset(in_distribution_pairs)
ood_df, missing_ood = create_dataset(out_of_distribution_pairs)

# 10. Report missing drugs
missing_drugs = set()
for pair in missing_in + missing_ood:
    for drug in pair:
        if drug not in embeddings1 or drug not in embeddings2:
            missing_drugs.add(drug)

# 11. Train/Test split with label coverage
def fast_pair_aware_split(df, test_size=0.01, random_state=42, enforce_all_labels=True):
    df = df.copy()
    df['pair_key'] = list(zip(df['STITCH1'], df['STITCH2']))
    label_cols = [col for col in df.columns if col.startswith("cat_")]
    grouped = df.groupby("pair_key")[label_cols].max()
    all_labels = set()

    for col in label_cols:
        if grouped[col].sum() > 0:
            all_labels.add(col)

    selected_test_pairs = set()
    covered_labels = set()

    for label in all_labels:
        candidates = grouped[grouped[label] == 1].index.difference(pd.Index(selected_test_pairs))
        if len(candidates) > 0:
            selected_test_pairs.add(candidates[0])
            covered_labels.add(label)
        else:
            print(f"[Warning] No pair found for label: {label}")

    remaining_pairs = list(set(grouped.index) - selected_test_pairs)
    np.random.seed(random_state)
    np.random.shuffle(remaining_pairs)
    desired_test_size = int(len(grouped) * test_size)
    remaining_needed = max(0, desired_test_size - len(selected_test_pairs))
    selected_test_pairs.update(remaining_pairs[:remaining_needed])

    is_test = df['pair_key'].isin(selected_test_pairs)
    test_df = df[is_test].drop(columns=['pair_key']).reset_index(drop=True)
    train_df = df[~is_test].drop(columns=['pair_key']).reset_index(drop=True)

    test_label_coverage = (test_df[label_cols].sum() > 0)
    missing_labels = [col for col in label_cols if not test_label_coverage[col]]

    if missing_labels:
        msg = f"[Error] The following labels are missing from the test set: {missing_labels}"
        if enforce_all_labels:
            raise ValueError(msg)
        else:
            print(msg)
    else:
        print("[Info] All labels are covered in the test set.")

    return train_df, test_df

train_df, test_df = fast_pair_aware_split(in_dist_df, test_size=0.031)
train_df.to_csv('train_in_distribution.csv', index=False)
test_df.to_csv('test_in_distribution.csv', index=False)
ood_df.to_csv('out_of_distribution.csv', index=False)

# 12. Analyze drug presence across splits
train_drugs = set(train_df['STITCH1']).union(set(train_df['STITCH2']))
test_drugs = set(test_df['STITCH1']).union(set(test_df['STITCH2']))
ood_drugs = set(ood_df['STITCH1']).union(set(ood_df['STITCH2']))

print(f"\nTrain: {len(train_df)} | Test: {len(test_df)} | OOD: {len(ood_df)}")
print(f"Drugs only in Test set: {test_drugs - train_drugs}")
print(f"Drugs only in OOD set: {ood_drugs - train_drugs}")
print(f"Missing embeddings for {len(missing_drugs)} unique drugs.")

# 13. Label distribution analysis
def analyze_distribution(df, name):
    label_cols = [col for col in df.columns if col.startswith("cat_")]
    print(f"\n{name} set distribution ({len(df)} samples):")
    
    for idx, col in enumerate(label_cols):
        count = df[col].sum()
        print(f"{categories[idx]}: {count} ({count/len(df):.1%})")
    
    df['num_labels'] = df[label_cols].sum(axis=1)
    print("\nMulti-label distribution:")
    print(df['num_labels'].value_counts().sort_index())

analyze_distribution(train_df, "Training")
analyze_distribution(test_df, "Test")
analyze_distribution(ood_df, "OOD Sample")
