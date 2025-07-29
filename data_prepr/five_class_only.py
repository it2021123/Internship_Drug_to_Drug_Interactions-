# -*- coding: utf-8 -*-
"""

"""
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from collections import defaultdict
from sklearn.utils import resample
import math
import random

# 1. Load data
print("Loading data...")
df_pos = pd.read_csv('dataset_with_both_AB_and_BA.csv')
df_neg = pd.read_csv('negative_dataset_with_both_AB_and_BA.csv')
df_neg = df_neg.sample(frac=0.1, random_state=42)


# 2. Load embeddings
print("Loading embeddings...")
with open('drug_embeddings.json', 'r') as f:
    embeddings1 = json.load(f)
with open('drug_embeddings_smiles.json', 'r') as f:
    embeddings2 = json.load(f)

#3. OOD categories
# Optional 'unseen' interaction?
ood_categories = [
    "inherited metabolic disorder",
    "benign neoplasm",
    "reproductive system disease",
    "sleep disorder",
    "bacterial infectious disease",
    "developmental disorder of mental health"
                 
]

# 4. Create pair keys and filter negatives
print("Processing drug pairs...")
df_pos['pair_key'] = list(zip(df_pos['STITCH 1'], df_pos['STITCH 2']))
df_neg['pair_key'] = list(zip(df_neg['STITCH 1'], df_neg['STITCH 2']))
df_neg = df_neg[~df_neg['pair_key'].isin(set(df_pos['pair_key']))].copy()

# 5. Group positive interactions
pos_interactions = defaultdict(list)
for _, row in df_pos.iterrows():
    pos_interactions[row['pair_key']].append(row['Disease Class'])

# 6. Select  seen categories. For example (top 7 + No-Interaction)
freq_per_class = df_pos['Disease Class'].value_counts(dropna=True)
top_6_categories = freq_per_class.head(6).index.tolist()
categories = sorted(top_6_categories + ['No-Interaction'])
#categories = sorted(top_6_categories) ->without No -Interaction
cat_to_idx = {cat: idx for idx, cat in enumerate(categories)}
num_categories = len(categories)

print("\nSelected categories:", categories)
print("Number of categories:", num_categories)
pd.Series(categories).to_csv('categories.csv', index=False, header=False)


# 7. Multi-hot encoding with category filtering
print("\nCreating multi-hot encodings...")
all_pairs = {}
category_counts = {cat: 0 for cat in categories}

# Positive pairs
for pair, interactions in pos_interactions.items():
    encoding = np.zeros(num_categories, dtype=np.int8)
    has_ood = False
    has_selected = False
    
    for interaction in interactions:
        if interaction in ood_categories:
            has_ood = True
        elif interaction in categories:
            idx = cat_to_idx[interaction]
            encoding[idx] = 1
            category_counts[interaction] += 1
            has_selected = True
    
    if has_selected or not interactions:
        all_pairs[pair] = {
            'encoding': encoding,
            'stitch1': pair[0],
            'stitch2': pair[1],
            'has_ood': has_ood,
            'interactions': interactions
        }

# Negative pairs -negative samples
for pair in df_neg['pair_key'].unique():
    if pair not in all_pairs:
        encoding = np.zeros(num_categories, dtype=np.int8)
        encoding[cat_to_idx['No-Interaction']] = 1
        category_counts['No-Interaction'] += 1
        all_pairs[pair] = {
            'encoding': encoding,
            'stitch1': pair[0],
            'stitch2': pair[1],
            'has_ood': False,
            'interactions': ['No-Interaction']
        }
   
print("\nCategory distribution in full dataset:")
for cat, count in category_counts.items():
    print(f"{cat}: {count} pairs")

# 8. Strict OOD splitting
print("\nSplitting in-distribution vs OOD...")
in_dist_pairs = {}
ood_pairs = {}

for pair, data in all_pairs.items():
    if data['has_ood']:
        if any(interaction in ood_categories for interaction in data['interactions']):
            ood_pairs[pair] = data
        else:
            in_dist_pairs[pair] = data
    else:
        in_dist_pairs[pair] = data

print(f"In-distribution pairs: {len(in_dist_pairs)}")
print(f"OOD pairs: {len(ood_pairs)}")

# 9. Create final datasets with embeddings
def create_dataset(pairs_dict):
    data = []
    missing = []
    
    for pair, data_dict in pairs_dict.items():
        s1, s2 = pair
        if s1 in embeddings1 and s2 in embeddings1 and s1 in embeddings2 and s2 in embeddings2 :
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

in_dist_df, missing_in = create_dataset(in_dist_pairs)
ood_df, missing_ood = create_dataset(ood_pairs)

# 10. Enhanced balanced sampling
def enhanced_balanced_sampling(df, target_size, random_state=42):
    if target_size >= len(df):
        return df.copy()
    
    df = df.copy()
    df['_temp_id'] = df.apply(lambda row: str(row['STITCH1']) + '_' + str(row['STITCH2']), axis=1)
    
    label_cols = [col for col in df.columns if col.startswith("cat_")]
    
    pos_counts = {col: df[col].sum() for col in label_cols}
    min_samples = min(pos_counts.values()) if pos_counts else 0
    
    samples_per_cat = {col: min(min_samples + 50, pos_counts[col]) for col in label_cols}
    
    sampled_dfs = []
    for col, n in samples_per_cat.items():
        if n > 0:
            subset = df[df[col] == 1]
            if len(subset) > 0:
                sampled = resample(subset, 
                                 replace=len(subset) < n,
                                 n_samples=min(n, len(subset)),
                                 random_state=random_state)
                sampled_dfs.append(sampled)
    
    if sampled_dfs:
        balanced_df = pd.concat(sampled_dfs).drop_duplicates(subset=['_temp_id'])
    else:
        balanced_df = pd.DataFrame(columns=df.columns)
    
    if len(balanced_df) < target_size:
        remaining = df[~df['_temp_id'].isin(balanced_df['_temp_id'])]
        needed = target_size - len(balanced_df)
        if len(remaining) > 0:
            balanced_df = pd.concat([
                balanced_df,
                remaining.sample(min(needed, len(remaining)), 
                random_state=random_state)
            ])
    
    return balanced_df.drop(columns=['_temp_id']).sample(frac=1, random_state=random_state)

# 11. Improved train/test split
def stratified_train_test_split(df, test_size=0.1, random_state=42):
    label_cols = [col for col in df.columns if col.startswith("cat_")]
    
    df['temp_pair_key'] = df.apply(lambda row: str(row['STITCH1']) + '_' + str(row['STITCH2']), axis=1)
    grouped = df.groupby('temp_pair_key')[label_cols].max()
    
    train_pairs = set()
    test_pairs = set()
    
    for col in label_cols:
        if grouped[col].sum() > 0:
            candidates = list(grouped[grouped[col] == 1].index)
            if candidates:
                random.shuffle(candidates)
                split_idx = int(len(candidates) * 0.8)
                train_pairs.update(candidates[:split_idx])
                test_pairs.update(candidates[split_idx:])
    
    remaining_pairs = [p for p in grouped.index if p not in train_pairs and p not in test_pairs]
    if remaining_pairs:
        random.shuffle(remaining_pairs)
        needed_train = int(len(grouped) * 0.8) - len(train_pairs)
        train_pairs.update(remaining_pairs[:needed_train])
        test_pairs.update(remaining_pairs[needed_train:])
    
    train_df = df[df['temp_pair_key'].isin(train_pairs)].drop(columns=['temp_pair_key'])
    test_df = df[df['temp_pair_key'].isin(test_pairs)].drop(columns=['temp_pair_key'])
    
    return train_df, test_df

# Apply splits with 80-20 ratio
print("\nSplitting data with 80-20 ratio...")
train_df, test_df = stratified_train_test_split(in_dist_df)

# Apply balanced sampling to training set
print("\nBalancing training samples...")
train_sample_size = len(train_df)
train_df = enhanced_balanced_sampling(train_df, train_sample_size)

# Keep all OOD samples that meet our criteria
ood_sample = ood_df.copy()
print(f"\nKeeping all {len(ood_sample)} OOD samples with at least one OOD interaction")

# 12. Analysis function
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
analyze_distribution(ood_sample, "OOD Sample")

"""
# 13. Create smaller versions of the datasets
def create_smaller_version(df, target_size, class_column):
    classes = df[class_column].unique()
    num_classes = len(classes)
    
    # Πόσα δείγματα θα πάρεις από κάθε κλάση
    samples_per_class = target_size // (num_classes )

    # Δείγμα με ισοκατανομή
    sampled_df = pd.concat([
        df[df[class_column] == cls].sample(n=min(samples_per_class, len(df[df[class_column] == cls])), random_state=42)
        for cls in classes
    ])

    # Αν περισσεύουν δείγματα (λόγω στρογγυλοποίησης), πάρε λίγα ακόμη
    remaining = target_size - len(sampled_df)
    if remaining > 0:
        available = df[~df.index.isin(sampled_df.index)]
        if len(available) >= remaining:
            extra = available.sample(n=remaining, random_state=42)
        else:
            print(f"[WARNING] Could only sample {len(available)} extra rows, needed {remaining}.")
            extra = available
        sampled_df = pd.concat([sampled_df, extra])
    return sampled_df
"""
def create_smaller_version(df, target_size, class_column):
    classes = df[class_column].unique()
    class_counts = df[class_column].value_counts()
    
    # Find the smallest available class
    min_class = class_counts.idxmin()
    min_count = class_counts.min()
    num_classes = len(classes)
    
    # Calculate how many samples you can take from each class
    # so as not to exceed the number of the smallest
    samples_per_class = min(target_size // num_classes, min_count)

    print(f"[INFO] Sampling {samples_per_class} examples per class (min class: '{min_class}' with {min_count} samples)")

    # Equal distribution with limit
    sampled_df = pd.concat([
        df[df[class_column] == cls].sample(n=min(samples_per_class, len(df[df[class_column] == cls])), random_state=42)
        for cls in classes
    ])

    # If there are excess samples, supplement them with extra samples from available classes
    remaining = target_size - len(sampled_df)
    if remaining > 0:
        available = df[~df.index.isin(sampled_df.index)]
        if len(available) >= remaining:
            extra = available.sample(n=remaining, random_state=42)
        else:
            print(f"[WARNING] Could only sample {len(available)} extra rows, needed {remaining}.")
            extra = available
        sampled_df = pd.concat([sampled_df, extra])

    return sampled_df.sample(frac=1, random_state=42).reset_index(drop=True)



# Target sizes (adjust these as needed)
small_train_size = 90000  
small_test_size = 2000    
small_ood_size = 72000    

# Create smaller datasets
small_train = create_smaller_version(train_df, small_train_size,'cat_0')
small_test = create_smaller_version(test_df, small_test_size,'cat_0')
small_ood = create_smaller_version(ood_sample, small_ood_size,'cat_0')

# Verify distributions
analyze_distribution(small_train, "Small Training")
analyze_distribution(small_test, "Small Test")
analyze_distribution(small_ood, "Small OOD")

# 14. Save datasets
print("\nSaving datasets...")
# Save full versions
train_df.to_csv('train_in_distribution.csv', index=False)
test_df.to_csv('test_in_distribution.csv', index=False)
ood_sample.to_csv('out_of_distribution.csv', index=False)

# Save smaller versions
small_train.to_csv('small_train_in_distribution.csv', index=False)
small_test.to_csv('small_test_in_distribution.csv', index=False)
small_ood.to_csv('small_out_of_distribution.csv', index=False)

# 15. Drug coverage analysis
def get_drugs(df):
    return set(df['STITCH1']).union(set(df['STITCH2']))

train_drugs = get_drugs(train_df)
test_drugs = get_drugs(test_df)
ood_drugs = get_drugs(ood_sample)

print("\nDrug coverage analysis:")
print(f"Unique drugs in training: {len(train_drugs)}")
print(f"Unique drugs in test only: {len(test_drugs - train_drugs)}")
print(f"Unique drugs in OOD only: {len(ood_drugs - train_drugs)}")
print(f"Missing embeddings for {len(missing_in + missing_ood)} unique drugs")

print("\nSmall dataset drug coverage:")
small_train_drugs = get_drugs(small_train)
small_test_drugs = get_drugs(small_test)
small_ood_drugs = get_drugs(small_ood)

print(f"Unique drugs in small training: {len(small_train_drugs)}")
print(f"Unique drugs in small test only: {len(small_test_drugs - small_train_drugs)}")
print(f"Unique drugs in small OOD only: {len(small_ood_drugs - small_train_drugs)}")

print("\nProcessing complete! Smaller datasets created successfully.")