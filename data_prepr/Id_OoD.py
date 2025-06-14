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
import math  # για έλεγχο NaN

# 1. Φόρτωση δεδομένων
df_pos = pd.read_csv('dataset_with_both_AB_and_BA.csv')  # Θετικά δείγματα
df_neg = pd.read_csv('negative_dataset_with_both_AB_and_BA.csv')  # Αρνητικά δείγματα

# 2. Φόρτωση embeddings
with open('drug_embeddings.json', 'r') as f:
    embeddings = json.load(f)

# 3. Ορισμός OOD κατηγοριών (Out-of-Distribution)
ood_categories = [
    'psoriatic arthritis', 'monogenic disease', 'hypospadias',
    'chromosomal disease', 'polycystic ovary syndrome',
    'orofacial cleft', 'cryptorchidism'
]

# 4. Δημιουργία κλειδιού ζεύγους (pair key) για AB/BA ξεχωριστά
def create_pair_key(row):
    return (row['STITCH 1'], row['STITCH 2'])

df_pos['pair_key'] = df_pos.apply(create_pair_key, axis=1)
df_neg['pair_key'] = df_neg.apply(create_pair_key, axis=1)

# 5. Ομαδοποίηση θετικών αλληλεπιδράσεων
pos_interactions = defaultdict(list)
for _, row in df_pos.iterrows():
    pos_interactions[row['pair_key']].append(row['Disease Class'])

# 6. Κατηγορίες (θετικές + No-Interaction), εξαιρώντας τις OOD
all_categories = set(df_pos['Disease Class'].dropna().unique().tolist() + ['No-Interaction'])
print(f"all categories:{all_categories}")
print(f"oof categories:{ood_categories}")
categories = all_categories
cat_to_idx = {cat: idx for idx, cat in enumerate(categories)}
num_categories = len(categories)

# 7. Δημιουργία multi-hot encoding ανά ζεύγος
all_pairs = {}

# Θετικά δείγματα
for pair, interactions in pos_interactions.items():
    encoding = np.zeros(num_categories, dtype=np.int8)
    for interaction in interactions:
        # έλεγχος για NaN και OOD
        if (interaction is not None and 
            not (isinstance(interaction, float) and math.isnan(interaction)) and
            interaction not in ood_categories):
            encoding[cat_to_idx[interaction]] = 1
    all_pairs[pair] = {
        'encoding': encoding,
        'stitch1': pair[0],
        'stitch2': pair[1],
        'Disease Class': interactions
    }

# Αρνητικά δείγματα
for pair in df_neg['pair_key'].unique():
    if pair not in all_pairs:
        encoding = np.zeros(num_categories, dtype=np.int8)
        encoding[cat_to_idx['No-Interaction']] = 1
        all_pairs[pair] = {
            'encoding': encoding,
            'stitch1': pair[0],
            'stitch2': pair[1],
            'Disease Class': ['No-Interaction']
        }

# 8. Διαχωρισμός In-Distribution και OOD
in_distribution_pairs = {}
out_of_distribution_pairs = {}

for pair, data in all_pairs.items():
    disease_classes = data['Disease Class']
    if any(dc in ood_categories for dc in disease_classes):
        out_of_distribution_pairs[pair] = data
    else:
        in_distribution_pairs[pair] = data

# 9. Δημιουργία τελικών datasets
def create_final_dataset(pairs_dict):
    final_data = []
    missing_pairs = []

    for pair, data in pairs_dict.items():
        stitch1, stitch2 = pair
        if stitch1 in embeddings and stitch2 in embeddings:
            final_data.append({
                'STITCH1': stitch1,
                'STITCH2': stitch2,
                'embedding1': embeddings[stitch1]['embedding'],
                'embedding2': embeddings[stitch2]['embedding'],
                **{f'cat_{i}': data['encoding'][i] for i in range(num_categories)}
            })
        else:
            missing_pairs.append(pair)

    return pd.DataFrame(final_data), missing_pairs

in_dist_df, missing_in = create_final_dataset(in_distribution_pairs)
out_dist_df, missing_out = create_final_dataset(out_of_distribution_pairs)

print(f"Πλήθος in-distribution ζευγών: {len(in_dist_df)}")
print(f"Πλήθος out-of-distribution ζευγών: {len(out_dist_df)}")
print(f"Λείπουν embeddings για {len(missing_in)} in-dist ζεύγη και {len(missing_out)} OOD ζεύγη.")

# 10. Υπολογισμός μοναδικών φαρμάκων χωρίς embedding
missing_drugs = set()
for pair in missing_in + missing_out:
    for drug in pair:
        if drug not in embeddings:
            missing_drugs.add(drug)

print(f"Συνολικά λείπουν embeddings για {len(missing_drugs)} μοναδικά φάρμακα.")

# 11. Αποθήκευση αποτελεσμάτων
"""
in_dist_df.to_csv('in_distribution_data.csv', index=False)
out_dist_df.to_csv('out_of_distribution_data.csv', index=False)
pd.Series(categories).to_csv('categories.csv', index=False)
"""

# 12. Διαχωρισμός σε train/test μόνο για in-distribution
def pair_aware_split(df, test_size=0.2, random_state=42):
    df['pair_key'] = df.apply(lambda row: (row['STITCH1'], row['STITCH2']), axis=1)
    unique_pairs = df['pair_key'].unique()
    train_pairs, test_pairs = train_test_split(unique_pairs, test_size=test_size, random_state=random_state)
    train_df = df[df['pair_key'].isin(train_pairs)].copy()
    test_df = df[df['pair_key'].isin(test_pairs)].copy()
    train_df.drop(columns=['pair_key'], inplace=True)
    test_df.drop(columns=['pair_key'], inplace=True)
    return train_df, test_df

train_df, test_df = pair_aware_split(in_dist_df)

# 13. Δειγματοληψία για πιο γρήγορο πείραμα
train_sample_size = max(1, len(train_df) // 100)
test_sample_size = max(1, len(test_df) // 100)
ood_sample_size = max(1, len(out_dist_df) // 50)

train_df = train_df.sample(n=train_sample_size, random_state=42)
test_df = test_df.sample(n=test_sample_size, random_state=42)
out_dist_df = out_dist_df.sample(n=ood_sample_size, random_state=42)

# 14. Τελική αποθήκευση
train_df.to_csv('train_in_distribution.csv', index=False)
test_df.to_csv('test_in_distribution.csv', index=False)
out_dist_df.to_csv('out_of_distribution.csv', index=False)

# 15. Έλεγχος κάλυψης φαρμάκων μεταξύ sets
train_drugs = set(train_df['STITCH1']).union(set(train_df['STITCH2']))
test_drugs = set(test_df['STITCH1']).union(set(test_df['STITCH2']))
ood_drugs = set(out_dist_df['STITCH1']).union(set(out_dist_df['STITCH2']))

print(f"Train: {len(train_df)} ζευγάρια | Test: {len(test_df)} | OOD: {len(out_dist_df)}")
print("Φάρμακα μόνο στο test:", test_drugs - train_drugs)
print("Φάρμακα μόνο στο OOD:", ood_drugs - train_drugs)
