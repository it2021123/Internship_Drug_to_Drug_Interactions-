# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 14:46:32 2025
Create Negative Samples

@author: giopo
"""
from itertools import combinations
import pandas as pd

# --- 1. Load files ---
df_effects = pd.read_csv('bio-decagon-combo.csv')
df_disease_class = pd.read_csv('known_effects.csv')

# --- 2. Merge based on Side Effect Name ---
df = df_effects.merge(df_disease_class, on='Side Effect Name', how='left')

# Save the merged dataset with side effects info
df.to_csv("my_positive_dataset_with_side_effect.csv", index=False)

# --- 3. Extract all unique drugs from both columns ---
all_drugs = pd.unique(df[['STITCH 1', 'STITCH 2']].values.ravel())

# --- 4. Generate all possible drug pairs (combinations of 2) ---
all_combinations = pd.DataFrame(list(combinations(all_drugs, 2)), columns=['STITCH 1', 'STITCH 2'])

# --- 5. Extract positive pairs (existing interactions) ---
positive_pairs = df[['STITCH 1', 'STITCH 2']].drop_duplicates()

# --- 6. Perform a left-anti join to find pairs NOT in positive samples ---
merged = all_combinations.merge(positive_pairs, on=['STITCH 1', 'STITCH 2'], how='left', indicator=True)
negative_samples = merged[merged['_merge'] == 'left_only'].drop('_merge', axis=1)

# --- 7. Label negative samples as 'No-Interaction' ---
negative_samples['Side Effect Name'] = 'No-Interaction'

# --- 8. Save the negative samples to CSV ---
negative_samples.to_csv('negative_samples.csv', index=False)
