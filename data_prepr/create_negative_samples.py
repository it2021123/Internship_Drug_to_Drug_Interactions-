# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 14:46:32 2025

@author: giopo
"""
from itertools import combinations
import pandas as pd

# --- 1. Φόρτωση αρχείων ---
df_effects = pd.read_csv('C:/Users/giopo/OneDrive/Έγγραφα/Πρόβλεψη αλληλεπιδράσεων φαρμάκων/bio-decagon-combo.csv')
df_disease_class = pd.read_csv('known_effects.csv')

# --- 2. Join με βάση το Side Effect Name ---
df = df_effects.merge(df_disease_class, on='Side Effect Name', how='left')

df.to_csv("my_positive_dataset_with_side_effect.csv", index=False)

all_drugs = pd.unique(df[['STITCH 1', 'STITCH 2']].values.ravel())

all_combinations = pd.DataFrame(list(combinations(all_drugs, 2)), columns=['STITCH 1', 'STITCH 2'])

# Tο αρχικό σου df περιέχει τα θετικά δείγματα
positive_pairs = df[['STITCH 1', 'STITCH 2']].drop_duplicates()

# Κάνε left-anti join (δηλαδή κράτα μόνο τα ζεύγη που ΔΕΝ υπάρχουν ήδη)
merged = all_combinations.merge(positive_pairs, on=['STITCH 1', 'STITCH 2'], how='left', indicator=True)
negative_samples = merged[merged['_merge'] == 'left_only'].drop('_merge', axis=1)

negative_samples ['Side Effect Name'] = 'No-Interaction'
negative_samples.to_csv('negative_samples.csv', index=False)
