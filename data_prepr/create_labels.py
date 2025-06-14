# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 18:21:38 2025

@author: giopo
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Φόρτωση αρχείων ---
df_effects = pd.read_csv('C:/Users/giopo/OneDrive/Έγγραφα/Πρόβλεψη αλληλεπιδράσεων φαρμάκων/bio-decagon-combo.csv')
df_labels = pd.read_csv('C:/Users/giopo/OneDrive/Έγγραφα/Πρόβλεψη αλληλεπιδράσεων φαρμάκων/bio-decagon-effectcategories.csv')

# --- 2. Προετοιμασία labels ---
df_labels.rename(columns={'Interaction Name': 'Side Effect Name'}, inplace=True)

# --- 3. Join datasets και ταξινόμηση ---
merged_all = df_effects[['Side Effect Name']].dropna().drop_duplicates().merge(df_labels, on='Side Effect Name', how='left')

# Διαχωρισμός γνωστών / άγνωστων
known_effects = merged_all[merged_all['Disease Class'].notna()]
unknown_effects = merged_all[merged_all['Disease Class'].isna()]

print(f"Γνωστά: {len(known_effects)}, Άγνωστα: {len(unknown_effects)}")

#
# --- 4. Αποθήκευση των γνωστών ---
known_effects.to_csv('known_effects.csv', index=False)

