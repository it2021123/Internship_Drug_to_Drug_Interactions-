# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 18:21:38 2025

@author: giopo
"""

import pandas as pd


# --- 1. Load CSV files ---
df_effects = pd.read_csv('bio-decagon-combo.csv')
df_labels = pd.read_csv('bio-decagon-effectcategories.csv')

# --- 2. Prepare labels ---
df_labels.rename(columns={'Interaction Name': 'Side Effect Name'}, inplace=True)  # Rename column for consistency

# --- 3. Merge datasets and classify side effects ---
merged_all = df_effects[['Side Effect Name']].dropna().drop_duplicates().merge(df_labels, on='Side Effect Name', how='left')

# Split into known and unknown effects based on availability of Disease Class label
known_effects = merged_all[merged_all['Disease Class'].notna()]
unknown_effects = merged_all[merged_all['Disease Class'].isna()]

print(f"Known effects: {len(known_effects)}, Unknown effects: {len(unknown_effects)}")

# --- 4. Save known effects to CSV ---
known_effects.to_csv('known_effects.csv', index=False)
