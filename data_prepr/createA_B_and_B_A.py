# -*- coding: utf-8 -*-
"""
Created on Sat Jun  7 14:39:36 2025

@author: giopo
"""

import pandas as pd

print("Positive Samples")

# Load the positive samples dataset
df = pd.read_csv('my_positive_dataset_with_side_effect.csv')

# Create unordered pairs so that (A, B) equals (B, A)
unordered_pairs = df[['STITCH 1', 'STITCH 2']].apply(lambda x: frozenset(x), axis=1)

# Count how many unique unordered pairs exist
num_unordered = unordered_pairs.nunique()

# Count how many unique ordered pairs exist
num_ordered = df[['STITCH 1', 'STITCH 2']].drop_duplicates().shape[0]

print("Number of unique ordered pairs:", num_ordered)
print("Number of unique unordered pairs:", num_unordered)

if num_ordered > num_unordered:
    print("➡️ The dataset contains symmetric pairs (both (A, B) and (B, A) exist)")
else:
    print("✅ No symmetric pairs found. Dataset is either symmetric or unique as unordered pairs.")

# Reverse columns to get (B, A) pairs
reversed_df = df.rename(columns={'STITCH 1': 'STITCH 2', 'STITCH 2': 'STITCH 1'})

# Concatenate original and reversed pairs
df_with_reversed = pd.concat([df, reversed_df], ignore_index=True)

# (Optionally) drop duplicates after concatenation
df_with_reversed = df_with_reversed.drop_duplicates()

# Save the new dataset containing both (A, B) and (B, A) pairs
df_with_reversed.to_csv('dataset_with_both_AB_and_BA.csv', index=False)

print("✅ Saved file with all pairs (A, B) and (B, A)")

#-------------------------------------------------------------------------------------------------------------------

print("Negative Samples")

# Load the negative samples dataset
df = pd.read_csv('negative_samples.csv')

# Create unordered pairs so that (A, B) equals (B, A)
unordered_pairs = df[['STITCH 1', 'STITCH 2']].apply(lambda x: frozenset(x), axis=1)

# Count how many unique unordered pairs exist
num_unordered = unordered_pairs.nunique()

# Count how many unique ordered pairs exist
num_ordered = df[['STITCH 1', 'STITCH 2']].drop_duplicates().shape[0]

print("Number of unique ordered pairs:", num_ordered)
print("Number of unique unordered pairs:", num_unordered)

if num_ordered > num_unordered:
    print("➡️ The dataset contains symmetric pairs (both (A, B) and (B, A) exist)")
else:
    print("✅ No symmetric pairs found. Dataset is either symmetric or unique as unordered pairs.")

# Reverse columns to get (B, A) pairs
reversed_df = df.rename(columns={'STITCH 1': 'STITCH 2', 'STITCH 2': 'STITCH 1'})

# Concatenate original and reversed pairs
df_with_reversed = pd.concat([df, reversed_df], ignore_index=True)

# (Optionally) drop duplicates after concatenation
df_with_reversed = df_with_reversed.drop_duplicates()

# Save the new dataset containing both (A, B) and (B, A) pairs
df_with_reversed.to_csv('negative_dataset_with_both_AB_and_BA.csv', index=False)
