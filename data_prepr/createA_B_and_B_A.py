
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  7 14:39:36 2025

@author: giopo
"""

import pandas as pd

print("Postive Samples")

df = pd.read_csv('my_positive_dataset_with_side_effect.csv')

# Κάνουμε τα ζεύγη αταξινόμητα ώστε (A, B) = (B, A)
unordered_pairs = df[['STITCH 1', 'STITCH 2']].apply(lambda x: frozenset(x), axis=1)

# Ελέγχουμε πόσα διαφορετικά "unordered" ζεύγη υπάρχουν
num_unordered = unordered_pairs.nunique()

# Ελέγχουμε πόσα αρχικά "ordered" ζεύγη υπάρχουν
num_ordered = df[['STITCH 1', 'STITCH 2']].drop_duplicates().shape[0]

print("Διαφορετικά ordered ζεύγη:", num_ordered)
print("Διαφορετικά unordered ζεύγη:", num_unordered)

if num_ordered > num_unordered:
    print("➡️ Το dataset περιέχει συμμετρικά ζεύγη (δηλαδή υπάρχουν και τα (A, B) και (B, A))")
else:
    print("✅ Δεν υπάρχουν συμμετρικά (A, B) και (B, A). Το dataset είναι συμμετρικό ή οι γραμμές είναι μοναδικές ως unordered.")



# Αντιστρέφουμε τις στήλες STITCH 1 και STITCH 2 για να πάρουμε (B, A)
reversed_df = df.rename(columns={'STITCH 1': 'STITCH 2', 'STITCH 2': 'STITCH 1'})

# Ενώνουμε το αρχικό και το αντεστραμμένο
df_with_reversed = pd.concat([df, reversed_df], ignore_index=True)

# (Προαιρετικά) αφαιρούμε τα duplicates
df_with_reversed = df_with_reversed.drop_duplicates()

# Αποθήκευση σε αρχείο
df_with_reversed.to_csv('dataset_with_both_AB_and_BA.csv', index=False)

print("✅ Το αρχείο αποθηκεύτηκε με όλα τα ζεύγη (A, B) και (B, A)")

#-------------------------------------------------------------------------------------------------------------------

print("Νegative Samples")

df = pd.read_csv('negative_samples.csv')

# Κάνουμε τα ζεύγη αταξινόμητα ώστε (A, B) = (B, A)
unordered_pairs = df[['STITCH 1', 'STITCH 2']].apply(lambda x: frozenset(x), axis=1)

# Ελέγχουμε πόσα διαφορετικά "unordered" ζεύγη υπάρχουν
num_unordered = unordered_pairs.nunique()

# Ελέγχουμε πόσα αρχικά "ordered" ζεύγη υπάρχουν
num_ordered = df[['STITCH 1', 'STITCH 2']].drop_duplicates().shape[0]

print("Διαφορετικά ordered ζεύγη:", num_ordered)
print("Διαφορετικά unordered ζεύγη:", num_unordered)

if num_ordered > num_unordered:
    print("➡️ Το dataset περιέχει συμμετρικά ζεύγη (δηλαδή υπάρχουν και τα (A, B) και (B, A))")
else:
    print("✅ Δεν υπάρχουν συμμετρικά (A, B) και (B, A). Το dataset είναι συμμετρικό ή οι γραμμές είναι μοναδικές ως unordered.")
# Αντιστρέφουμε τις στήλες STITCH 1 και STITCH 2 για να πάρουμε (B, A)
reversed_df = df.rename(columns={'STITCH 1': 'STITCH 2', 'STITCH 2': 'STITCH 1'})

# Ενώνουμε το αρχικό και το αντεστραμμένο
df_with_reversed = pd.concat([df, reversed_df], ignore_index=True)

# (Προαιρετικά) αφαιρούμε τα duplicates
df_with_reversed = df_with_reversed.drop_duplicates()

# Αποθήκευση σε αρχείο
df_with_reversed.to_csv('negative_dataset_with_both_AB_and_BA.csv', index=False)