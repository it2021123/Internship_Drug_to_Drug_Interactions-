# -*- coding: utf-8 -*-
"""
Δημιουργία BioBERT Embeddings για φάρμακα με βάση το όνομά τους
"""

import pandas as pd
import requests
import time
import torch
import pickle
import json
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
import numpy as np

# === Βήμα 1: Φόρτωσε το CSV αρχείο ===
df = pd.read_csv('C:/Users/giopo/OneDrive/Έγγραφα/Πρόβλεψη αλληλεπιδράσεων φαρμάκων/bio-decagon-combo.csv')

# === Βήμα 2: Συγκέντρωσε όλα τα μοναδικά STITCH IDs ===
all_stitch_ids = pd.unique(df[['STITCH 1', 'STITCH 2']].values.ravel())

# === Βήμα 3: Συνάρτηση μετατροπής STITCH ID σε CID ===
def stitch_to_cid(stitch_id):
    # Αφαιρεί το "CID" και τα αρχικά μηδενικά
    return stitch_id[3:].lstrip("0")

# === Βήμα 4: Ανάκτηση κοινού ονόματος φαρμάκου από το PubChem ===
def get_drug_name_from_stitch(stitch_id):
    cid = stitch_to_cid(stitch_id)
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/Title/JSON"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        return data['PropertyTable']['Properties'][0]['Title']
    except Exception as e:
        print(f"[ΣΦΑΛΜΑ] {stitch_id} (CID: {cid}) -> {e}")
        return None

# === Βήμα 5: Φόρτωση μοντέλου BioBERT ===
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

# === Βήμα 6: Δημιουργία embedding από το όνομα ===
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    # Παίρνουμε το CLS token
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

# === Βήμα 7: Ανάκτηση embeddings για όλα τα φάρμακα ===
raw_embeddings = []
valid_ids = []
names = []

for i, stitch_id in enumerate(all_stitch_ids):
    name = get_drug_name_from_stitch(stitch_id)
    if name:
        emb = get_embedding(name)
        raw_embeddings.append(emb)
        valid_ids.append(stitch_id)
        names.append(name)
        print(f"[{i+1}/{len(all_stitch_ids)}] {stitch_id} → {name}")
    else:
        print(f"[{i+1}/{len(all_stitch_ids)}] {stitch_id} → Όνομα ΔΕΝ βρέθηκε")
    time.sleep(0.5)  # για αποφυγή υπερφόρτωσης PubChem

# === Βήμα 8: Μείωση διαστάσεων με PCA (768 → 128 διαστάσεις) ===
print("\n Εκτελείται PCA...")
X = np.vstack(raw_embeddings)
pca = PCA(n_components=128)
X_reduced = pca.fit_transform(X)

# Έλεγχος ποσοστού διατηρούμενης διακύμανσης
print(f" Διατηρούμενη διακύμανση: {pca.explained_variance_ratio_.sum():.4f}")

# === Βήμα 9: Δημιουργία λεξικού αποτελεσμάτων ===
embeddings = {}

for i, stitch_id in enumerate(valid_ids):
    embeddings[stitch_id] = {
        "name": names[i],
        "embedding": X_reduced[i].tolist()
    }

# === Βήμα 10: Αποθήκευση σε JSON ===
with open("drug_embeddings.json", "w", encoding="utf-8") as f:
    json.dump(embeddings, f, indent=2, ensure_ascii=False)

print("\n Ολοκληρώθηκε η δημιουργία των embeddings και αποθηκεύτηκαν στο 'drug_embeddings.json'")
