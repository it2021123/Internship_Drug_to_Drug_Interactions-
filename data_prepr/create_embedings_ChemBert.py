# -*- coding: utf-8 -*-
"""
Create ChemBerta embeddings for drugs based on their names
Created on Mon Jun 23 15:26:44 2025

@author: giopo
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

# === Step 1: Load the CSV file containing drug interactions ===
df = pd.read_csv('bio-decagon-combo.csv')

# === Step 2: Extract all unique STITCH IDs from both columns ===
all_stitch_ids = pd.unique(df[['STITCH 1', 'STITCH 2']].values.ravel())

# === Step 3: Function to convert STITCH ID to PubChem CID ===
def stitch_to_cid(stitch_id):
    # Remove the 'CID' prefix and leading zeros to get the numeric CID
    return stitch_id[3:].lstrip("0")

# === Step 4: Retrieve Canonical SMILES string from PubChem using the CID ===
def get_smiles_from_stitch(stitch_id):
    cid = stitch_to_cid(stitch_id)
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES/JSON"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        return data['PropertyTable']['Properties'][0]['CanonicalSMILES']
    except Exception as e:
        print(f"[ERROR] {stitch_id} (CID: {cid}) -> {e}")
        return None

# === Step 5: Load the ChemBERTa model and tokenizer ===
tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

# === Step 6: Create embedding from SMILES string using ChemBERTa ===
def get_embedding(smiles):
    inputs = tokenizer(smiles, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    # Extract the embedding of the [CLS] token as the representation of the SMILES
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

# === Step 7: Retrieve embeddings for all valid SMILES strings ===
raw_embeddings = []
valid_ids = []
smiles_list = []

for i, stitch_id in enumerate(all_stitch_ids):
    smiles = get_smiles_from_stitch(stitch_id)
    if smiles:
        emb = get_embedding(smiles)
        raw_embeddings.append(emb)
        valid_ids.append(stitch_id)
        smiles_list.append(smiles)
        print(f"[{i+1}/{len(all_stitch_ids)}] {stitch_id} → {smiles}")
    else:
        print(f"[{i+1}/{len(all_stitch_ids)}] {stitch_id} → SMILES NOT found")
    time.sleep(0.5)  # Sleep to avoid overloading PubChem API

# === Step 8: Dimensionality reduction with PCA (from 768 to 128) ===
print("\nPerforming PCA dimensionality reduction...")
X = np.vstack(raw_embeddings)
pca = PCA(n_components=128)
X_reduced = pca.fit_transform(X)
print(f"Explained variance ratio retained: {pca.explained_variance_ratio_.sum():.4f}")

# === Step 9: Create a dictionary to store the results ===
embeddings = {}

for i, stitch_id in enumerate(valid_ids):
    embeddings[stitch_id] = {
        "smiles": smiles_list[i],
        "embedding": X_reduced[i].tolist()
    }

# === Step 10: Save the embeddings dictionary to a JSON file ===
with open("drug_embeddings_smiles.json", "w", encoding="utf-8") as f:
    json.dump(embeddings, f, indent=2, ensure_ascii=False)

print("\nFinished creating embeddings from SMILES and saved to 'drug_embeddings_smiles.json'")
