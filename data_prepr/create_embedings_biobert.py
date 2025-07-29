# -*- coding: utf-8 -*-
"""
Create BioBERT embeddings for drugs based on their names
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
    # Remove the "CID" prefix and leading zeros to get the numeric CID
    return stitch_id[3:].lstrip("0")

# === Step 4: Retrieve the common drug name from PubChem using the CID ===
def get_drug_name_from_stitch(stitch_id):
    cid = stitch_to_cid(stitch_id)
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/Title/JSON"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        return data['PropertyTable']['Properties'][0]['Title']
    except Exception as e:
        print(f"[ERROR] {stitch_id} (CID: {cid}) -> {e}")
        return None

# === Step 5: Load the BioBERT model and tokenizer ===
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

# === Step 6: Create embedding for a given text (drug name) using BioBERT ===
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    # Extract the embedding of the [CLS] token as the representation of the input text
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

# === Step 7: Retrieve embeddings for all valid drugs ===
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
        print(f"[{i+1}/{len(all_stitch_ids)}] {stitch_id} → Name NOT found")
    time.sleep(0.5)  # Sleep to avoid overloading PubChem API

# === Step 8: Dimensionality reduction with PCA (from 768 to 128 dimensions) ===
print("\nPerforming PCA dimensionality reduction...")
X = np.vstack(raw_embeddings)
pca = PCA(n_components=128)
X_reduced = pca.fit_transform(X)

# Check explained variance ratio (how much information is retained)
print(f"Explained variance ratio retained: {pca.explained_variance_ratio_.sum():.4f}")

# === Step 9: Create a dictionary with results ===
embeddings = {}

for i, stitch_id in enumerate(valid_ids):
    embeddings[stitch_id] = {
        "name": names[i],
        "embedding": X_reduced[i].tolist()
    }

# === Step 10: Save the embeddings dictionary to a JSON file ===
with open("drug_embeddings.json", "w", encoding="utf-8") as f:
    json.dump(embeddings, f, indent=2, ensure_ascii=False)

print("\nFinished creating embeddings and saved to 'drug_embeddings.json'")
