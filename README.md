# 💊 Internship 2025 NSCR Demokritos: Drug-to-Drug Interactions (DDI) Classification & OoD Detection

### This project focuses on the classification of drug-drug interactions (DDI) using the [Decagon](https://snap.stanford.edu/decagon/) dataset and on the detection of Out-of-Distribution (OoD) samples and how negative samples and SuperVised Contrastive Learning help.

---

## Project Structure
Drug_Interaction_Analysis/
```bash
##### ├── README.md
##### ├── .gitignore
##### ├── DNN.py
##### ├── DrugInteractionDataset.py
##### ├── contrastive_learning_OoD.py
##### ├── emb_contrastive.py
##### ├── logit_contrastive.py
##### ├── out_of_Distribution.py
##### ├── train.py
##### ├── best_model.pt
##### ├── best_projection.pt
##### ├── data_prepr/
##### │      ├── Id_OoD.py
##### │      ├── categories.csv
##### │      ├── createA_B_and_B_A.py
##### │      ├── create_embedings_ChemBert.py
##### │      ├── create_embedings_biobert.py
##### │      ├── create_labels.py
##### │      ├── create_negative_samples.py
##### │      ├── five_class_only.py
##### │      ├── known_effects.csv
```

---

## 🧠 Models & Training

- **`DrugInteractionDataset.py`**: Custom `Dataset` class that loads embeddings of two drugs for the DNN input.
- **`DNN.py`**: Fully-connected neural network model for predicting drug interactions.
- **`train.py`**: Trains the DNN using cross-entropy loss and saves the best-performing model as `best_model.pt`.
- **`contrastive_learning_OoD.py`**: Applies contrastive learning to the last hidden layer of the DNN to detect OoD samples. Computes AUROC and AUPR scores.
- **`emb_contrastive.py`**: Applies contrastive loss to BioBERT or ChemBERT embeddings. Evaluates with AUROC and AUPR for OoD detection.
- **`logit_contrastive.py`**: Applies contrastive loss directly to the model's logits. Evaluates OoD performance with AUROC and AUPR.
- **`out_of_Distribution.py`**: Computes AUROC and AUPR for OoD detection without contrastive learning. Supports using either the logits or last hidden layer as features for k-NN distinction between In-Distribution (ID) and OoD samples.

---

## 📊 Data

- **`DrugInteractionDataset.py`**: PyTorch `Dataset` class that handles loading of embeddings, labels, and OoD split.
- **`data_prepr/`**: Folder containing data preparation scripts:
  - `create_negative_samples.py`: Generates negative (non-interacting) drug pairs.
  - `create_labels.py`: Generates labels for training/testing.
  - `Id_OoD.py`: Splits the dataset into In-Distribution and Out-of-Distribution sets.
  - `five_class_only.py`: Creates reduced-size datasets for faster experimentation.
  - `create_embedings_biobert.py`: Generates embeddings using BioBERT.
  - `create_embedings_ChemBert.py`: Generates embeddings using ChemBERT.

---

## ⚙️ Requirements

- Python 3.8+
- PyTorch
- HuggingFace Transformers
- scikit-learn
- pandas
- numpy
- umap-learn *(optional, for visualization)*
- hdbscan *(optional, for clustering)*

```bash
pip install torch transformers scikit-learn pandas numpy umap-learn hdbscan matplotlib
```

## 🚀 Execution Instructions

### 1. Data Preprocessing

Run the following scripts in order to prepare the data:

```bash
python data_prepr/createA_B_and_B_A.py
python data_prepr/create_negative_samples.py
python data_prepr/create_labels.py
python data_prepr/Id_OoD.py
ή
python data_prepr/five_class_only.py
```
## 2. Creating Embeddings

Run the scripts that create embeddings with pretrained BioBERT and ChemBERT models:
```bash
python data_prepr/create_embedings_biobert.py
python data_prepr/create_embedings_ChemBert.py
```
### 3. Supervised Model Training

Train the neural network to predict drug interactions by running the following script:

```bash
python train.py
```
### 4. Depending on what you want to do with or without Contrastive Learning
#### 4.1 Out of Distribution without Contrastive Learning (MAKE THE CORRESPONDING CHANGES TO THE CODE IF YOU USE Last Layer DNN as Feature extractor)
```bash
python out_of_Distribution.py
```
#### 4.2 Out of Distribution with Contrastive Learning -With Last Layer DNN as Feature extractor for OoD detection
```bash
python contrastive_learning_OoD.py
```
#### 4.3 Out of Distribution with Contrastive Learning -With Logit (sigmoid output of DNN) as Feature extractor for OoD detection
```bash
python logit_contrastive.py
```
#### 4.4 Out of Distribution with Contrastive Learning -With raw data of dharma for OoD detection
```bash
python emb_contrastive.py
```
---

## Data that must be downloaded before running the code and placed in data preparation
- Comes from the [Decagon dataset (Stanford SNAP)](https://snap.stanford.edu/decagon/).

---
