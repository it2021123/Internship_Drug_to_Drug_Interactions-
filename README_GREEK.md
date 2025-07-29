# 💊 Internship 2025 NSCR Demokritos: Drug-to-Drug Interactions (DDI) Classification & OoD Detection

Αυτό το project εστιάζει στην κατηγοριοποίηση αλληλεπιδράσεων μεταξύ φαρμάκων (Drug-Drug Interactions - DDI) με χρήση του dataset [Decagon](https://snap.stanford.edu/decagon/) και στον εντοπισμό Out-of-Distribution (OoD) δειγμάτων και στον τρόπο που βοηθάνε τα negative samples και SuperVised Contrastive Learning.

---

## Δομή Project
Drug_Interaction_Analysis/
### ├── README.md
### ├── .gitignore
### ├── DNN.py
### ├── DrugInteractionDataset.py
### ├── contrastive_learning_OoD.py
### ├── emb_contrastive.py
### ├── logit_contrastive.py
### ├── out_of_Distribution.py
### ├── train.py
### ├── best_model.pt
### ├── best_projection.pt
### ├── data_prepr/
### │   ├── Id_OoD.py
### │   ├── categories.csv
### │   ├── createA_B_and_B_A.py
### │   ├── create_embedings_ChemBert.py
### │   ├── create_embedings_biobert.py
### │   ├── create_labels.py
### │   ├── create_negative_samples.py
### │   ├── five_class_only.py
### │   ├── known_effects.csv


---

## 🧠 Περιγραφή Κύριων Αρχείων

### Μοντέλα & Εκπαίδευση
- **DrugInteractionDataset.py** : Κλασση Dataset για να φορτώνει τα embedings των δυο φαρμακων στο DNN
- **DNN.py**: Νευρωνικό δίκτυο με fully-connected layers για πρόβλεψη αλληλεπιδράσεων.
- **train.py**: Εκπαιδεύει το DNN με χρήση cross-entropy και αποθηκεύει το `best_model.pt`.
- **contrastive_learning_OoD.py**: Contrastive learning σε embeddings για ανίχνευση out-of-distribution δειγμάτων με Last Layer Έξοδο του DNN -Υπολογισμός AUROC, AUPR για OoD αναγνώριση.
- **emb_contrastive.py**: Εφαρμόζει contrastive loss στα embeddings από BioBERT ή ChemBERT -Υπολογισμός AUROC, AUPR για OoD αναγνώριση.
- **logit_contrastive.py**: Εφαρμόζει contrastive loss στις εξόδους (logits) του μοντέλου -Υπολογισμός AUROC, AUPR για OoD αναγνώριση.
- **out_of_Distribution.py**: Υπολογισμός AUROC, AUPR για OoD αναγνώριση χωρίς την χρήση Contrastive Learning - Ο κωδικάς είναι κατάλληλος για δοκιμή logit και Last Layer για είσοδο στο k-NN για τον Διαχωρισμό In Distribution ΚΑΙ Out Of Distribution.

### Δεδομένα
- **DrugInteractionDataset.py**: PyTorch Dataset class – φορτώνει embeddings, labels και OoD split.
- **data_prepr/**: Scripts για:
  - Δημιουργία θετικών και αρνητικών παραδειγμάτων (`create_negative_samples.py`)
  - Δημιουργία ετικετών (`create_labels.py`)
  - Χωρισμός σε ID / OoD (`Id_OoD.py` ,`five_class_only.py` *το οποίο δημιουργεί μικρότερα συνολά δεδομένων*)
  - Δημιουργία embeddings (`create_embedings_biobert.py`, `create_embedings_ChemBert.py`)

---

## ⚙️ Απαιτήσεις

- Python 3.8+  
- PyTorch  
- Transformers (HuggingFace)  
- scikit-learn, pandas, numpy  
- umap-learn, hdbscan (προαιρετικά για clustering & visualization)

```bash
pip install torch transformers scikit-learn pandas numpy umap-learn hdbscan matplotlib
```

## 🚀 Οδηγίες Εκτέλεσης

### 1. Προεπεξεργασία Δεδομένων

Εκτέλεσε τα παρακάτω scripts με τη σειρά για να προετοιμάσεις τα δεδομένα:

```bash
python data_prepr/createA_B_and_B_A.py
python data_prepr/create_negative_samples.py
python data_prepr/create_labels.py
python data_prepr/Id_OoD.py
ή
python data_prepr/five_class_only.py
```
## 2. Δημιουργία Embeddings

Τρέξε τα scripts που δημιουργούν embeddings με προεκπαιδευμένα μοντέλα BioBERT και ChemBERT:

```bash
python data_prepr/create_embedings_biobert.py
python data_prepr/create_embedings_ChemBert.py
```
### 3. Εκπαίδευση Επιβλεπόμενου Μοντέλου

Εκπαίδευσε το νευρωνικό δίκτυο για την πρόβλεψη αλληλεπιδράσεων φαρμάκων με την εκτέλεση του παρακάτω script:

```bash
python train.py
```
### 4. Αναλόγως τι θές να εκτελεσεις με ή χωρις Contrastive Learning 
####  4.1 Out of Distribution without Contrastive Learning (ΚΑΝΕ ΤΙΣ ΑΝΑΛΟΓΕΣ ΑΛΛΑΓΕΣ ΣΤΟΝ ΚΩΔΙΚΑ ΑΝ ΘΕΣ το Last Layer DNN ως Feature extractor)
```bash
python out_of_Distribution.py
```
####  4.2 Out of Distribution with Contrastive Learning -Mε  Last Layer DNN ως Feature extractor για την ανίχνευση OoD
```bash
python contrastive_learning_OoD.py
```
####  4.3 Out of Distribution with Contrastive Learning -Mε  Logit (sigmoid output of DNN) ως Feature extractor για την ανίχνευση OoD
```bash
python logit_contrastive.py
```
####  4.4 Out of Distribution with Contrastive Learning -Mε raw data των δαρμάκων για την ανίχνευση OoD
```bash
python emb_contrastive.py
```
---

## Δεδομένα που πρεπεί να κατεβούν πριν την εκτέλεση του κώδικα και να τοποθετηθούν στο data prep
- Προέρχονται από το [Decagon dataset (Stanford SNAP)](https://snap.stanford.edu/decagon/).

---
