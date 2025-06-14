# Internship_Drug_Drug_Interactions

## Περιγραφή

Αυτό το project αφορά την πρόβλεψη αλληλεπιδράσεων μεταξύ φαρμάκων με τη χρήση νευρωνικών δικτύων, βασισμένο σε δημόσια διαθέσιμα δεδομένα από το [Decagon Dataset](https://snap.stanford.edu/decagon/). Χρησιμοποιούνται τεχνικές προεπεξεργασίας, δημιουργία embeddings, και διαχωρισμός δεδομένων σε in-distribution (train/test) και out-of-distribution (OoD) σύνολα, ώστε να εκπαιδευτεί και να αξιολογηθεί ένα πολυετικετικό μοντέλο.

---

## Αρχεία και Σειρά Εκτέλεσης

Απαιτούνται τα αρχεία:

- `bio-decagon-combo.csv`
- `bio-decagon-effectcategories.csv`

που κατεβάζονται από: [https://snap.stanford.edu/decagon/](https://snap.stanford.edu/decagon/)

Η σειρά εκτέλεσης των scripts είναι:

1. `data_prepr/create_labels.py`  
2. `data_prepr/create_negative_samples.py`  
3. `data_prepr/create_embeddings_biobert.py`  
4. `data_prepr/create_A_B_and_B_A.py`  
5. `data_prepr/Id_OoD.py`  
6. `DNN.py`  
7. `DrugInteractionDataset.py`  
8. `train.py`  

---

## Σημαντικές Λεπτομέρειες

- Οι 31 in-distribution κατηγορίες (seen) είναι διαχωρισμένες από out-of-distribution (OoD) κατηγορίες.
- Όλα τα φάρμακα στο train set υπάρχουν και στο test set, αντίστοιχα για OoD.
- Multi-label encoding εφαρμόζεται ώστε κάθε ζεύγος φαρμάκων με πολλαπλές αλληλεπιδράσεις να παραμένει στο ίδιο σύνολο (train ή test) για να αποφεύγεται διαρροή.
- Υλοποίηση oversampling και undersampling για ισορροπημένα δεδομένα στο `DrugInteractionDataset`.
- Το μοντέλο εκπαιδεύεται και αξιολογείται μόνο στα in-distribution δεδομένα.

---

## Οδηγίες Εκτέλεσης

1. Κατέβασε τα απαραίτητα αρχεία CSV από [Decagon Dataset](https://snap.stanford.edu/decagon/).
2. Εκτέλεσε τα scripts με τη σειρά που αναφέρεται για προετοιμασία των δεδομένων.
3. Εκπαίδευσε το μοντέλο μέσω του `train.py`.
4. Αξιολόγησε το μοντέλο σε validation και test σύνολα.

---

## Περιβάλλον και Εξαρτήσεις

- PyTorch
- Pandas
- NumPy
- Transformers (για BioBERT)
- scikit-learn
- tqdm

---

