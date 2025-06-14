import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from DrugInteractionDataset import DrugInteractionDataset

class DrugInteractionDNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Args:
            input_dim (int): Διάσταση των input features (αυτόματα προσδιορίζεται από το Dataset)
            output_dim (int): Αριθμός κατηγοριών (αυτόματα προσδιορίζεται από το Dataset)
        """
        super(DrugInteractionDNN, self).__init__()
        
        # Ορισμός των κρυφών επιπέδων σύμφωνα με τις προδιαγραφές
        self.hidden1 = nn.Linear(input_dim, 1024)
        self.hidden2 = nn.Linear(1024, 1024)
        self.hidden3 = nn.Linear(1024, 1024)
        self.hidden4 = nn.Linear(1024, 1024)
        self.hidden5 = nn.Linear(1024, 256)
        
        # Επίπεδο εξόδου
        self.output = nn.Linear(256, output_dim)
        
        # Dropout για regularization
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        # Forward pass μέσω των κρυφών επιπέδων
        x = F.relu(self.hidden1(x))
        x = self.dropout(x)
        
        x = F.relu(self.hidden2(x))
        x = self.dropout(x)
        
        x = F.relu(self.hidden3(x))
        x = self.dropout(x)
        
        x = F.relu(self.hidden4(x))
        x = self.dropout(x)
        
        x = F.relu(self.hidden5(x))
      
        
        # Επίπεδο εξόδου με sigmoid ενεργοποίηση
        x = torch.sigmoid(self.output(x))
        
        return x


# Βελτιωμένη έκδοση που δέχεται το Dataset για αυτόματο προσδιορισμό διαστάσεων
def create_model_from_dataset(dataset):
    """
    Βοηθητική συνάρτηση για δημιουργία μοντέλου που ταιριάζει με το Dataset
    """
    # Πάρουμε ένα δείγμα για να προσδιορίσουμε τις διαστάσεις
    sample_features, sample_labels = dataset[0]
    
    input_dim = sample_features.shape[0]
    output_dim = sample_labels.shape[0]
    
    print(f"Αυτόματος προσδιορισμός διαστάσεων:")
    print(f"- Διάσταση εισόδου (input_dim): {input_dim}")
    print(f"- Διάσταση εξόδου (output_dim): {output_dim}")
    
    return DrugInteractionDNN(input_dim=input_dim, output_dim=output_dim)


# Παράδειγμα χρήσης:
if __name__ == "__main__":
    
    # Φόρτωση του dataset
    dataset = DrugInteractionDataset('C:/Users/giopo/OneDrive/Έγγραφα/Πρόβλεψη αλληλεπιδράσεων φαρμάκων/code/main/data preparation/train_in_distribution.csv', 'C:/Users/giopo/OneDrive/Έγγραφα/Πρόβλεψη αλληλεπιδράσεων φαρμάκων/code/main/data preparation/categories.csv')
    
    # Δημιουργία μοντέλου που ταιριάζει με το dataset
    model = create_model_from_dataset(dataset)
    
    # Δημιουργία DataLoader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Παράδειγμα batch
    batch_features, batch_labels = next(iter(dataloader))
    print(f"\nBatch features shape: {batch_features.shape}")
    print(f"Batch labels shape: {batch_labels.shape}")
    
    # Forward pass
    outputs = model(batch_features)
    print(f"Model outputs shape: {outputs.shape}")
    print(f"output: {outputs} ")