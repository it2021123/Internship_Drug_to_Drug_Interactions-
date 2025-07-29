import torch
import torch.nn as nn
import torch.nn.functional as F

#MLP - SEN MULTILABEL CLASSIFIER

class DrugInteractionDNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DrugInteractionDNN, self).__init__()

        self.hidden1 = nn.Linear(input_dim, 512)
        self.hidden2 = nn.Linear(512,512)
        self.hidden3 = nn.Linear(512,512)
        self.hidden4 = nn.Linear(512,512)
        self.hidden5 = nn.Linear(512,64)

        #batch norm for every Level
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(64)

        self.dropout = nn.Dropout(0.25)
        self.output = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.hidden1(x)))
        x = self.dropout(x)

        x = F.relu(self.bn2(self.hidden2(x)))
        x = self.dropout(x)

        x = F.relu(self.bn3(self.hidden3(x)))
        x = self.dropout(x)

        x = F.relu(self.bn4(self.hidden4(x)))
        x = self.dropout(x)

        x = F.relu(self.bn5(self.hidden5(x)))
        x = self.dropout(x)

        x = torch.sigmoid(self.output(x)) 
        
        return x
