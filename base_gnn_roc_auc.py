import os
import pandas as pd
import torch
import torchvision
from torchvision.transforms import ToTensor
import numpy as np
from sklearn import preprocessing
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
import sys
import pickle
from sklearn.metrics import roc_auc_score

#setting up cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class anomaly_anticipation(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        #encoder
        self.conv1 = GCNConv(in_channels, 300)
        self.conv2 = GCNConv(300, 100)
        self.conv3 = GCNConv(100, out_channels)

        #dense layer
        self.fc1 = torch.nn.Linear(out_channels,16)
        self.fc2 = torch.nn.Linear(16,1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        x = self.fc1(x)
        x = self.fc2(x)
        return x

roc_list = []

for f_w in [4,6,12,24,32,64,96,192,288]:
    pred_list = []
    y_true = []
    for rack in [10,22]:
        print(f"{rack},{f_w}")

        with open(f"gnn_graphs/{rack}/graphs_{rack}_{f_w}.pickle","rb") as f:
            graphs = pickle.load(f)

        model = anomaly_anticipation(417, 16)
        print(model)
        model.load_state_dict(torch.load(f"gnn_model/{f_w}/{rack}_{f_w}.pth"))
        model.eval()

        model = model.to(device)

        test_loader = DataLoader(graphs, shuffle = False)

        for d in test_loader:
            d = d.to(device)
            out = model(d.x,d.edge_index)
            pred = torch.sigmoid(out)
            pred_list.append(pred)
            y_true.append(d.y.detach().cpu().numpy())


    for i in range(len(pred_list)):
        pred_list[i] = pred_list[i].detach().cpu().numpy()
    y_true = [item for sublist in y_true for item in sublist]
    y_true = [int(item) for item in y_true]
    pred_list = [item for sublist in pred_list for item in sublist]
    pred_list = [float(item) for item in pred_list]
    len(y_true),len(pred_list)

    error_df = pd.DataFrame({'prob': pred_list,'true_class': y_true})

    roc = roc_auc_score(error_df.true_class, error_df.prob)

    print(f"roc: {roc}")
    roc_list.append(roc)

print(pd.DataFrame({"FW": [4,6,12,24,32,64,96,192,288],"AUC":roc_list}))
