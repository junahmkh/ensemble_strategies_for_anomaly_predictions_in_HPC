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
import time
import pickle
from sklearn.metrics import roc_auc_score
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

f_ws = [4,6,12,24,32,64,96,192,288]
racks = [10,22]

def get_nodename(x):
  tmp = x.split('/')
  tmp = tmp[-1]
  tmp = tmp.split('.')
  return tmp[0]

def new_label_creation(df: pd.DataFrame, t_n: int) -> pd.DataFrame:
  """
  Create new_labels for anomaly anticipation for future window (t_n). The algorithm looks
  for an anomaly in the next t_n timesteps. The current timestep is said to be an anomaly
  if atleast one anomaly ahead in the future window (t_n)
  """
  value = df['value'].to_numpy()
  new_label = []
  for i in range(len(value)):
      anomaly_ahead = False
      for j in range(i+1,i+1+t_n):
          if(j>=len(value)):
              break
          else:
              if(value[j]==1):
                  anomaly_ahead = True
                  break
      if(anomaly_ahead):
          new_label.append(1)
      else:
          new_label.append(0)
  df['new_label'] = new_label
  return df

def feature_extraction(df):
    df_feat = df.drop(columns=['new_label'])
    df_feat = df_feat.to_numpy()
    df_feat = torch.tensor(df_feat, dtype=torch.float)

    return df_feat

def labels_extraction(df):
    df_labels = df['new_label']
    df_labels = df_labels.to_numpy()
    #df_labels = torch.tensor(df_labels, dtype=torch.int)

    return df_labels

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

class baseline_1(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        #encoder
        self.fc1 = torch.nn.Linear(in_channels, 300)
        self.fc2 = torch.nn.Linear(300, 100)
        self.fc3 = torch.nn.Linear(100, 16)

        #output
        self.fc4 = torch.nn.Linear(16,1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

with open("eligible_ts.pickle","rb") as f:
    ts_list = pickle.load(f)

racks_nodes = {10:[200,201,202,203,204,209,210,211,213,214,215,217,219],
               22:[440,444,445,446,447,449,450,451,453,454,455,456,457,459]}

# idx = racks_nodes[10].index(204)

auc_fw = []

for f_w in f_ws:
    soft_probs = []
    ground_truths = []

    for rack in racks:
        with open(f"gnn_graphs/{rack}/graphs_{rack}_{f_w}.pickle","rb") as f:
            graphs = pickle.load(f)

        gnn_model = anomaly_anticipation(417, 16)
        print(gnn_model)
        gnn_model.load_state_dict(torch.load(f"gnn_model/{f_w}/{rack}_{f_w}.pth"))
        gnn_model.eval()

        gnn_model = gnn_model.to(device)

        gnn_pred_list = []

        #reading files for all the nodes in a rack
        dir_path = 'data/{}/'.format(rack)

        files = []
        # loop over the contents of the directory
        for filename in os.listdir(dir_path):
            # construct the full path of the file
            file_path = os.path.join(dir_path, filename)
            # check if the file is a regular file (not a directory)
            if os.path.isfile(file_path):
                files.append(file_path)

        for idx,ts in enumerate(ts_list):
            print(f_w,rack,idx)
            g = graphs[idx]

            g = g.to(device)
            out = gnn_model(g.x,g.edge_index)
            gnn_pred = torch.sigmoid(out)

            # print(gnn_pred)
            for i in range(len(gnn_pred)):
                gnn_pred_list.append(gnn_pred[i].item())
            # print(pred_list,type(pred_list))

            # print(gnn_pred_list)
            # time.sleep(5)
            # print(pred[0].item())
            # exit()

            for i,filename in enumerate(files):
                node = get_nodename(filename)

                # print(rack,node)
                gnn_idx = racks_nodes[rack].index(int(node))

                gnn_prob = gnn_pred_list[gnn_idx]

                # print(racks_nodes[10].index(214))
                # print(node,gnn_idx)

                # print(gnn_pred_list[gnn_idx])
                # time.sleep(5)
                #print(filename,rack,node,f_w)

                with open(f"ensemble_models/RF/{f_w}/{rack}_{node}.pickle","rb") as f:
                    rf_model = pickle.load(f)
                with open(f"ensemble_models/DT/{f_w}/{rack}_{node}.pickle","rb") as f:
                    dt_model = pickle.load(f)
                with open(f"ensemble_models/SVM/{f_w}/{rack}_{node}.pickle","rb") as f:
                    svm_model = pickle.load(f)
                with open(f"ensemble_models/LR/{f_w}/{rack}_{node}.pickle","rb") as f:
                    lr_model = pickle.load(f)


                with open(f"processed_data/features/{node}/{idx}.pickle","rb") as f:
                    feat_vector = pickle.load(f)

                # print(feat_vector)
                with open(f"processed_data/new_labels/{node}/{f_w}/{ts}.pickle","rb") as f:
                    label = pickle.load(f)
                # print(label)

                # time.sleep(5)
                # df = pd.read_parquet(filename)
                # df = df.fillna(0)
                # df = new_label_creation(df,f_w)

                # ts_col = df['timestamp']

                # df = df.drop(columns=['timestamp'])
                # scaler = preprocessing.MinMaxScaler()
                # names = df.columns
                # d = scaler.fit_transform(df)
                # df = pd.DataFrame(d, columns=names)

                # df['timestamp'] = ts_col

                # df_ts = df[df['timestamp'] == ts]
                # #print(df_ts)
                # df_ts = df_ts.drop(columns=['timestamp'])
                # df_ts = df_ts.astype(float)

                # feat_vector = feature_extraction(df_ts)
                # label = labels_extraction(df_ts)

                ground_truths.append(label)

                # in_channels = len(feat_vector[0])

                # dnn_model = baseline_1(in_channels)
                # dnn_model = dnn_model.to(device)

                # dnn_model.load_state_dict(torch.load(f"ensemble_models/DNN/{f_w}/{rack}_{node}.pickle"))
                # dnn_model.eval()

                # x = feat_vector.to(device)
                # out_dnn = dnn_model(x)
                # dnn_prob = torch.sigmoid(out_dnn)
                # dnn_prob = dnn_prob.item()
                # print(f"DNN: {dnn_prob}")

                # Random Forest
                rf_prob = rf_model.predict_proba(feat_vector)[:, 1]
                # rf_preds.append(rf_probability)
                # print(f"RF: {rf_prob[0]}")

                # Support Vector Machine
                svm_prob = svm_model.predict_proba(feat_vector)[:, 1]
                # svm_preds.append(svm_probability)
                # print(f"SVM: {svm_prob[0]}")

                # Decision Tree
                dt_prob = dt_model.predict_proba(feat_vector)[:, 1]
                # dt_preds.append(dt_probability)
                # print(f"DT: {dt_prob[0]}")

                # Logistic Regression
                lr_prob = lr_model.predict_proba(feat_vector)[:, 1]
                # lr_preds.append(lr_probability)
                # print(f"LR: {lr_prob[0]}")

                # 1. **Soft Voting:** Average the probabilities
                soft_voting_probs = (gnn_prob + rf_prob + svm_prob + dt_prob + lr_prob) / 5
                soft_probs.append(soft_voting_probs)

    print(ground_truths)
    print(soft_probs)
    # Evaluate ROC-AUC for soft voting
    roc_auc_soft = roc_auc_score(ground_truths, soft_probs)
    print("Soft Voting ROC-AUC:", roc_auc_soft)

    auc_fw .append(roc_auc_soft)

print("Results:")
print(pd.DataFrame({"FW": f_ws,"AUC":auc_fw}))