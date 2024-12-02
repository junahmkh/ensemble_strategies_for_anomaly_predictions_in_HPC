import os
import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import MinMaxScaler
import pickle
import sys

inpts = sys.argv

rack = int(inpts[1])
f_w = int(inpts[2])

print(rack,f_w)

#setting up cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

def read_file(node_dir):
    node_data = pd.read_parquet(node_dir)
    node_data = node_data.dropna()
    return node_data.reset_index(drop=True)

def feature_extraction(df):
    df_feat = df.drop(columns=['new_label'])
    df_feat = df_feat.to_numpy()
    df_feat = torch.tensor(df_feat, dtype=torch.float)

    return df_feat

def labels_extraction(df):
    df_labels = df['new_label']
    df_labels = df_labels.to_numpy()
    df_labels = torch.tensor(df_labels, dtype=torch.int)

    return df_labels

print("Future prediction(hours) : ",f_w/4)

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


edges = []
for i in range(len(files)):
    temp = []
    if i == 0:
        temp.append([i,i+1])
    elif i == len(files)-1:
        temp.append([i,i-1])
    else:
        temp.append([i,i-1])
        temp.append([i,i+1])
    edges = edges + temp



edges = torch.tensor(edges, dtype=torch.long)
print(edges.t().contiguous())

with open(f"eligible_ts.pickle",'rb') as f:
    ts_list = pickle.load(f)

scaler = MinMaxScaler()

input_graphs = []
for idx,ts in enumerate(ts_list):
    # print(ts)
    DATA = pd.DataFrame()
    for file_path in files:
        tmp = file_path.split("/")
        tmp = tmp[-1].split(".")
        node = tmp[0]
        # print(node)

        with open(f"new_labels/{f_w}/{node}.pickle",'rb') as f:
            labels = pickle.load(f)
        # print(labels[:2],len(labels))

        ts_label = labels[labels['timestamp'] == ts]
        ts_label = ts_label.reset_index(drop = True)
        label = ts_label['new_label'][0]
        # print(label)

        data = read_file(file_path)
        # data = new_label_creation(data,f_w)
        data = data[data['timestamp'] == ts]
        data['new_label'] = label
        # print(data)

        DATA = pd.concat([DATA,data])
        DATA = DATA.drop('timestamp', axis=1)
        DATA = DATA.reset_index(drop=True)

        # print(DATA)

        # Fit and transform the data
        scaled_data = scaler.fit_transform(DATA.fillna(0))
        # Convert the scaled data back to a DataFrame
        scaled_df = pd.DataFrame(scaled_data, columns=DATA.columns)

    # print(scaled_df)

    converted_graph = Data(x=feature_extraction(scaled_df), edge_index=edges.t().contiguous(), y=labels_extraction(scaled_df))
    print(idx,converted_graph)
    input_graphs.append(converted_graph)

with open(f"gnn_graphs/{rack}/graphs_{rack}_{f_w}.pickle",'wb') as f:
    pickle.dump(input_graphs,f)