import os
import pandas as pd
import torch
import numpy as np
from sklearn import preprocessing
import sys
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.metrics import roc_auc_score

inpts = sys.argv
rack = int(inpts[1])
f_w = int(inpts[2])

print(rack,f_w)

def read_file(node_dir):
    node_data = pd.read_parquet(node_dir)
    node_data = node_data.dropna()
    return node_data

def feature_and_new_label_extract(train_df,test_df):
    train_feat = train_df.drop(columns=['new_label'])
    train_feat = train_feat.to_numpy()
    train_feat = torch.tensor(train_feat, dtype=torch.float)

    test_feat = test_df.drop(columns=['new_label'])
    test_feat = test_feat.to_numpy()
    test_feat = torch.tensor(test_feat, dtype=torch.float)

    train_label = train_df['new_label']
    train_label = train_label.to_numpy()
    train_label = torch.tensor(train_label, dtype=torch.float)

    test_label = test_df['new_label']
    test_label = test_label.to_numpy()
    test_label = torch.tensor(test_label, dtype=torch.float)

    return train_feat,test_feat,train_label,test_label

def create_dataset(train_x,test_x,train_y,test_y):
    train_data = []
    for i in range(train_x.shape[0]):
        train_data.append([train_x[i],train_y[i]])
    test_data = []
    for i in range(test_x.shape[0]):
        test_data.append([test_x[i],test_y[i]])
    return train_data,test_data

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

def get_nodename(x):
  tmp = x.split('/')
  tmp = tmp[-1]
  tmp = tmp.split('.')
  return tmp[0]

print(f"Future prediction(hours) : {f_w/4}")

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

print(files)

for i in range(len(files)):
    DATA = read_file(files[i])
    DATA.reset_index(drop=True, inplace = True)
    DATA['value'] = DATA['value'].replace(2,1)
    DATA['value'] = DATA['value'].replace(3,1)

    DATA = new_label_creation(DATA,f_w)


    DATA.reset_index(drop=True, inplace = True)
    DATA = DATA.fillna(0)
    DATA = DATA.drop(columns=['timestamp'])
    DATA = DATA.astype(float)

    scaler = preprocessing.MinMaxScaler()
    names = DATA.columns
    d = scaler.fit_transform(DATA)
    DATA = pd.DataFrame(d, columns=names)

    print(DATA.shape)

    print("Before : ")
    print(DATA['value'].value_counts())
    print("After : ")
    print(DATA['new_label'].value_counts())

    train = DATA[:int(DATA.shape[0]*0.8)]
    train.reset_index(drop=True, inplace = True)
    test = DATA[int(DATA.shape[0]*0.8):]
    test.reset_index(drop=True, inplace = True)

    train_feat,test_feat,train_label,test_label = feature_and_new_label_extract(train,test)

    train_data,test_data = create_dataset(train_feat,test_feat,train_label,test_label)


    rf = RandomForestClassifier()
    rf.fit(train_feat, train_label)

    y_pred = rf.predict_proba(test_feat)[:, 1]
    error_df = pd.DataFrame({'prob': y_pred,'true_class': test_label})
    print(error_df)

    auc = roc_auc_score(error_df.true_class, error_df.prob)
    print(f"AUC = {auc}")

    os.makedirs(f"ensemble_models/RF/{f_w}", exist_ok=True)

    filename = 'ensemble_models/RF/{}/{}_{}.pickle'.format(f_w,rack,get_nodename(files[i]))

    with open(filename, 'wb') as f:
        pickle.dump(rf, f)