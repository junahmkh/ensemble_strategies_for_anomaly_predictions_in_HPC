import os
import pickle
import torch
import pandas as pd
from sklearn import preprocessing

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

with open("eligible_ts.pickle","rb") as f:
    ts_list = pickle.load(f)

for rack in racks:
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

    for filename in files:
        node = get_nodename(filename)

        df = pd.read_parquet(filename)
        df = df.fillna(0)

        df['value'] = df['value'].replace(2,1)
        df['value'] = df['value'].replace(3,1)

        label_save_path = f"processed_data/new_labels/{node}"
        os.makedirs(label_save_path, exist_ok=True)
        for f_w in f_ws:
            print(f"new_label - {f_w}")
            label_df = new_label_creation(df,f_w)
            ts_label = label_df['timestamp'].to_list()
            new_label = label_df['new_label'].to_list()

            for i_label in range(len(ts_label)):
                save_ts_label_path = os.path.join(label_save_path,f"{f_w}")
                os.makedirs(save_ts_label_path, exist_ok=True)
                with open(os.path.join(save_ts_label_path,f"{ts_label[i_label]}.pickle"),"wb") as f:
                    pickle.dump(new_label[i_label],f)

        ts_col = df['timestamp']

        df = df.drop(columns=['timestamp'])
        scaler = preprocessing.MinMaxScaler()
        names = df.columns
        d = scaler.fit_transform(df)
        df = pd.DataFrame(d, columns=names)

        df['timestamp'] = ts_col

        feats_save_path = f"processed_data/features/{node}"
        os.makedirs(feats_save_path, exist_ok=True)

        for idx,ts in enumerate(ts_list):
            print(f"ts - {rack} {node} {idx}")
            df_ts = df[df['timestamp'] == ts]
            #print(df_ts)
            df_ts = df_ts.drop(columns=['timestamp'])
            df_ts = df_ts.astype(float)
            feat_vector = feature_extraction(df_ts)

            # ts_feat_save_path = os.path.join(feats_save_path,f"{ts}.pickle")
            with open(os.path.join(feats_save_path,f"{idx}.pickle"),"wb") as f:
                pickle.dump(feat_vector,f)