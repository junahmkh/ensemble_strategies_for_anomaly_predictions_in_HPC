import pickle
import pandas as pd
import os
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import time

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
    df_labels = torch.tensor(df_labels, dtype=torch.int)

    return df_labels

with open("eligible_ts.pickle","rb") as f:
    ts_list = pickle.load(f)

#ts_list = ts_list[-8000:]
#print(len(ts_list))

rf_fw = []
dt_fw = []
svm_fw = []
lr_fw = []

truths_fw = []

for f_w in f_ws:
    count = 0

    rf_preds = []
    dt_preds = []
    svm_preds = []
    lr_preds = []

    ground_truths = []

    for rack in racks:
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

        for i,filename in enumerate(files):
            node = get_nodename(filename)
            #print(filename,rack,node,f_w)
            with open(f"ensemble_models/RF/{f_w}/{rack}_{node}.pickle","rb") as f:
                rf_model = pickle.load(f)
            with open(f"ensemble_models/DT/{f_w}/{rack}_{node}.pickle","rb") as f:
                dt_model = pickle.load(f)
            with open(f"ensemble_models/SVM/{f_w}/{rack}_{node}.pickle","rb") as f:
                svm_model = pickle.load(f)
            with open(f"ensemble_models/LR/{f_w}/{rack}_{node}.pickle","rb") as f:
                lr_model = pickle.load(f)

            df = pd.read_parquet(filename)
            df = df.dropna()
            df.reset_index(drop=True, inplace = True)
            df['value'] = df['value'].replace(2,1)
            df['value'] = df['value'].replace(3,1)
            #print(df.shape)
            #print(df['value'].value_counts())
            df = df.fillna(0)
            #print(df.shape)
            #print(df['value'].value_counts())
            df = new_label_creation(df,f_w)
            #print(df['new_label'].value_counts())
            #print(df.shape)
            #time.sleep(5)

            for idx,ts in enumerate(ts_list):
                print(f_w,rack,i,idx)
                df_ts = df[df['timestamp'] == ts]
                #print(df_ts)
                df_ts = df_ts.drop(columns=['timestamp'])
                df_ts = df_ts.astype(float)

                feat_vector = feature_extraction(df_ts)
                label = labels_extraction(df_ts)

                #print(label.item())
                if label.item()==1:
                    count = count + 1

                ground_truths.append(label.item())

                # Random Forest
                rf_probability = rf_model.predict_proba(feat_vector)[:, 1][0]
                rf_preds.append(rf_probability)

                # Support Vector Machine
                svm_probability = svm_model.predict_proba(feat_vector)[:, 1][0]
                svm_preds.append(svm_probability)

                # Decision Tree
                dt_probability = dt_model.predict_proba(feat_vector)[:, 1][0]
                dt_preds.append(dt_probability)

                # Logistic Regression
                lr_probability = lr_model.predict_proba(feat_vector)[:, 1][0]
                lr_preds.append(lr_probability)

                # Display Results
                #print("RF Probability:", rf_probability)

                #print("SVM Probability:", svm_probability)

                #print("DT Probability:", dt_probability)

                #print("LR Probability:", lr_probability)
                #print(len(rf_preds),len(ground_truths))
                #print(ground_truths[0].item())

    print(f"count : {count}")
    auc_rf = roc_auc_score(ground_truths, rf_preds)
    auc_dt = roc_auc_score(ground_truths, dt_preds)
    auc_svm = roc_auc_score(ground_truths, svm_preds)
    auc_lr = roc_auc_score(ground_truths, lr_preds)

    print(f"AUCs: RF = {auc_rf}, DT = {auc_dt}, SVM = {auc_svm}, LR = {auc_lr}")

    rf_fw.append(auc_rf)
    dt_fw.append(auc_dt)
    svm_fw.append(auc_svm)
    lr_fw.append(auc_lr)


print("RF:")
print(pd.DataFrame({"FW": f_ws,"AUC":rf_fw}))
print("DT:")
print(pd.DataFrame({"FW": f_ws,"AUC":dt_fw}))
print("SVM:")
print(pd.DataFrame({"FW": f_ws,"AUC":svm_fw}))
print("LR:")
print(pd.DataFrame({"FW": f_ws,"AUC":lr_fw}))