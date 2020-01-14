import numpy as np
import pandas as pd


train_data_df = pd.read_csv('data/kaggle_data/train_transaction.csv')
train_id_df = pd.read_csv('data/kaggle_data/train_identity.csv')

train_data_npy = train_data_df.to_numpy()
train_id_npy = train_id_df.to_numpy()

train_labels = train_data_npy[:,1]

np.delete(train_data_npy, 1, 0)


print(train_data_npy)
print(train_labels)


