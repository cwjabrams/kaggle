import pandas as pd
import numpy as np




def categoryCode(dataframe):
    for col_name in dataframe.columns:
        if dataframe[col_name].dtype == 'object':
            dataframe[col_name] = dataframe[col_name].astype('category')
            dataframe[col_name] = dataframe[col_name].cat.codes

train_df = pd.read_csv('train.csv')
train_df = train_df.drop('SalePrice', axis=1)
test_df  = pd.read_csv('test.csv')

print(train_df)
print(test_df)

combined = pd.concat([train_df,test_df],axis=0, ignore_index=True)
print(combined)

categoryCode(combined)
print(combined)

train_df = combined.iloc[:1460,:]
test_df = combined.iloc[1460:, :]

print(train_df)
print(test_df)


train_df = train_df.drop('Id', axis=1)
test_df = test_df.drop('Id', axis=1)

train_df.to_csv('data/combined_codefied.csv')

train_data = train_df.to_numpy()
test_data = test_df.to_numpy()

train_file = open('data/train_coded_combined.npy', 'wb')
test_file = open('data/test_coded_combined.npy', 'wb')

np.save(train_file, train_data)
np.save(test_file, test_data)

