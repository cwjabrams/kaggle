import pandas as pd
import numpy as np


# METHODS START ====================================================================
def categoryCode(dataframe):
    for col_name in dataframe.columns:
        if dataframe[col_name].dtype == 'object':
            dataframe[col_name] = dataframe[col_name].astype('category')
            dataframe[col_name] = dataframe[col_name].cat.codes

def cleanData(data):
    center(data)
    for i in range(len(data)):
        for j in range(len(data[i])):
            if np.isnan(data[i,j]) or np.isinf(data[i,j]):
                data[i, j] = np.nan_to_num(data[i,j]) 
    scp.normalize(data, norm='l1',axis=1) 

def normalize(data, norm, axis):
    scp.normalize(data, norm=norm,axis=1) 

def center(data):
    mean_data_point = get_mean_data_point(data)
    for index in range(len(data)):
        data[index] = data[index] - mean_data_point

def get_mean_data_point(data):
    mean_data_point = data[0]
    for index in range(1, len(data)):
        mean_data_point += data[index]
    mean_data_point = mean_data_point / (len(data))
    return mean_data_point

def prepData(data, labels):
    scu.shuffle(data, labels)
    cleanData(data)
    return data, labels

def makeContinuous(dataframe, col_name, scalar=1):
    column = dataframe[col_name].to_numpy()
    min_val = np.amin(column)
    max_val = np.amax(column)
    dataframe[col_name] = dataframe[col_name].apply(lambda x: scalar * np.absolute((x-min_val)/(max_val-min_val)))
    


# METHODS END =========================================================================

train_df = pd.read_csv('train.csv')
train_df = train_df.drop('SalePrice', axis=1)
test_df  = pd.read_csv('test.csv')

print(train_df)
print(test_df)

combined = pd.concat([train_df,test_df],axis=0, ignore_index=True)
print(combined)

continuous_cols = list()
continuous_cols.append("yrsold")
continuous_cols.append("grlivarea")
continuous_cols.append("1stflrsf")
continuous_cols.append("masvnrarea")
continuous_cols.append("garageyrblt")
continuous_cols.append("lotarea")

categoryCode(combined)
for col_name in combined.columns:
    if col_name.lower() in continuous_cols or "year" in col_name.lower():
        makeContinuous(combined, col_name, 10)

print(combined)

train_df = combined.iloc[:1460,:]
test_df = combined.iloc[1460:, :]

train_df = train_df.drop('Id', axis=1)
test_df = test_df.drop('Id', axis=1)

print(train_df)
print(test_df)

train_df.to_csv('data/combined_codefied.csv')

train_data = train_df.to_numpy()
test_data = test_df.to_numpy()

train_file = open('data/train_coded_combined.npy', 'wb')
test_file = open('data/test_coded_combined.npy', 'wb')

np.save(train_file, train_data)
np.save(test_file, test_data)

