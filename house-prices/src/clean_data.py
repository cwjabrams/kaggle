import pandas as pd
import numpy as np


# METHODS START ====================================================================
def categoryCode(dataframe):
    for col_name in dataframe.columns:
        if dataframe[col_name].dtype == 'object':
            categoryCodeSingle(dataframe, col_name)

def categoryCodeSingle(dataframe, col_name):
    dataframe[col_name] = dataframe[col_name].astype('category')
    dataframe[col_name] = dataframe[col_name].cat.codes


def cleanData(data):
    center(data)
    for i in range(len(data)):
        for j in range(len(data[i])):
            if np.isnan(data[i,j]) or np.isinf(data[i,j]):
                data[i, j] = np.nan_to_num(data[i,j]) 
    scp.normalize(data, norm='l2',axis=0) 

def normalize(data, norm, axis=0):
    scp.normalize(data, norm=norm,axis=axis) 

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

train_df = pd.read_csv('data/train.csv')
train_prices_df = train_df['SalePrice']
train_df = train_df.drop('SalePrice', axis=1)
test_df  = pd.read_csv('data/test.csv')
test_id_df = test_df['Id']

print(train_df)
print(test_df)

combined = pd.concat([train_df,test_df],axis=0, ignore_index=True)
print(combined)



garageYrBltAvg = int(combined['GarageYrBlt'].mean())
combined['GarageYrBlt'] = combined['GarageYrBlt'].fillna(garageYrBltAvg) 

#combined['MasVnrArea'] = combined['MasVnrArea'].apply(lambda x: 1 if x > 0 else 0)

cutoff_year = 1989
combined['YearBuilt'] = combined['YearBuilt'].apply(lambda x: 5000 if x > cutoff_year else 0)

avg_year_remodled = int(combined['YearRemodAdd'].mean())
combined['YearRemodAdd'] = combined['YearRemodAdd'].apply(lambda x: avg_year_remodled if x > avg_year_remodled else x)

categoryCodeSingle(combined, 'MSSubClass')
categoryCodeSingle(combined, 'YrSold')
categoryCode(combined)


"""
continuous_cols = list()
continuous_cols.append('OverallQual')
continuous_cols.append('OverallCond')

for col_name in combined.columns:
    if col_name.lower() in continuous_cols:
        makeContinuous(combined, col_name, 10)
"""

combined = combined.drop('LotArea', axis=1)
combined = combined.drop('LotFrontage', axis=1)
combined = combined.drop('LowQualFinSF', axis=1)
combined = combined.drop('MasVnrArea', axis=1)
combined = combined.drop('MiscVal', axis=1)
combined = combined.drop('MoSold', axis=1)
combined = combined.drop('OpenPorchSF', axis=1)
combined = combined.drop('PoolArea', axis=1)
combined = combined.drop('ScreenPorch', axis=1)
combined = combined.drop('3SsnPorch', axis=1)
combined = combined.drop('BsmtFinSF2', axis=1)
combined = combined.drop('BsmtFullBath', axis=1)
combined = combined.drop('BsmtHalfBath', axis=1)
combined = combined.drop('EnclosedPorch', axis=1)

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
train_prices = train_prices_df.to_numpy()
test_id = test_id_df.to_numpy()

train_file = open('data/train_coded_combined.npy', 'wb')
test_file = open('data/test_coded_combined.npy', 'wb')
train_prices_file = open('data/labels.npy', 'wb')
test_id_file = open('data/test_data_id.npy', 'wb')

np.save(train_file, train_data)
np.save(test_file, test_data)
np.save(train_prices_file, train_prices)
np.save(test_id_file, test_id)


