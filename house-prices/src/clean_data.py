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

def isNumeric(series):
	return series.dtype=='int64' or series.dtype=='float64'

def codeCategories(dataframe, cat_col_name, value_col_name):
	categories = dataframe[cat_col_name].astype(str)		 
	categories_set = categories.unique()
	values_arr = dataframe[value_col_name].to_numpy(dtype='float64')
	avg, std = np.mean(values_arr), np.std(values_arr)
	coef_variation = std/avg
	avg_values = list()
	category_to_avg = dict()
	for category in categories_set:
		avg = get_avg_value(category, categories, values_arr)
		avg_values.append(avg)
		category_to_avg[category] = avg
	categories = categories.to_numpy()
	if (coef_variation) < 1:
		category_codes = np.zeros(len(categories))
		for i in range(len(categories)):
			category_codes[i] = category_to_avg[categories[i]]
	else:
		category_codes = np.zeros(len(categories))
		for i in range(len(categories)):
			category_codes[i] = np.random()*std
	dataframe[cat_col_name] = pd.Series(category_codes, dtype='float64') 

def get_avg_value(category, categories, values_arr):
	avg = 0
	n = 0
	for i in range(len(categories)):
		if categories.iloc[i] == category:
			avg += values_arr[i]
			n += 1
	return avg/n
        


# METHODS END =========================================================================

train_df = pd.read_csv('data/train.csv')

test_df  = pd.read_csv('data/test.csv')
test_id_df = test_df['Id']

CORR_TOL = 0.2
for col_name in train_df.columns:
	if isNumeric(train_df[col_name]) and col_name != 'MSSubClass' and col_name != 'Id':
		corr_coef = pd.concat([train_df[col_name], train_df['SalePrice']], axis=1).corr(method='spearman').to_numpy()[0,1]
		if np.absolute(corr_coef) < CORR_TOL:
			train_df = train_df.drop(col_name, axis=1)
			test_df = test_df.drop(col_name, axis=1)
	elif train_df[col_name].dtype=='object' or col_name != 'Id':
		codeCategories(train_df, col_name, 'SalePrice')

train_prices_df = train_df['SalePrice']
train_df = train_df.drop('SalePrice', axis=1)

print(train_df)
print(test_df)

combined = pd.concat([train_df,test_df],axis=0, ignore_index=True)
print(combined)


'''
garageYrBltAvg = int(combined['GarageYrBlt'].mean())
combined['GarageYrBlt'] = combined['GarageYrBlt'].fillna(garageYrBltAvg) 

#combined['MasVnrArea'] = combined['MasVnrArea'].apply(lambda x: 1 if x > 0 else 0)

cutoff_year = 1989
combined['YearBuilt'] = combined['YearBuilt'].apply(lambda x: 5000 if x > cutoff_year else 0)

avg_year_remodled = int(combined['YearRemodAdd'].mean())
combined['YearRemodAdd'] = combined['YearRemodAdd'].apply(lambda x: avg_year_remodled if x > avg_year_remodled else x)
'''

#categoryCodeSingle(train_df, 'MSSubClass')
#categoryCode(combined)

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

np.save(train_file, train_data, allow_pickle=True)
np.save(test_file, test_data, allow_pickle=True)

np.save(train_prices_file, train_prices, allow_pickle=True)

np.save(test_id_file, test_id, allow_pickle=True)



