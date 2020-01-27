import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_avg_price_by_numeric(dataframe, col_name):
	x_values = list(dataframe[col_name])	
	y_values = list(dataframe['SalePrice'])
	zipped = list(zip(x_values, y_values))
	zipped_sorted = sorted(zipped, key = lambda x: x[0])
	x_values = [x[0] for x in zipped_sorted]	
	y_values = [x[1] for x in zipped_sorted]	

	fig, axis = plt.subplots()
	axis.plot(x_values, y_values, 'bo', markersize=1)
	fig.suptitle(col_name + " vs Sales Price")
	plt.savefig('images/data_price_graphs/' + col_name + '.png')
	plt.close()

def plot_avg_price_by_category(dataframe, col_name):
	plot_avg_by_category(dataframe, col_name, 'SalePrice')

def plot_avg_by_category(dataframe, cat_col_name, value_col_name):
	categories = dataframe[cat_col_name].astype(str)		 
	categories_set = categories.unique()
	values_arr = dataframe[value_col_name].to_numpy()
	
	avg_values = list()
	for category in categories_set:
		avg = get_avg_value(category, categories, values_arr)
		avg_values.append(avg)

	fig, axis = plt.subplots()
	axis.bar(list(categories_set), avg_values)
	fig.suptitle(cat_col_name + " vs Sales Price")
	plt.savefig('images/data_price_graphs/' + cat_col_name + '.png')
	plt.close()

def get_avg_value(category, categories, values_arr):
	avg = 0
	n = 0
	for i in range(len(categories)):
		if categories.iloc[i] == category:
			avg += values_arr[i]
			n += 1
	return avg/n

train_df = pd.read_csv('data/train.csv')

for col_name in train_df.columns:
	if train_df[col_name].dtype == 'object':
		plot_avg_price_by_category(train_df, col_name)
	elif train_df[col_name].dtype=='int64' or train_df[col_name].dtype=='float64':
		plot_avg_price_by_numeric(train_df, col_name)





