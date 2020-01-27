import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
		






