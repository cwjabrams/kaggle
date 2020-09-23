import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn.preprocessing as scp
import sklearn.utils as scu 
import pickle
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K

@tf.function
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(K.log(y_pred) - K.log(y_true)))) 

def rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(K.log(y_pred) - K.log(y_true)))) 

@tf.function
def relu_mod(X):
    return K.clip(X, 0.00001, K.max(X))

def relu_shift(X):
	return K.relu(X) + 0.00001

def buildModel(data):

    initializer = keras.initializers.RandomNormal(mean=0.0, stddev=np.sqrt(1/len(data[0])), seed=None)

    output_layer1_size = int((2/3)*len(data[0]))

    model = keras.Sequential([
        layers.Dense(output_layer1_size, input_shape=[len(data[0])], activation='selu',
            activity_regularizer=regularizers.l2(0.01),kernel_initializer=initializer),
        layers.Dropout(0.05),
        layers.Dense(1, input_shape=[output_layer1_size+1], activation='relu'),
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.0084)
    #optimizer = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss='mean_squared_error',
            optimizer=optimizer,
            metrics=[rmse, 'mae'])

    return model

# Whatever we do to the training data we will need to do to
# the validationa and test data.
def cleanData(data):
    for i in range(len(data)):
        for j in range(len(data[i])):
            if np.isnan(data[i,j]) or np.isinf(data[i,j]):
                data[i, j] = np.nan_to_num(data[i,j]) 
    center(data)
    scp.normalize(data, norm='l2',axis=0) 

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

def addOnes(data):
    #Add column of all 1's to end of data matrix
    n,m = data.shape
    ones = np.ones((n,1))
    data = np.hstack((data, ones))

def main():
    data_file = open('data/train_coded_combined.npy', 'rb')
    labels_file = open('data/labels.npy', 'rb')
    test_data = open('data/test_coded_combined.npy', 'rb')
    test_id_file = open('data/test_data_id.npy', 'rb')
    data = np.load(data_file, allow_pickle=True)
    labels = np.load(labels_file, allow_pickle=True)
    test_data = np.load(test_data, allow_pickle=True)
    test_id = np.load(test_id_file, allow_pickle=True)

    results_list = list()
    
    num_models = 1
    num_epochs = 1000

    # Clean training data and test data
    train_data, train_labels = prepData(data, labels)
    cleanData(test_data)
    # Add a 1 to every training and test data point
    addOnes(train_data)
    addOnes(test_data)

    for i in range(num_models):
        
        model = buildModel(train_data)
        
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_rmse', patience=2000, verbose=1, restore_best_weights=True)

        history = model.fit(train_data, train_labels, shuffle=True,
                epochs=num_epochs, validation_split=0.2, verbose=1, callbacks=[early_stopping])

        # Plot training & validation loss values
        plt.plot(history.history['rmse'])
        plt.plot(history.history['val_rmse'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(fname='images/loss_' + str(i) + '.png')
        plt.clf()

        results = model.predict(test_data)
        results_list.append(results)

        

    results_arr = np.zeros((len(test_data), num_models))
    print(results_arr)
    for i in range(len(results_arr)):
            for j in range(len(results_arr[i])):
                results_arr[i,j] = results_list[j][i]
    print(results_arr)
    
    final_results = np.zeros(len(test_data))
    for i in range(len(test_data)):
        mean = np.mean(results_arr[i,:])
        final_results[i] = mean
    print(final_results)

    test_results = pd.DataFrame(columns=['Id', 'SalePrice'])
    test_results['Id'] = test_id
    test_results['SalePrice'] = results
    test_results.to_csv('test_results.csv', index=False)


if __name__=="__main__":
    main()

