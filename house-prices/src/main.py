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


def buildModel(data):
    model = keras.Sequential([
        layers.Dense(44, input_shape=[len(data[0])], activation='elu'),
        layers.Dense(1, input_shape=[45], activation='relu'),
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
            optimizer=optimizer,
            metrics=['mae', 'mse', 'acc'])

    return model

# Whatever we do to the training data we will need to do to
# the validationa and test data.
def cleanData(data):
    for i in range(len(data)):
        for j in range(len(data[i])):
            if np.isnan(data[i,j]) or np.isinf(data[i,j]):
                data[i, j] = np.nan_to_num(data[i,j]) 
    center(data)
    scp.normalize(data, norm='l1',axis=1) 

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
    data_file = open('data/data_points.npy', 'rb')
    labels_file = open('data/labels.npy', 'rb')
    test_data = open('data/test_data.npy', 'rb')
    test_id_file = open('data/test_data_id.npy', 'rb')
    data = np.load(data_file)
    labels = np.load(labels_file)
    test_data = np.load(test_data)
    test_id = np.load(test_id_file)

    
    results_list = list()
    
    num_models = 10 
    num_epochs = 3000

    # Clean training data and test data
    train_data, train_labels = prepData(data, labels)
    cleanData(test_data)
    # Add a 1 to every training and test data point
    addOnes(train_data)
    addOnes(test_data)

    for i in range(num_models):
        
        model = buildModel(train_data)
        
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_mse', patience=400, verbose=1, restore_best_weights=True)

        history = model.fit(train_data, train_labels, shuffle=True,
                epochs=num_epochs, validation_split=0.2, verbose=1, callbacks=[early_stopping])

        # Plot training & validation loss values
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(fname='images/acc_' + str(i) + '.png')
        plt.clf()


        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
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

