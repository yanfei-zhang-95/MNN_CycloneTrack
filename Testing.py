import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import Data_processing
import Add_layer
from sklearn.metrics import mean_squared_error

def testing(data, data_1, window_size):

    Y_pred = [1] * len(data)
    Y_true = [1] * len(data)

    table = np.asmatrix(data[0])
    (m, n) = np.shape(table)

    a = np.random.randn(1,n-2)
    b = np.random.randn(1,n-2)

    for i in range(0, len(data)):
        [X, Y] = Data_processing.data_processing(data[i], window_size)

        Y_true_tmp = Y
        Y_true_tmp = np.reshape(Y_true_tmp, (Y_true_tmp.shape[0], Y_true_tmp.shape[2]))
        Y_true[i] = Y_true_tmp

        Y_pred[i] = data_1[i]

        a = np.concatenate((a,Y_pred[i]),axis=0)
        b = np.concatenate((b,Y_true[i]), axis=0)

        print("Testing Set: i = ",i)

    a = np.asmatrix(a)
    a = a[1:,:]
    b = np.asmatrix(b)
    b = b[1:,:]

    score = mean_squared_error(a, b)
    print(score)

    return a,b,Y_true,Y_pred