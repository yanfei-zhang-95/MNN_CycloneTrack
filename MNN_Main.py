import scipy.io as sio
import numpy as np
from sklearn.metrics import mean_squared_error

import Training
import Testing

pred = np.load('./Y_pred_trueMNN.npy')

train = sio.loadmat('cytrack_train.mat')
train = train['cyclones_train']
train_final = train[0]

test = sio.loadmat('cytrack_test.mat')
test = test['cyclones_test']
test_final = test[0]

[a,b,Y_true,Y_pred] = Training.training(data_1=train_final,data_2=test_final,window_size=4,first_hid=[2,25,8],second_hid=[2,50,10])
# [a,b,Y_true,Y_pred] = Testing.testing(data = test_final, data_1 = pred,window_size=4)
score = mean_squared_error(a, b)
print(score)

np.save('Y_pred_trueMNN.npy',Y_pred)
np.save('Y_test.npy',Y_true)