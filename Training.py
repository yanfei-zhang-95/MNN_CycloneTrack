import tensorflow as tf
import numpy as np
import Data_processing
import Add_layer
import matplotlib.pyplot as plt

def training(data_1, data_2, window_size, first_hid, second_hid):
    # define placeholder for inputs to network
    xs = tf.placeholder(tf.float32, [None, window_size, 2])
    ys = tf.placeholder(tf.float32, [None, 1, 2])
    keep_prob = tf.placeholder(tf.float32)

    (W_1, V_1, l1) = Add_layer.add_layer(xs, [2, window_size,2], first_hid, activation_function=tf.nn.sigmoid)
    (W_2, V_2, l2) = Add_layer.add_layer(l1, first_hid, second_hid, activation_function=tf.nn.sigmoid)
    (W_3, V_3, prediction) = Add_layer.add_layer(l2, second_hid, [2,1,2], activation_function=None)

    loss = tf.reduce_mean(tf.square(ys - prediction))\
                + 0.02 * (tf.nn.l2_loss(W_1) + tf.nn.l2_loss(W_2) + tf.nn.l2_loss(W_3)+ tf.nn.l2_loss(V_1) + tf.nn.l2_loss(V_2) + tf.nn.l2_loss(V_3))
    train_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(loss)

    for i in range(0,len(data_1)):

        [X, Y] = Data_processing.data_processing(data_1[i], window_size)

        x_data = X
        y_data = Y

        if i == 0:
            init = tf.global_variables_initializer()
            sess = tf.Session()
            sess.run(init)
        else:
            saver.restore(sess, "./tmp/model")

        for j in range(1000):
            sess.run(train_step, feed_dict={xs: x_data, ys: y_data})

        # print('i = ', i, '\n', 'W_1 = \n', sess.run(W_1))
        program = [W_1, W_2, W_3, V_1, V_2, V_3]
        saver = tf.train.Saver(program)
        saver.save(sess, "./tmp/model")

        print("Training Set: i=",i)

    Y_pred = [1] * len(data_2)
    Y_true = [1] * len(data_2)

    table = np.asmatrix(data_2[0])
    (m, n) = np.shape(table)

    a = np.random.randn(1,n-2)
    b = np.random.randn(1,n-2)

    #Testing
    for i in range(0, len(data_2)):
        [X, Y] = Data_processing.data_processing(data_2[i], window_size)

        x_data = X

        Y_true_tmp = Y
        Y_true_tmp = np.reshape(Y_true_tmp, (Y_true_tmp.shape[0], Y_true_tmp.shape[2]))
        Y_true[i] = Y_true_tmp

        saver.restore(sess, "./tmp/model")
        # print('i = ', i, '\n', 'W_1', sess.run(W_1))

        Y_pred_tmp = sess.run(prediction, feed_dict={xs: x_data})
        Y_pred_tmp = np.reshape(Y_pred_tmp, (Y_pred_tmp.shape[0], Y_pred_tmp.shape[2]))
        Y_pred[i] = Y_pred_tmp

        a = np.concatenate((a,Y_pred[i]),axis=0)
        b = np.concatenate((b,Y_true[i]), axis=0)

        # print("Testing Set: ,i = ",i)

    a = np.asmatrix(a)
    a = a[1:,:]
    b = np.asmatrix(b)
    b = b[1:,:]

    return a,b,Y_true,Y_pred