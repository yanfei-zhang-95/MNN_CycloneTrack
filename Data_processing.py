import numpy as np

def data_processing(data_tmp, window_size):
    table = np.asmatrix(data_tmp)
    (m, n) = np.shape(table)
    if n == 4:
        table = table[:, [1, 2]]
    elif n == 3:
        table = table[:, [1]]

    n = n - 2
    mtx_3d = np.zeros(shape=(window_size, n, m - window_size))

    for j in range(window_size, m):
        window_tmp = np.zeros(shape=(window_size, n))
        window_tmp = np.asmatrix(window_tmp)
        for k in range(window_size - 1, -1, -1):
            window_tmp[window_size - k - 1, :] = table[j - k - 1, :]
        mtx_3d[:, :, j - window_size] = window_tmp

    X = np.zeros((m - window_size, window_size, n))
    Y = table[window_size:, [0,1]]
    Y = np.asarray(Y)
    Y = np.reshape(Y, (m - window_size, 1, 2))

    for p in range(0, m - window_size):
        X[p, :, :] = mtx_3d[:, :, p]

    return (X, Y)