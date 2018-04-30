import tensorflow as tf

def add_layer(inputs, in_shape, out_shape, activation_function=None):
    W = tf.Variable(tf.random_normal([out_shape[1], in_shape[1]],stddev=1))
    V = tf.Variable(tf.random_normal([out_shape[2], in_shape[2]],stddev=1))
    biases = tf.Variable(tf.zeros([1, out_shape[1], out_shape[2]]))
    Wx = tf.tensordot(inputs, tf.transpose(W), axes=[[1], [0]])
    WxV = tf.tensordot(Wx, V, axes=[[1], [1]])
    WxV_plus_b = WxV + biases
    if activation_function is None:
        outputs = WxV_plus_b
    else:
        # WxV_plus_b = tf.nn.dropout(WxV_plus_b, keep_prob=0.999)
        outputs = activation_function(WxV_plus_b)
    return W, V, outputs