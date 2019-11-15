import tensorflow as tf


def mlp(x: tf.Tensor, nlabels):
    """
    multi layer perceptrone: x -> linear > relu > linear.
    :param x: symbolic tensor with shape (batch, 28, 28)
    :param nlabels: the dimension of the output.
    :return: a symbolic tensor, the result of the mlp, with shape (batch, nlabels). the model return logits (before softmax).
    """
    # ---------- defining variables ----------
    w1 = tf.get_variable('weights1', shape=[100, x.shape[1] * x.shape[2]], dtype=tf.float32, trainable=True)
    b1 = tf.get_variable('bias1', shape=[100, 1], dtype=tf.float32, trainable=True)
    w2 = tf.get_variable('weights2', shape=[nlabels, 100], dtype=tf.float32, trainable=True)
    b2 = tf.get_variable('bias2', shape=[nlabels, 1], dtype=tf.float32, trainable=True)

    # --------- reshaping ------------
    x_flattened = tf.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))
    x_reshaped = tf.cast(tf.transpose(x_flattened), tf.float32)

    # ---------- first linear layer ----------
    linear1 = tf.add(tf.matmul(w1, x_reshaped, name='matmul_1'), b1, name='linear_1')

    # ---------- ReLU ----------
    relu_res = tf.nn.relu(linear1, name='relu')

    # ---------- second linear layer ----------
    result = tf.transpose(tf.add(tf.matmul(w2, relu_res, name='matmul_2'), b2, name='linear_2'))
    return result


def conv_net(x: tf.Tensor, nlabels):
    """
    convnet.
    in the convolution use 3x3 filteres with 1x1 strides, 20 filters each time.
    in the  maxpool use 2x2 pooling.
    :param x: symbolic tensor with shape (batch, 28, 28)
    :param nlabels: the dimension of the output.
    :return: a symbolic tensor, the result of the mlp, with shape (batch, nlabels). the model return logits (before softmax).
    """
    # YOUR CODE HERE
    pass
