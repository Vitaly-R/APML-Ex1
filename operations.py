import tensorflow as tf
import numpy as np


# ----------------- relu example --------------------

# the relu itself
def relu_numpy(x: np.ndarray):
    result = np.zeros_like(x)
    result[x > 0] = x[x > 0]
    return result


# the relu gradient
def relu_grad_numpy(x: np.ndarray, dy: np.ndarray):
    # x and y should have the same shapes.
    result = np.zeros_like(x)
    result[x > 0] = dy[x > 0]
    return result


# the relu tensorflow operation
# @tf.custom_gradient
def relu_tf(x):
    result = tf.py_func(relu_numpy, [x], tf.float32, name='my_relu_op')

    def grad(dy):
        return tf.py_func(relu_grad_numpy, [x, dy], tf.float32, name='my_relu_grad_op')

    return result, grad


# ----------------- batch matrix multiplication --------------
# a.shape = (n, k)
# b.shape = (k, m)
def matmul_numpy(a: np.ndarray, b: np.ndarray):
    result = np.dot(a, b)
    return result


# dy_dab.shape = (n, m)  ---- answer ----
# dy_da.shape = a.shape
# dy_db.shape = b.shape
def matmul_grad_numpy(a: np.ndarray, b: np.ndarray, dy_dab: np.ndarray):
    dy_da = np.dot(dy_dab, b.T)
    dy_db = np.dot(a.T, dy_dab)
    return [dy_da, dy_db]


# @tf.custom_gradient
def matmul_tf(a, b):
    result = tf.py_func(matmul_numpy, [a, b], tf.float32, name='my_matmul_op')

    def grad(dy_dab):
        return tf.py_func(matmul_grad_numpy, [a, b, dy_dab], tf.float32, name='my_matmul_grad_op')

    return result, grad


# ----------------- mse loss --------------
# y.shape = (batch)
# ypredict.shape = (batch)
# the result is a scalar
# dloss_dyPredict.shape = (batch)  ---- answer ----
def mse_numpy(y, ypredict):
    loss = np.mean(np.square(ypredict - y))
    return loss


def mse_grad_numpy(y, yPredict, dy):  # dy is gradient from next node in the graph, not the gradient of our y!
    dloss_dyPredict = 2*(y.shape**-1)*(yPredict - y)
    dloss_dy = dloss_dyPredict * dy  # @todo: check this, because it doesn't seem right...
    return [dloss_dy, dloss_dyPredict]


# @tf.custom_gradient
def mse_tf(y, y_predict):
    loss = tf.py_func(mse_numpy, [y, y_predict], tf.float32, name='my_mse_func')

    def grad(dy):
        return tf.py_func(mse_grad_numpy, [y, y_predict, dy], tf.float32, name='my_mse_grad_func')

    return loss, grad
