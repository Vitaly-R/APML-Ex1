import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import operations as ops


def load_data():
    from sklearn.datasets import load_boston
    boston_dataset = load_boston()
    X = np.array(boston_dataset.data)
    y = np.array(boston_dataset.target)
    return X, y


def model(x: tf.Tensor):
    """
    linear regression model: y_predict = W*x + b
    please use your matrix multiplication implementation.
    :param x: symbolic tensor with shape (batch, dim)
    :return:  a tuple that contains: 1.symbolic tensor y_predict, 2. list of the variables used in the model: [W, b]
                the result shape is (batch)
    """
    # YOUR CODE HERE
    w = None
    b = None
    y_predict = None
    return y_predict, [w, b]


def train(epochs, learning_rate, batch_size):
    """
    create linear regression using model() function from above and train it on boston houses dataset using batch-SGD.
    please normalize your data as a pre-processing step.
    please use your mse-loss implementation.
    :param epochs: number of epochs
    :param learning_rate: the learning rate of the SGD
    :return: list contains the mean loss from each epoch.
    """
    x_data, y_data = load_data()
    X = tf.placeholder(tf.float32, name='X')
    Y = tf.placeholder(tf.float32, name='Y')
    W = tf.get_variable('weights', initializer=tf.constant(0.0))
    b = tf.get_variable('bias', initializer=tf.constant(0.0))
    y_predict = tf.add(ops.matmul_tf(X, W), b)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for _ in range(epochs):
            #
            pass
    return []


def main():
    losses = train(50, 0.01, 32)
    plt.plot(losses)
    plt.show()


if __name__== "__main__":
  main()

