import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize


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
    w = tf.get_variable('weights', shape=[x.shape[1], 1], dtype=tf.float32)
    b = tf.get_variable('bias', shape=[1, 1], dtype=tf.float32)
    return tf.add(tf.matmul(x, w), b), [w, b]


def train(epochs, learning_rate, batch_size):
    """
    create linear regression using model() function from above and train it on boston houses dataset using batch-SGD.
    please normalize your data as a pre-processing step.
    please use your mse-loss implementation.
    :param epochs: number of epochs
    :param learning_rate: the learning rate of the SGD
    :param batch_size: size of learning batch
    :return: list contains the mean loss from each epoch.
    """
    x_data, y_data = load_data()
    inds = np.arange(x_data.shape[0])
    np.random.shuffle(inds)
    x_data = x_data[inds]
    y_data = y_data[inds]

    x_data = normalize(x_data)
    y_data = y_data - np.mean(y_data)
    y_data /= np.std(y_data)
    y_data = np.reshape(y_data, (y_data.shape[0], 1))

    X = tf.placeholder(tf.float32, shape=(batch_size, x_data.shape[1]), name='X')
    Y = tf.placeholder(tf.float32, shape=(batch_size, 1), name='Y')

    y_predicted, [weights, bias] = model(X)
    loss = tf.losses.mean_squared_error(labels=Y, predictions=y_predicted)

    gradients = tf.gradients(loss, [weights, bias])
    grad_w = gradients[0]
    grad_b = gradients[1]

    training_step_w = tf.assign(weights, weights - learning_rate * grad_w)
    training_step_b = tf.assign(bias, bias - learning_rate * grad_b)

    init = tf.global_variables_initializer()
    losses = list()
    with tf.Session() as sess:
        sess.run(init)
        for j in range(epochs):
            epoch_losses = list()
            for i in range(y_data.size // batch_size):
                x_batch = x_data[i * batch_size:(i + 1) * batch_size]
                y_batch = y_data[i * batch_size:(i + 1) * batch_size]
                _, _, loss_val = sess.run([training_step_w, training_step_b, loss], feed_dict={X: x_batch, Y: y_batch})
                epoch_losses.append(loss_val)
            losses.append(np.mean(epoch_losses))
    return losses


def main():
    losses = train(epochs=10000, learning_rate=0.1, batch_size=32)
    plt.figure()
    plt.title('Training loss for linear regression at each epoch')
    plt.xlabel('epoch')
    plt.ylabel('training loss')
    plt.plot(losses)
    plt.show()


if __name__ == "__main__":
    main()
