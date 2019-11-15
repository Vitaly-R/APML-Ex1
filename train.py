import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from model import mlp, conv_net


def train(model_fn, batch_size, learning_rate=None):
    """
    load FashionMNIST data.
    create model using model_fn, and train it on FashionMNIST.
    :param model_fn: a function to create the model (should be one of the functions from model.py)
    :param batch_size: the batch size for the training
    :param learning_rate: optional parameter - option to specify learning rate for the optimizer.
    :return:
    """
    # ---------- load data and pre-processing ----------
    ((x_train_np, y_train_np), (x_test_np, y_test_np)) = tf.keras.datasets.fashion_mnist.load_data()

    x_train_np = x_train_np - np.mean(x_train_np)
    x_train_np = x_train_np / np.std(x_train_np)
    training_inds = list(range(x_train_np.shape[0]))

    np.random.shuffle(training_inds)
    x_train_np = x_train_np[training_inds]
    y_train_np = y_train_np[training_inds]

    x_test_np = x_test_np - np.mean(x_test_np)
    x_test_np = x_test_np / np.std(x_test_np)
    test_inds = list(range(x_test_np.shape[0]))

    np.random.shuffle(test_inds)
    x_test_np = x_test_np[test_inds]
    y_test_np = y_test_np[test_inds]

    n_labels = len(np.unique(y_train_np))

    # ---------- set placeholders ----------
    x_ph = tf.placeholder(tf.float32, shape=[batch_size, 28, 28], name='X_input')
    y_ph = tf.placeholder(tf.int32, shape=[batch_size], name='Y_input')

    # ---------- define model ----------
    logits = model_fn(x_ph, n_labels)

    # ---------- define loss function and optimizer ----------
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_ph, logits=logits, name='loss')

    if learning_rate is None:
        optimizer = tf.train.AdamOptimizer().minimize(loss)
    else:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    # ---------- initializer ----------
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        # ---------- training ----------
        training_losses = list()
        training_accuracies = list()
        for j in range(x_train_np.shape[0] // batch_size):
            x_training_batch = x_train_np[j * batch_size: (j + 1) * batch_size]
            y_training_batch = y_train_np[j * batch_size: (j + 1) * batch_size]
            loss_val, _, training_logits = sess.run([loss, optimizer, logits], feed_dict={x_ph: x_training_batch, y_ph: y_training_batch})
            e_training_logits = np.exp(training_logits)
            for v in e_training_logits:
                v = v / np.sum(v)
            results = np.zeros(e_training_logits.shape[0]).astype(np.int32)
            for i in range(results.shape[0]):
                results[i] = np.argmax(e_training_logits[i])
            training_accuracies.append(len(results[results == y_training_batch]) / len(results))
            training_losses.append(np.mean(loss_val))

        # ---------- testing ----------
        test_losses = list()
        test_accuracies = list()
        for j in range(x_test_np.shape[0] // batch_size):
            x_test_batch = x_test_np[j * batch_size: (j + 1) * batch_size]
            y_test_batch = y_test_np[j * batch_size: (j + 1) * batch_size]
            logits_res = sess.run(logits, feed_dict={x_ph: x_test_batch, y_ph: y_test_batch})
            e_logits = np.exp(logits_res)
            for i in range(e_logits.shape[0]):
                e_logits[i] = e_logits[i] / np.sum(e_logits[i])
            results = np.zeros(e_logits.shape[0]).astype(np.int32)
            for i in range(results.shape[0]):
                results[i] = np.argmax(e_logits[i])
            misses = np.zeros(results.shape)
            misses[np.where(results != y_test_batch)] = 1
            test_losses.append(np.mean(misses))
            test_accuracies.append(len(results[results == y_test_batch]) / len(results))

    # ---------- plotting the graphs ----------
    plt.figure()
    plt.title('training loss values')
    plt.plot(training_losses)
    plt.figure()
    plt.title('training accuracies')
    plt.plot(training_accuracies)
    plt.figure()
    plt.title('test loss values')
    plt.plot(test_losses)
    plt.figure()
    plt.title('test accuracies')
    plt.plot(test_accuracies)

    plt.show()


def main():
    train(mlp, 64)
    # train(conv_net, 64)


if __name__ == "__main__":
    main()

