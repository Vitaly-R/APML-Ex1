import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from model import mlp, conv_net


np.random.seed(0)
tf.random.set_random_seed(0)


def normalize(data):
    res = data - np.mean(data)
    res = res / np.std(res)
    return res


def batch_dataset(data: np.ndarray, labels: np.ndarray, batch_size):
    inds = np.arange(data.shape[0])
    np.random.shuffle(inds)
    s_data = data[inds]
    s_labels = labels[inds]
    split_inds = [(i + 1) * batch_size for i in range(data.shape[0] // batch_size)]
    data_batches = np.array_split(s_data, split_inds)
    labels_batches = np.array_split(s_labels, split_inds)
    if data_batches[-1].shape[0] < batch_size:
        data_batches.pop(-1)
        labels_batches.pop(-1)
    return data_batches, labels_batches


def load_datasets(batch_size):
    ((x_train_np, y_train_np), (x_test_np, y_test_np)) = tf.keras.datasets.fashion_mnist.load_data()
    n_x_train = normalize(x_train_np)
    n_x_test = normalize(x_test_np)

    x_train, y_train = batch_dataset(n_x_train, y_train_np, batch_size)
    x_test, y_test = batch_dataset(n_x_test, y_test_np, batch_size)

    return x_train, y_train, x_test, y_test, len(np.unique(y_train_np))


def calculate_accuracy(logits, labels):
    s_logits = tf.nn.softmax(logits=logits, axis=1).eval()
    results = np.zeros(s_logits.shape[0]).astype(np.int32)
    for i in range(results.shape[0]):
        results[i] = np.argmax(s_logits[i])
    return round(100 * len(results[results == labels]) / len(results), 5)


def plot_graph(to_plot, title):
    plt.figure()
    plt.title(title)
    plt.plot(to_plot)


def train(model_fn, batch_size, learning_rate=None, epochs=10, regularize=False):
    """
    load FashionMNIST data.
    create model using model_fn, and train it on FashionMNIST.
    :param model_fn: a function to create the model (should be one of the functions from model.py)
    :param batch_size: the batch size for the training
    :param learning_rate: optional parameter - option to specify learning rate for the optimizer.
    :param epochs: optional parameter - number of epochs to train the model.
    :param regularize: optional parameter - option to train the model with regularization.
    :return:
    """
    # ---------- load data and pre-processing ----------
    x_train, y_train, x_test, y_test, n_labels = load_datasets(batch_size)

    # ---------- set placeholders ----------
    x_ph = tf.placeholder(tf.float32, shape=[batch_size, 28, 28], name='X_input')
    y_ph = tf.placeholder(tf.int32, shape=[batch_size], name='Y_input')

    # ---------- define model ----------
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.1) if regularize else None
    model = model_fn(x_ph, n_labels, regularizer=regularizer)
    loss_func = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_ph, logits=model, name='loss')
    if regularize:
        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
        loss = loss_func + reg_term
    else:
        loss = loss_func

    if learning_rate is None:
        optimizer = tf.train.AdamOptimizer().minimize(loss)
    else:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    # ---------- initializer ----------
    init = tf.global_variables_initializer()

    # ---------- collecting results ----------
    training_losses = list()
    training_accuracies = list()
    test_losses = list()
    test_accuracies = list()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(1, epochs + 1):

            e_training_losses = list()
            e_training_accuracies = list()
            e_test_losses = list()
            e_test_accuracies = list()

            # ---------- training ----------
            training_counter = 1
            for data, labels in zip(x_train, y_train):
                losses, _, logits = sess.run([loss, optimizer, model], feed_dict={x_ph: data, y_ph: labels})
                round_loss = np.mean(losses)
                round_accuracy = calculate_accuracy(logits, labels)
                training_losses.append(round_loss)
                training_accuracies.append(round_accuracy)
                e_training_losses.append(round_loss)
                e_training_accuracies.append(round_accuracy)
                print('Epoch {}, training round {}, loss: {}, accuracy: {}%'.format(epoch, training_counter, round_loss, round_accuracy)) \
                    if training_counter == 1 or not training_counter % 100 else None
                training_counter += 1

            # ---------- testing ----------
            test_counter = 1
            for t_data, t_labels in zip(x_test, y_test):
                logits, losses = sess.run([model, loss], feed_dict={x_ph: t_data, y_ph: t_labels})
                round_loss = np.mean(losses)
                round_accuracy = calculate_accuracy(logits, t_labels)
                test_losses.append(round_loss)
                test_accuracies.append(round_accuracy)
                e_test_losses.append(round_loss)
                e_test_accuracies.append(round_accuracy)
                print('Epoch {}, test round {}, loss: {}, accuracy: {}%'.format(epoch, test_counter, round_loss, round_accuracy)) if test_counter == 1 or not test_counter % 50 else None
                test_counter += 1

            # ---------- informing the user ----------
            template = 'Epoch {} summary - training loss: {}, training accuracy: {}%, test loss: {}, test accuracy: {}%\n'
            print(template.format(epoch, np.mean(e_training_losses), np.mean(e_training_accuracies), np.mean(e_test_losses), np.mean(e_test_accuracies)))

    # ---------- plotting the graphs ----------
    func_title = 'MLP\n' if model_fn == mlp else 'ConvNet\n'
    reg_title = ', with regularization' if regularize else ', without regularization'
    plot_graph(training_losses, func_title + 'training loss values\nwith {} epochs, and batch size {}'.format(epochs, batch_size) + reg_title)
    plot_graph(training_accuracies, func_title + 'training accuracies values\nwith {} epochs, and batch size {}'.format(epochs, batch_size) + reg_title)
    plot_graph(test_losses, func_title + 'test loss values\nwith {} epochs, and batch size {}'.format(epochs, batch_size) + reg_title)
    plot_graph(test_accuracies, func_title + 'test accuracy values\nwith {} epochs, and batch size {}'.format(epochs, batch_size) + reg_title)


def main():
    train(mlp, 100, epochs=10, regularize=True)
    tf.reset_default_graph()
    train(mlp, 100, epochs=10)
    # tf.reset_default_graph()
    # train(conv_net, 64, epochs=1, regularize=True)
    # tf.reset_default_graph()
    # train(conv_net, 64, epochs=1)
    plt.show()


if __name__ == "__main__":
    main()

