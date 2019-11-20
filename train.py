import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from model import mlp, conv_net


def normalize(data):
    """
    :param data: the array to normalize.
    :return: a normalized version of the input array.
    """
    res = data - np.mean(data)
    res = res / np.std(res)
    return res


def batch_dataset(data: np.ndarray, labels: np.ndarray, batch_size):
    """
    Divides the given dataset (consisting of data, and labels) into batches of size batch_size.
    Assumes that batch_size is smaller than the number of elements in the arrays.
    :param data: data of the dataset.
    :param labels: labels of the corresponding data vectors.
    :param batch_size: size of batches.
    :return: a list of batches of data points, and a list of batches of the corresponding labels.
    """
    inds = np.arange(data.shape[0])
    np.random.shuffle(inds)
    s_data = data[inds]
    s_labels = labels[inds]
    split_inds = [(i + 1) * batch_size for i in range(data.shape[0] // batch_size)]
    data_batches = np.array_split(s_data, split_inds)
    labels_batches = np.array_split(s_labels, split_inds)
    if data_batches[-1].shape[0] < batch_size:
        f_inds = np.arange(batch_size - data_batches[-1].shape[0])
        data_batches[-1] = np.array(list(data_batches[-1]) + list(s_data[f_inds]))
        labels_batches[-1] = np.array(list(labels_batches[-1]) + list(s_labels[f_inds]))
    return data_batches, labels_batches


def load_datasets(batch_size):
    """
    Loads and pre processes the fashion mnist dataset
    :param batch_size: size of batches to return.
    :return: a list of batches of data points, and a list of batches of the corresponding labels, for both the training and test sets.
    As well as the number of labels in the set.
    """
    ((x_train_np, y_train_np), (x_test_np, y_test_np)) = tf.keras.datasets.fashion_mnist.load_data()
    n_x_train = normalize(x_train_np)
    n_x_test = normalize(x_test_np)

    x_train, y_train = batch_dataset(n_x_train, y_train_np, batch_size)
    x_test, y_test = batch_dataset(n_x_test, y_test_np, batch_size)

    return x_train, y_train, x_test, y_test, len(np.unique(y_train_np))


def calculate_accuracy(logits, labels):
    """
    Calculates the accuracy of the predictions represented by the logits.
    :param logits: logits which represent the prediction results.
    :param labels: the correct labels to compare against
    :return: the ratio of correct predictions to all predictions.
    """
    s_logits = np.exp(logits)
    s_logits = s_logits / (np.sum(s_logits, axis=1)[..., np.newaxis])
    results = np.zeros(s_logits.shape[0]).astype(np.int32)
    for i in range(results.shape[0]):
        results[i] = np.argmax(s_logits[i])
    return 100 * len(results[results == labels]) / len(results)


def plot_graph(to_plot, title):
    """
    Plots given data which is given as a function of the number of epochs.
    :param to_plot: data to plot.
    :param title: title of the graph
    """
    plt.figure()
    plt.title(title)
    plt.xlabel('epoch')
    plt.plot(to_plot)


def train(model_fn, batch_size, learning_rate=None, epochs=10, regularize=False, plot=True):
    """
    load FashionMNIST data.
    create model using model_fn, and train it on FashionMNIST.
    :param model_fn: a function to create the model (should be one of the functions from model.py)
    :param batch_size: the batch size for the training
    :param learning_rate: optional parameter - option to specify learning rate for the optimizer.
    :param epochs: optional parameter - number of epochs to train the model.
    :param regularize: optional parameter - option to train the model with regularization.
    :param plot: optional parameter - allows the user to choose weather to plot the training/test loss/accuracy plots.
    :return:
    """
    template = 'Epoch {} - training loss: {}, training accuracy: {}%, test loss: {}, test accuracy: {}%'
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
            for data, labels in zip(x_train, y_train):
                losses, _, logits = sess.run([loss, optimizer, model], feed_dict={x_ph: data, y_ph: labels})
                round_loss = np.mean(losses)
                round_accuracy = calculate_accuracy(logits, labels)
                e_training_losses.append(round_loss)
                e_training_accuracies.append(round_accuracy)

            # ---------- testing ----------
            for t_data, t_labels in zip(x_test, y_test):
                logits, losses = sess.run([model, loss], feed_dict={x_ph: t_data, y_ph: t_labels})
                round_loss = np.mean(losses)
                round_accuracy = calculate_accuracy(logits, t_labels)
                e_test_losses.append(round_loss)
                e_test_accuracies.append(round_accuracy)

            # ---------- updating measurements and informing user ----------
            training_losses.append(np.mean(e_training_losses))
            training_accuracies.append(np.mean(e_training_accuracies))
            test_losses.append(np.mean(e_test_losses))
            test_accuracies.append(np.mean(e_test_accuracies))
            print(template.format(epoch, np.mean(e_training_losses), np.mean(e_training_accuracies), np.mean(e_test_losses), np.mean(e_test_accuracies)))

    # ---------- plotting the graphs ----------
    if plot:
        func_title = 'MLP\n' if model_fn == mlp else 'ConvNet\n'
        reg_title = ', with regularization' if regularize else ', without regularization'
        plot_graph(training_losses, func_title + 'training loss values\nwith {} epochs, and batch size {}'.format(epochs, batch_size) + reg_title)
        plot_graph(training_accuracies, func_title + 'training accuracies values\nwith {} epochs, and batch size {}'.format(epochs, batch_size) + reg_title)
        plot_graph(test_losses, func_title + 'test loss values\nwith {} epochs, and batch size {}'.format(epochs, batch_size) + reg_title)
        plot_graph(test_accuracies, func_title + 'test accuracy values\nwith {} epochs, and batch size {}'.format(epochs, batch_size) + reg_title)


def main():
    train(mlp, 64, epochs=50)
    tf.reset_default_graph()
    train(conv_net, 64, epochs=50)
    tf.reset_default_graph()
    train(mlp, 64, epochs=150, regularize=True)
    plt.show()


if __name__ == "__main__":
    main()
