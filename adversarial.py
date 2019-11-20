import tensorflow as tf
import numpy as np
from model import mlp
from train import load_datasets, calculate_accuracy


def main():
    # ---------- general parameters ----------
    epsilon = 0.1    # noise multiplier
    batch_size = 64
    adversary_batches = 5  # number of adversary batches to create
    epochs = 5         # number of training / testing epochs

    # ---------- model definition ----------
    # getting batches to train
    x_train, y_train, x_test, y_test, n_labels = load_datasets(batch_size)

    # setting placeholders
    x_ph = tf.placeholder(tf.float32, shape=[batch_size, 28, 28], name='X_input')
    y_ph = tf.placeholder(tf.int32, shape=[batch_size], name='Y_input')

    # defining model, loss, regularizer, and optimizer
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
    model = mlp(x_ph, n_labels, regularizer=regularizer)
    loss_func = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_ph, logits=model, name='loss')
    reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
    loss = loss_func + reg_term
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    # variable initializer
    init = tf.global_variables_initializer()

    # ---------- training and generating examples ----------
    with tf.Session() as sess:
        sess.run(init)
        # training the network
        for epoch in range(epochs):
            for data, labels in zip(x_train, y_train):
                sess.run([loss, optimizer, model], feed_dict={x_ph: data, y_ph: labels})

        # ---------- generating adversarial examples ----------
        # 1. choosing batches of images out of the test set

        inds = np.random.randint(len(x_test), size=adversary_batches)
        data_set = [x_test[ind] for ind in inds]
        label_set = [y_test[ind] for ind in inds]

        # 2. calculating the sign of the gradient of the loss function with respect to the images
        gradients = tf.gradients(loss, [x_ph])
        s_gradients = tf.sign(gradients)

        original_accuracies = list()
        adversarial_accuracies = list()
        for data, labels in zip(data_set, label_set):
            s_grads = sess.run([s_gradients], feed_dict={x_ph: data, y_ph: labels})
            s_grads = np.reshape(s_grads[0], s_grads[0].shape[1:])

            # 3. creating the adversarial versions of the images in the batch
            adversarial_images = data + epsilon * s_grads

            # 4. checking accuracy on original images
            logits = sess.run([model], feed_dict={x_ph: data})
            logits = logits[0]
            original_accuracies.append(calculate_accuracy(logits, labels))

            # 5. checking accuracy on adversarial images
            logits = sess.run([model], feed_dict={x_ph: adversarial_images})
            logits = logits[0]
            adversarial_accuracies.append(calculate_accuracy(logits, labels))

        # 6. comparison
        print('accuracy of original images:', np.mean(original_accuracies))
        print('accuracy of adversarial images:', np.mean(adversarial_accuracies))


if __name__ == '__main__':
    main()
