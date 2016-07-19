#!/usr/bin/python

from __future__ import print_function

__author__ = "Matan Lachmish"
__copyright__ = "Copyright 2016, Tel Aviv University"
__version__ = "1.0"
__status__ = "Development"

import tensorflow as tf
import numpy as np
import pickle

def getBatch(data, labels, batchSize, iteration):
    startOfBatch = (iteration * batchSize) % len(data)
    endOfBacth = (iteration * batchSize + batchSize) % len(data)

    if startOfBatch < endOfBacth:
        return data[startOfBatch:endOfBacth], labels[startOfBatch:endOfBacth]
    else:
        dataBatch = np.vstack((data[startOfBatch:],data[:endOfBacth]))
        labelsBatch = np.vstack((labels[startOfBatch:],labels[:endOfBacth]))

        return dataBatch, labelsBatch


if __name__ == "__main__":

    # Parameters
    learning_rate = 0.001
    training_iters = 100000
    batch_size = 64
    display_step = 1
    train_size = 800

    # Network Parameters
    # n_input = 599 * 128*2
    n_input = 599 * 13 * 5
    n_classes = 10
    dropout = 0.75  # Dropout, probability to keep units

    # Load data
    data = []
    with open("data", 'r') as f:
        content = f.read()
        data = pickle.loads(content)
    data = np.asarray(data)
    data = data
    data = data.reshape((data.shape[0], n_input))

    labels = []
    with open("labels", 'r') as f:
        content = f.read()
        labels = pickle.loads(content)

    # #Hack
    # data = np.random.random((1000, n_input))
    # labels = np.random.random((1000, 10))

    # Shuffle data
    permutation = np.random.permutation(len(data))
    data = data[permutation]
    labels = labels[permutation]

    # Split Train/Test
    trainData = data[:train_size]
    trainLabels = labels[:train_size]

    testData = data[train_size:]
    testLabels = labels[train_size:]


    # tf Graph input
    x = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.float32, [None, n_classes])
    keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)


    # Create model
    def conv2d(sound, w, b):
        return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(sound, w, strides=[1, 1, 1, 1],
                                                      padding='SAME'), b))


    def max_pool(sound, k):
        return tf.nn.max_pool(sound, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


    def conv_net(_X, _weights, _biases, _dropout):
        # Reshape input picture
        _X = tf.reshape(_X, shape=[-1, 599, 13, 5])

        # Convolution Layer
        conv1 = conv2d(_X, _weights['wc1'], _biases['bc1'])
        # Max Pooling (down-sampling)
        conv1 = max_pool(conv1, k=4)
        # Apply Dropout
        conv1 = tf.nn.dropout(conv1, _dropout)

        # Convolution Layer
        conv2 = conv2d(conv1, _weights['wc2'], _biases['bc2'])
        # Max Pooling (down-sampling)
        conv2 = max_pool(conv2, k=2)
        # Apply Dropout
        conv2 = tf.nn.dropout(conv2, _dropout)
        #
        # # Convolution Layer
        # conv3 = conv2d(conv2, _weights['wc3'], _biases['bc3'])
        # # Max Pooling (down-sampling)
        # conv3 = max_pool(conv3, k=2)
        # # Apply Dropout
        # conv3 = tf.nn.dropout(conv3, _dropout)

        # Fully connected layer
        # Reshape conv3 output to fit dense layer input
        dense1 = tf.reshape(conv2, [-1, _weights['wd1'].get_shape().as_list()[0]])
        # Relu activation
        dense1 = tf.nn.relu(tf.add(tf.matmul(dense1, _weights['wd1']), _biases['bd1']))
        # Apply Dropout
        dense1 = tf.nn.dropout(dense1, _dropout)  # Apply Dropout

        # Output, class prediction
        out = tf.add(tf.matmul(dense1, _weights['out']), _biases['out'])
        return out


    # Store layers weight & bias
    weights = {
        # 4x4 conv, 1 input, 149 outputs
        'wc1': tf.Variable(tf.random_normal([4, 4, 5, 149])),
        # 4x4 conv, 149 inputs, 73 outputs
        'wc2': tf.Variable(tf.random_normal([4, 4, 149, 73])),
        # 4x4 conv, 73 inputs, 35 outputs
        'wc3': tf.Variable(tf.random_normal([2, 2, 73, 35])),
        # fully connected, 38*8*35 inputs, 2^13 outputs
        'wd1': tf.Variable(tf.random_normal([75 * 2 * 73, 1024])),
        # 2^13 inputs, 13 outputs (class prediction)
        'out': tf.Variable(tf.random_normal([1024, n_classes]))
    }

    biases = {
        'bc1': tf.Variable(tf.random_normal([149])),
        'bc2': tf.Variable(tf.random_normal([73])),
        'bc3': tf.Variable(tf.random_normal([35])),
        'bd1': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # Construct model
    pred = conv_net(x, weights, biases, keep_prob)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initializing the variables
    init = tf.initialize_all_variables()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Launch the graph
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        sess.run(init)
        step = 1
        # Keep training until reach max iterations
        while step * batch_size < training_iters:
            batch_xs, batch_ys = getBatch(trainData, trainLabels, batch_size, step)
            # Fit training using batch data
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})

            if step % display_step == 0:
                # Calculate batch accuracy
                acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
                # Calculate batch loss
                loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
                print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))

                save_path = saver.save(sess, "model.ckpt")
                print("Model saved in file: %s" % save_path)
            step += 1
        print("Optimization Finished!")

        save_path = saver.save(sess, "model.final")
        print("Model saved in file: %s" % save_path)

        # Calculate accuracy
        print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: testData,
                                                                 y: testLabels,
                                                                 keep_prob: 1.}))
