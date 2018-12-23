import tensorflow as tf
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import preprocessing

#dict => {placeholder: value,.....}, should be {acc_data pair, label pair}
#inputs are a pair: label, data
def make_feed_dict(hparams, pair_input, data_placeholder, label_placeholder):
    label_batch, data_batch = preprocessing.make_batches(hparams, pair_input)
    # data_batch, label_batch = [], []
    # for i in range(hparams.batch_size):
    #     """choose random starting index from range(0, len-window-1) in inputs[1],
    #         add window_size elements starting from that index in inputs[1] to data_batch
    #         so len(data_batch) will be batch_size, add corresponding labels/inputs[0]
    #         to label_batch
    #     """
    #     print("LENGTH OF INPUTS[1]:{0}".format(len(inputs[1])))
    #     make_batches(hparams, inputs[1])
    #     # random_start = random.randint(0, len(inputs[1][0]) - hparams.window_size)
    #     # random_data_window = inputs[1][random_start: random_start + hparams.window_size]
    #     # random_label_window = inputs[0][random_start: random_start + hparams.window_size]
    #     #pair = inputs[1].pop(random.randint(0, len(inputs) - 1))
    #
    #     #dims expanded to fit necessary shape for convolution
    #     data_batch.append(np.expand_dims(np.array(random_data_window).T, axis=1))
    #     label_batch.append(np.array(inputs[random_label_window]).T)
    data_batch = np.expand_dims(data_batch, axis=2)
    return {data_placeholder: data_batch,
            label_placeholder: label_batch}

def run_training(hparams, data_placeholder, label_placeholder, optimizer, accuracy, pair_input):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        start_time = time.time()
        with open(os.getcwd() + hparams.log_file_train, 'a+') as f:
                f.write('\n**************** NEW TRAINING SESSION ****************\n')

        for i in range(hparams.max_iterations):
            feed_dict = make_feed_dict(hparams, pair_input, data_placeholder, label_placeholder)
            sess.run(optimizer, feed_dict=feed_dict)

            if i % hparams.print_loss_freq == 0:
                acc = sess.run(accuracy, feed_dict=feed_dict)
                msg = "***** ITERATION: {0}, TRAINING ACCURACY: {1} *****"
                print(msg.format(i, acc))

            if i % hparams.writing_frequency == 0:
                acc = sess.run(accuracy, feed_dict=feed_dict)
                msg = "***** ITERATION: {0}, TRAINING ACCURACY: {1} *****"
                with open(os.getcwd() + hparams.log_file_train, 'a+') as f:
                        f.write(msg.format(i, acc))
