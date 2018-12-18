import tensorflow as tf
import time
import matplotlib.pyplot as plt
import numpy as np
import math
import random

#dict => {placeholder: value,.....}, should be {acc_data pair, label pair}
def make_feed_dict(inputs, data_placeholder, label_placeholder, batch_size):
    data_batch, label_batch = [], []
    for i in range(batch_size):
        #pair = inputs[1].pop(random.randint(0, len(inputs) - 1))
        data_batch.append(np.expand_dims(np.array(inputs[1]).T, axis=1))
        label_batch.append(np.array(inputs[0]).T)
    return {data_placeholder: data_batch,
            label_placeholder: label_batch}

def run_training(hparams, data_placeholder, label_placeholder, optimizer, accuracy, inputs):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        start_time = time.time()
        with open(hparams.log_file_train, 'a+') as f:
                f.write('\n**************** NEW TRAINING SESSION ****************\n')

        for i in range(hparams.max_iterations):
            feed_dict = make_feed_dict(inputs, data_placeholder, label_placeholder, hparams.batch_size)
            sess.run(optimizer, feed_dict=feed_dict)

            if i % hparams.print_loss_freq == 0:
                acc = sess.run(accuracy, feed_dict=feed_dict)
                msg = "***** ITERATION: {0}, TRAINING ACCURACY: {1} *****"
                print(msg.format(i, acc))

            if i % hparams.writing_frequency == 0:
                acc = sess.run(accuracy, feed_dict=feed_dict)
                msg = "***** ITERATION: {0}, TRAINING ACCURACY: {1} *****"
                with open(hparams.log_file_train, 'a+') as f:
                        f.write(msg.format(i, acc))
