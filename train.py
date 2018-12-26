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
    data_batch = np.expand_dims(data_batch, axis=2)
    return {data_placeholder: data_batch,
            label_placeholder: label_batch}

def run_training(hparams, data_placeholder, label_placeholder, optimizer, accuracy, input_pairs, summaries):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        start_time = time.time()

        merged_summaries = tf.summary.merge(summaries)
        writer = tf.summary.FileWriter("./tensorboard_logs2", sess.graph)

        with open(os.getcwd() + hparams.log_file_train, 'a+') as f:
                f.write('\n**************** NEW TRAINING SESSION ****************\n')

        for i in range(hparams.max_iterations):
            pair_input = input_pairs[random.randint(0, len(input_pairs) - 1)]
            feed_dict = make_feed_dict(hparams, pair_input, data_placeholder, label_placeholder)
            sess.run(optimizer, feed_dict=feed_dict)

            if i % hparams.tensorboard_ouput_freq == 0:
                s = sess.run(merged_summaries, feed_dict=feed_dict)
                writer.add_summary(s, i)

            if i % hparams.print_loss_freq == 0:
                acc = sess.run(accuracy, feed_dict=feed_dict)
                msg = "\n***** ITERATION: {0}, TRAINING ACCURACY: {1} *****\n"
                print(msg.format(i, acc))
                print(pair_input[0])

            if i % hparams.writing_frequency == 0:
                acc = sess.run(accuracy, feed_dict=feed_dict)
                msg = "\n***** ITERATION: {0}, TRAINING ACCURACY: {1} *****\n"
                with open(os.getcwd() + hparams.log_file_train, 'a+') as f:
                        f.write(msg.format(i, acc))
