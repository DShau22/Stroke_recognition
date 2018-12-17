import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math

#input, (conv, pool, relu) x3, FC, dropout, ReLu, FC, Softmax, classify
#input dims = H x W x D = 128 x 1 x 3

def make_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def make_bias(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

def make_conv_layer(
                    input,
                    num_input_channels,
                    filter_height,
                    filter_width,
                    num_filters,
                    stride,
                    use_pooling=True):

    #shape of the filter weights for convolution
    filter_shape = [filter_height, filter_width, num_input_channels, num_filters] #shape = filter dims, in chans, out chans

    filter_weights, biases = make_weights(shape=filter_shape), make_bias(length=num_filters)

    conv_layer = tf.nn.conv2d(input, filter_weights, stride, padding='VALID') #could use conv1d but this is more general

    conv_layer += biases

    if use_pooling:
        #do the 2x1 max pooling
        conv_layer = tf.nn.max_pool(conv_layer,
                                    [1, 2, 1, 1], #batch, height, width, channels
                                    [1, 2, 1, 1], #check the 3rd 1 if theres an error
                                    padding='VALID')
    conv_layer = tf.nn.relu(conv_layer)
    return conv_layer, filter_weights

#use tf.layer.flatten(input_tensor) => preserves batch axis, flattens tensor => [batch, #]
# input will be of shape [batch, flatten_len]
def make_fc_layer(input,
                  flattened_length,
                  num_outputs,
                  use_relu=True):

    weights, biases = make_weights(shape=[flattened_length, num_outputs]), make_bias(length=num_outputs)

    fc_layer = tf.matmul(input, weights) + biases

    if use_relu:
        fc_layer = tf.nn.relu(fc_layer)

    return fc_layer

def build_graph(hparams):

    #placeholders
    x = tf.placeholder(tf.float32, shape=[None, hparams.window_size], name='before-reshape')
    x_input = tf.reshape(x, [-1, hparams.window_size, hparams.input_width, hparams.input_channels], name='reshaped')

    y_true = tf.placeholder(tf.float32, shape=[None, hparams.num_classes], name='true_value') #one-hot
    y_class = tf.argmax(y_true, axis=1) #take the max entry in y_true, this is true stroke

    #already pooled and relu
    conv1, weights1 = make_conv_layer(x_input,
                                      hparams.input_channels,
                                      hparams.filter_height1,
                                      hparams.filter_width1,
                                      hparams.num_filters1,
                                      hparams.stride1)

    conv2, weights2 = make_conv_layer(conv1,
                                      hparams.num_filters1,
                                      hparams.filter_height2,
                                      hparams.filter_width2,
                                      hparams.num_filters2,
                                      hparams.stride2)

    conv3, weights3 = make_conv_layer(conv2,
                                      hparams.num_filters2,
                                      hparams.filter_height3,
                                      hparams.filter_width3,
                                      hparams.num_filters3,
                                      hparams.stride3)

    fc_input = tf.layers.flatten(conv3)
    flattened_length = shape = fc_input.get_shape().as_list()[1]

    fc_layer1 = make_fc_layer(fc_input,
                              flattened_length,
                              hparams.fc_size,
                              use_relu=False)
    fc_layer1 = tf.layers.dropout(fc_layer1, hparams.dropout_rate)
    fc_layer1 = tf.nn.relu(fc_layer1)

    fc_layer2 = make_fc_layer(fc_layer1,
                              hparams.fc_size,
                              hparams.num_classes,
                              use_relu=False)

    y_pred = tf.nn.softmax(fc_layer2)
    pred_stroke = tf.argmax(y_pred, axis=1)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=fc_layer2, #function does softmax internally
                                                            labels=y_true) #make sure to define/load y_true, one hot
    loss = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=hparams.learning_rate).minimize(loss)

    correct_prediction = tf.equal(pred_stroke, y_class)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return loss, optimizer, accuracy, x_input, y_true
