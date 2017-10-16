import tensorflow as tf
import numpy as np
import os, sys, re

def lrelu(x, rate=0.1):
    # return tf.nn.relu(x)
    return tf.maximum(tf.minimum(x * rate, 0), x)

# relu activation
# @param kernel_shape == filter shape,  which is [batch_samples, filter_height, filter_width, ...]
def conv_bn_relu(input, kernel_shape, bias_shape):
    # Create variable named "weights".
    weights = tf.get_variable("weights", kernel_shape,
        initializer=tf.random_normal_initializer())
    # Create variable named "biases".
    biases = tf.get_variable("biases", bias_shape,
        initializer=tf.constant_initializer(0.1))
    conv = tf.nn.conv2d(input, filter=weights,
        strides=[1, 1, 1, 1], padding='SAME')
    conv_res = conv + biases
    conv_bn = tf.contrib.layers.batch_norm(conv_res)
    return tf.nn.relu(conv_bn)


# leaky relu activation
def conv_bn_lrelu(input_x, kernel_shape, bias_shape):
    # create variable
    input = tf.contrib.layers.batch_norm(input_x)

    weights = tf.get_variable("weights", kernel_shape,
                              initializer=tf.random_normal_initializer())
    # Create variable named "biases".
    biases = tf.get_variable("biases", bias_shape,
                             initializer=tf.constant_initializer(0.1))

    conv = tf.nn.conv2d(input, weights,
                        strides=[1, 1, 1, 1], padding='SAME')  # stride : [batch, height, width, channel]
    conv_res = conv + biases
    conv_bn = tf.contrib.layers.batch_norm(conv_res)
    return lrelu(conv_bn)




