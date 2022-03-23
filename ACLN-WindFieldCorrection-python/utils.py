import tensorflow as tf
import cv2
import glob
import os
import numpy as np
import constants as c
"""
Author: Vignesh Gokul
Utilities inspired from https://github.com/carpedm20/DCGAN-tensorflow
"""
save_dir = '../save'

############################
# files: history and current path of all batches.
############################
def read_data(files, batch_size, histlen, futulen):  # lyq add histlen 0406
    data = np.empty([batch_size, (histlen + 1 + futulen), c.data_height, c.data_width, 2])
    for i in range(batch_size):
        data_single = np.empty([(histlen + 1 + futulen), c.data_height, c.data_width, 2])
        for j in range(histlen + 1 + futulen):
            data_single[j : j + 1, :, :, :] = np.load(files[i + j])[:, :, 0 : 2] + 20
        data[i, :, :, :, :] = data_single
    return data


def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim], initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
    return conv


def lrelu(x, leak=0.2, name="lrelu"):   # lyq 0.2->1
    return tf.maximum(x, leak*x)

class batch_norm(object):
  def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
    with tf.compat.v1.variable_scope(name):
      self.epsilon  = epsilon
      self.momentum = momentum
      self.name = name

  def __call__(self, x, train=True):  # when test, train = False ????? lyq0908
    return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum,
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=train,
                      scope=self.name)


def psnr_error(gen_videos,gt_videos):
    shape = tf.shape(gen_videos)
    num_pixels = tf.to_float(shape[1] * shape[2] * shape[3] * shape[4])
    square_diff = tf.to_float(tf.square(gt_videos - gen_videos))
    batch_errors = 10 * log10(1 / ((1 / num_pixels) * tf.reduce_sum(square_diff, [1,2,3,4])))
    return tf.reduce_mean(batch_errors)

def log10(t):
    numerator = tf.log(t)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

