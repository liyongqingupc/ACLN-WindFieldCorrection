# import os
# import scipy.misc
# import numpy as np

##!/usr/bin/env python
##-*- coding:UTF-8 -*-

from model import WindReanalysisWGAN
import tensorflow as tf
#tf.compat.v1.disable_v2_behavior()

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

flags = tf.compat.v1.app.flags
flags.DEFINE_integer("epoch", 100, "Epoch to train [25]")
flags.DEFINE_integer("train_batch_size", 4, "The size of batch images [64]")  # 4
flags.DEFINE_integer("test_batch_size", 1, "The size of batch images [64]")
flags.DEFINE_integer("checkpoint_file", None, "The checkpoint file name")
flags.DEFINE_integer("lambd", 10, "Gradient penalty lambda hyperparameter")
flags.DEFINE_integer("lambdl1", 10, "l1 norm lambda hyperparameter")
flags.DEFINE_integer("save_freq", 10, "Freq to save")    # save_freq
flags.DEFINE_integer("histlen", 2, "History seq length")
flags.DEFINE_integer("futulen", 0, "Future seq length")
flags.DEFINE_float("learn_rate", 5e-4, "learn_rate")
FLAGS = flags.FLAGS

def main(_):
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    run_config = tf.compat.v1.ConfigProto()
    run_config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config = run_config) as sess:
        wrwgan = WindReanalysisWGAN(sess, train_batch_size = FLAGS.train_batch_size,\
                            test_batch_size = FLAGS.test_batch_size,\
                            epochs = FLAGS.epoch, checkpoint_file = FLAGS.checkpoint_file,\
                            lambd = FLAGS.lambd,lambdl1 = FLAGS.lambdl1, save_freq = FLAGS.save_freq,\
                            histlen = FLAGS.histlen, futulen = FLAGS.futulen, learn_rate = FLAGS.learn_rate)
        wrwgan.build_model()
        wrwgan.train()

if __name__ == '__main__':
    tf.compat.v1.app.run()