"""
Author: Yongqing Li
Code structure inspired from https://github.com/carpedm20/DCGAN-tensorflow
"""

#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # 只显示 warning 和 Error lyq 0514

import tensorflow as tf
import numpy as np
import glob
import time
from utils import *
import constants as c

from tensorflow.keras.layers import ConvLSTM2D
#from tensorflow.keras.models import Sequential  # lyq 0728
#from tensorflow.keras.normalization import BatchNormalization

class WindReanalysisWGAN():

    def __init__(self,sess,train_batch_size,test_batch_size,epochs,checkpoint_file,lambd,lambdl1,save_freq,histlen,futulen,learn_rate):

        self.bs1 = batch_norm(name = "genb1")
        self.bs2 = batch_norm(name = "genb2")
        self.bs3 = batch_norm(name = "genb3")

        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.lambd = lambd
        self.lambdl1 = lambdl1   # lyq add: used for l1 norm term
        self.checkpoint_file = checkpoint_file
        self.sess = sess
        self.save_freq = save_freq   # lyq add
        self.histlen = histlen   # lyq add
        self.futulen = futulen  # lyq add
        self.learn_rate = learn_rate   # lyq add


    def build_model(self):
        self.train_original = tf.placeholder(tf.float32, [None, self.histlen + 1 + self.futulen, c.data_height, c.data_width, 2])
        self.train_revised = tf.placeholder(tf.float32, [None, 1, c.data_height, c.data_width, 2])
        self.train_revised_fake = self.generator(self.train_original)

        self.test_original = tf.placeholder(tf.float32, [None, (self.histlen + 1 + self.futulen), c.data_height, c.data_width, 2])
        self.test_revised = tf.placeholder(tf.float32, [None, 1, c.data_height, c.data_width, 2])
        #self.test_revised_fake = self.generator(self.test_original)  # revise_data->generator 0804
        self.test_revised_fake = self.revise_data(self.test_original)

        ########### D ##########
        train_revised_c = tf.concat((self.train_original[:, self.histlen: self.histlen + 1, :, :, :], self.train_revised), axis=-1) #lyq add 0908
        print('train_revised_c:', train_revised_c.shape)
        disc_real = self.discriminator(train_revised_c[:,0,:,:,:])
        train_revised_fake_c = tf.concat((self.train_original[:, self.histlen: self.histlen + 1, :, :, :], self.train_revised_fake), axis=-1) #lyq add 0908
        print('train_revised_fake_c:', train_revised_fake_c.shape)
        disc_fake = self.discriminator(train_revised_fake_c[:,0,:,:,:], reuse=True)
        # Standard WGAN loss:
        self.d_cost_adversarial = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
        # Gradient penalty loss:
        alpha = tf.random_uniform(
            shape=[self.train_batch_size, 1, c.data_height, c.data_width, 1], # lyq add 1, 0908
            minval=0.,
            maxval=1.
        )
        differences = self.train_revised_fake - self.train_revised
        print('differences:', differences.shape)
        interpolates = self.train_revised + (alpha * differences)  # alpha = tf.random_uniform()
        print('interpolates:', interpolates.shape)
        interpolates_c = tf.concat((self.train_original[:, self.histlen: self.histlen + 1, :, :, :], interpolates), axis=-1)  # lyq add 0908
        #gradients = tf.gradients(self.discriminator(interpolates, reuse=True), [interpolates])[0]  # original
        gradients = tf.gradients(self.discriminator(interpolates_c[:,0,:,:,:], reuse=True), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)

        # the final d_cost:
        self.d_cost = self.d_cost_adversarial + self.lambd * gradient_penalty

        ########## G ##########
        self.g_cost_adversarial = -tf.reduce_mean(disc_fake)
        self.g_cost_l1 = self.lambdl1 * tf.reduce_mean(tf.abs(self.train_revised - self.train_revised_fake)**1) #lyq sum->mean 1:l1,2:l2

        # the final g_cost:
        self.g_cost = self.g_cost_adversarial + self.g_cost_l1

        self.truth_error = tf.reduce_mean(tf.abs(self.test_original[:, self.histlen: self.histlen + 1, :, :, :] - self.test_revised))
        self.test_error = tf.reduce_mean(tf.abs(self.test_revised - self.test_revised_fake))  #lyq sum -> mean

        #self.u_truth_error = tf.reduce_mean(tf.abs(self.test_original[:, self.histlen: self.histlen + 1, :, :, 0:1] - self.test_revised[:,:,:,:,0:1]))
        #self.u_test_error = tf.reduce_mean(tf.abs(self.test_revised[:,:,:,:,0:1] - self.test_revised_fake[:,:,:,:,0:1]))  # lyq sum -> mean

        #self.v_truth_error = tf.reduce_mean(tf.abs(self.test_original[:, self.histlen: self.histlen + 1, :, :, 1:2] - self.test_revised[:,:,:,:,1:2]))
        #self.v_test_error = tf.reduce_mean(tf.abs(self.test_revised[:,:,:,:,1:2] - self.test_revised_fake[:,:,:,:,1:2]))  # lyq sum -> mean

        #print(np.mean(abs(self.test_error)))

    def train(self):
        global train_num
        global d_cost

        gen_var = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope = "generator")
        dis_var = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")

        ########## RMSPropOptimizer for G #############
        self.g_opt = tf.train.RMSPropOptimizer(learning_rate = 5e-4).minimize(self.g_cost, var_list = gen_var) #lyq0908
        self.d_opt = tf.train.AdamOptimizer(learning_rate=self.learn_rate).minimize(self.d_cost, var_list=dis_var)
        saver = tf.compat.v1.train.Saver()  #used to save model

        if self.checkpoint_file == "None":
            self.ckpt_file = None
        if self.checkpoint_file:  # checkpoint_file
            saver_ = tf.compat.v1.train.import_meta_graph('../save/models/' + self.checkpoint_file + '.meta')  # meta_graph
            saver_.restore(self.sess, tf.compat.v1.train.latest_checkpoint(c.save_models_dir))
            print ("Restored model")
        else:
            tf.compat.v1.global_variables_initializer().run()

        train_data_original = glob.glob(os.path.join(c.train_original_dir, '*'))  # lyq add 0410
        train_data_original.sort(key=lambda x:int(x.split('201802_201812_')[1].split('.npy')[0])) # lyq add 0410
        #################################
        #print(train_data_original[998:1005])   # see the index of input data
        #################################
        train_data_revised = glob.glob(os.path.join(c.train_revised_dir, '*')) # lyq delete sorted
        train_data_revised.sort(key=lambda x:int(x.split('201802_201812_')[1].split('.npy')[0])) # lyq add 0410
        test_data_original = glob.glob(os.path.join(c.test_original_dir, '*'))
        test_data_original.sort(key=lambda x:int(x.split('201801_')[1].split('.npy')[0]))
        test_data_revised = glob.glob(os.path.join(c.test_revised_dir, '*'))
        test_data_revised.sort(key=lambda x:int(x.split('201801_')[1].split('.npy')[0]))

        ####### define g_diff_whole:
        #g_diff_whole = np.empty([self.epochs, int(len(train_data_original) / self.train_batch_size) - 0]) # 0429#

        ####### define revise_diff_whole:
        #revise_diff_whole = [] # 0429#

        for epoch in range(self.epochs):
            start_time = time.time()

            f = open(c.save_name + ".txt", "a")  # open a txt file to save results
            print("......Epoch_", epoch, "......", file=f)

            #################### cut_num ##################
            cut_num = 10  # lyq add: num of deleted sample batches
            #################### cut_num ##################

            for counter in range(0, int((len(train_data_original) - self.histlen) / self.train_batch_size) - cut_num, 1):  # - cut_num
                train_num = int((len(train_data_original) - self.histlen) / self.train_batch_size) - cut_num

                if np.mod(counter, 0.5 * train_num - 1) == 0:
                    print("....Iteration....:", counter, '/', train_num)

                ###################
                batch_original_path = train_data_original[counter * self.train_batch_size : \
                                                          self.histlen + self.futulen + (counter + 1) * self.train_batch_size]
                input_original = read_data(batch_original_path, self.train_batch_size, self.histlen, self.futulen)

                ###################
                batch_revised_path = train_data_revised[self.histlen + counter * self.train_batch_size + 1: \
                                                          self.histlen + (counter + 1) * self.train_batch_size + 1]  # lyq add +1
                truth_revised = read_data(batch_revised_path, self.train_batch_size, 0, 0)  # lyq test: histlen = 0

                ################### Print error ##################
                #print('input_original-truth_revised:', np.mean(abs(input_original[:, :, :, 4:5] - truth_revised)))

                ################### type(revised_fake): list, can't use .shape
                #revised_fake = self.sess.run([self.train_revised_fake],
                #                         feed_dict = {self.train_original: input_original, self.train_revised: truth_revised})
                _, d_cost = self.sess.run([self.d_opt, self.d_cost],
                                         feed_dict = {self.train_original: input_original, self.train_revised: truth_revised})
                _, g_cost, g_cost_adversarial, g_cost_l1 = self.sess.run([self.g_opt, self.g_cost, self.g_cost_adversarial, self.g_cost_l1],
                                         feed_dict = {self.train_original: input_original, self.train_revised: truth_revised})
                #g_diff_whole[epoch, counter] = g_cost_difference

                if np.mod(counter, int(0.2 * train_num)) == 0:  # print g_cost_diff
                    print("d_cost: ", d_cost,"----g_cost_ad: ", g_cost_adversarial,"----g_cost_l1: ", g_cost_l1,"----g_cost: ", g_cost, file = f)
                    #print("Discriminator Loss: ", d_cost)
                    #print("Generator Adversarial Loss: ", g_cost_adversarial)
                    #print("Generator l1 Loss: ", g_cost_l1)

            ################# Print loss of each epoch ################
            #print('input_original:', input_original.shape) # input_original: (4, 241, 241, 2)
            #print('truth_revised:', truth_revised.shape) # truth_revised: (4, 241, 241, 2)
            #print('revised_fake:', np.array(revised_fake).shape) # revised_fake: (1, 4, 241, 241, 2)
            print('time', time.time() - start_time)
            '''print losses and save the generated data every epoch'''
            print("Discriminator Loss: ", d_cost)
            print("Generator Adversarial Loss: ", g_cost_adversarial)
            print("Generator l1 Loss: ", g_cost_l1)

            #################### Save generated data ####################
            #np.save(os.path.join(c.train_save_dir + '/train' + str(train_num) + '_histlen'+ str(self.histlen)
            #                     + '_epoch'+ str(epoch) + '.npy'), revised_fake)  # for every epoch
            #print("Saved generated data in training")

            '''test and save the model every save_freq epoches'''
            #####################################################
            if np.mod(epoch + 1, self.save_freq) == 0:
                print("......Testing.....")
                for tcounter in range(0, int((len(test_data_original) - self.histlen) / self.test_batch_size) - 2, 1):
                    test_num = int((len(test_data_original) - self.histlen) / self.test_batch_size) - 2 # lyq -1, avoid list index out of range
                    ####################
                    tbatch_original_path = test_data_original[tcounter * self.test_batch_size: \
                                           self.histlen + self.futulen + (tcounter + 1) * self.test_batch_size]
                    tinput_original = read_data(tbatch_original_path, self.test_batch_size, self.histlen, self.futulen)

                    ####################
                    tbatch_revised_path = test_data_revised[self.histlen + tcounter * self.test_batch_size + 1: \
                                          self.histlen + (tcounter + 1) * self.test_batch_size + 1]
                    ttruth_revised = read_data(tbatch_revised_path, self.test_batch_size, 0, 0)

                    ################### Print error ##################
                    #print('tinput_original-ttruth_revised:', np.mean(abs(tinput_original[:, :, :, self.histlen : self.histlen+1] - ttruth_revised)))

                    ########## two run can be merged? ##########
                    trevised_fake = self.sess.run([self.test_revised_fake],
                                            feed_dict = {self.test_original: tinput_original, self.test_revised: ttruth_revised})
                    #truth_diff, test_diff, u_truth_diff, u_test_diff, v_truth_diff, v_test_diff= self.sess.run([self.truth_error, self.test_error, self.u_truth_error,self.u_test_error,self.v_truth_error,self.v_test_error],
                                            #feed_dict = {self.test_original: tinput_original, self.test_revised: ttruth_revised}) #lyq 0804#
                    truth_diff, test_diff = self.sess.run([self.truth_error, self.test_error, ],
                                            feed_dict={self.test_original: tinput_original, self.test_revised: ttruth_revised})  #lyq 0908
                    #print('tinput_original:', np.mean(abs(tinput_original[:, self.histlen: self.histlen + 1, :, :, :])))
                    #print('ttruth_revised:', np.mean(abs(ttruth_revised)))

                    #truth_diff = np.mean(abs(tinput_original[:, self.histlen: self.histlen + 1, :, :, :] - ttruth_revised))  # lyq 0804 add
                    #test_diff = np.mean(abs(trevised_fake - ttruth_revised)) # lyq 0804 add

                    #revise_diff_whole.append(test_diff)

                    #print('test_num:',test_num)
                    #print('tcounter:', tcounter)
                    if np.mod(tcounter, 5) == 0:  # print error every half test samples #int(0.1 * test_num)
                        print("Truth error: ", truth_diff,"----Test error: ", test_diff, "----Truth-Test: ", truth_diff-test_diff, file=f)
                        #print("Truth:", truth_diff, "Test:",test_diff, "u_Truth:", u_truth_diff, "u_Test:",u_test_diff, "v_Truth:", v_truth_diff, "v_Test:",v_test_diff, file = f)
                        #print("Truth_diff-Test_diff:", truth_diff-test_diff, "----u_Truth-u_Test:", u_truth_diff-u_test_diff,"----v_Truth-v_Test:", v_truth_diff-v_test_diff, file = f)
                        #print('tinput_original:', (tinput_original[:, self.histlen: self.histlen + 1, :, :, :]).shape, 'ttruth_revised:', ttruth_revised.shape)
                        #tinput_original: (1, 1, 241, 241, 1)  ttruth_revised: (1, 1, 241, 241, 1)

                    test_save_dir = c.get_dir(os.path.join(c.test_save_dir, 'train' +  str(train_num) + \
                                    '_histlen'+ str(self.histlen) + '_epoch'+ str(epoch)))
                    '''save generated test data'''
                    np.save(os.path.join(test_save_dir, str(epoch) + '_' + str(tcounter) + '.npy'), trevised_fake)

                '''save the trained model every save_freq epoches'''
                saver.save(self.sess, os.path.join(c.save_models_dir, 'WindPred' + str(epoch) + '.ckpt'))
                print ('Saved models {}'.format(epoch))

                '''save the diff_whole every save_freq epoches'''
                #np.save(os.path.join(c.loss_save_dir, 'WindPred_epoch' + str(epoch) + '_g_diff_whole.npy'), g_diff_whole)
                #np.save(os.path.join(c.loss_save_dir, 'WindPred_epoch' + str(epoch) + '_r_diff_whole.npy'), revise_diff_whole)
            f.close()  # Close  lyq0605


    def generator(self, train_original, reuse = False):
        with tf.compat.v1.variable_scope("generator") as scope:

            revised_data1 = ConvLSTM2D(filters=16, kernel_size=(5, 5),
                                       input_shape=(None, 241, 241, 2),
                                       padding='same', return_sequences=True, activation='tanh',  #lyq 0823 must be tanh
                                       go_backwards=True,
                                       kernel_initializer='glorot_uniform',
                                       recurrent_initializer='orthogonal',
                                       recurrent_activation='hard_sigmoid', unit_forget_bias=True,
                                       dropout=0, recurrent_dropout=0)(train_original)
            print('train_original:', train_original.shape)
            print('g_revised_data1:',revised_data1.shape)
            revised_data1 = (self.bs1(revised_data1)) # lrelu() is defined in utils.py #0831 delete lrelu

            revised_data2 = ConvLSTM2D(filters=32, kernel_size=(3, 3),
                                       padding='same', return_sequences=True, activation='tanh',
                                       go_backwards=True,
                                       kernel_initializer='glorot_uniform',
                                       recurrent_initializer='orthogonal',
                                       recurrent_activation='hard_sigmoid', unit_forget_bias=True,
                                       dropout=0, recurrent_dropout=0
                                       )(revised_data1)
            print('g_revised_data2:', revised_data2.shape)
            revised_data2 = (self.bs2(revised_data2)) #0831 delete lrelu

            revised_data3 = ConvLSTM2D(filters=16, kernel_size=(3, 3), # lyq0830 2->1
                                       padding='same', return_sequences=False, activation='tanh',
                                       go_backwards=True,
                                       kernel_initializer='glorot_uniform',
                                       recurrent_initializer='orthogonal',
                                       recurrent_activation='hard_sigmoid', unit_forget_bias=True,
                                       dropout=0, recurrent_dropout=0
                                       )(revised_data2)
            print('g_revised_data3:', revised_data3.shape)
            revised_data3 = (self.bs3(revised_data3)) #0831 delete lrelu

            revised_data4 = tf.layers.conv2d(revised_data3, 1, kernel_size=[3, 3], strides=[1, 1], padding='SAME',name='Conv1')
            revised_data4 = tf.nn.tanh(revised_data4)  # lyq add 0909

            revised_data4_extend = revised_data4[:, np.newaxis, :, :, :]
            revised_data = revised_data4_extend + train_original[:, self.histlen: self.histlen + 1, :, :, :]
            print('g_revised_data:', revised_data.shape)

            return revised_data


    def revise_data(self, test_original):
        with tf.compat.v1.variable_scope("generator") as scope:
            scope.reuse_variables()
            revised_data1 = ConvLSTM2D(filters=16, kernel_size=(5, 5),
                                       input_shape=(None, 241, 241, 2),
                                       padding='same', return_sequences=True, activation='tanh',
                                       recurrent_activation='hard_sigmoid')(test_original)
            print('test_original:', test_original.shape)
            print('r_revised_data1:', revised_data1.shape)
            revised_data1 = (self.bs1(revised_data1,train=False))  #lrelu

            revised_data2 = ConvLSTM2D(filters=32, kernel_size=(3, 3),
                                       padding='same', return_sequences=True, activation='tanh',
                                       recurrent_activation='hard_sigmoid')(revised_data1)
            print('r_revised_data2:', revised_data2.shape)
            revised_data2 = (self.bs2(revised_data2,train=False))

            revised_data3 = ConvLSTM2D(filters=16, kernel_size=(3, 3),  # lyq 0819 1->2
                                       padding='same', return_sequences=False, activation='tanh',
                                       recurrent_activation='hard_sigmoid')(revised_data2)
            print('r_revised_data3:', revised_data3.shape)
            revised_data3 = (self.bs3(revised_data3,train=False))

            revised_data4 = tf.layers.conv2d(revised_data3, 1, kernel_size=[3, 3], strides=[1, 1], padding='SAME',name='Conv1')
            revised_data4 = tf.nn.tanh(revised_data4)  # lyq add 0909

            revised_data4_extend = revised_data4[:, np.newaxis, :, :, :]
            print('revised_data4_extend:', revised_data4_extend.shape)
            print('test_original:', (test_original[:, self.histlen: self.histlen + 1, :, :, :]).shape)
            revised_data = revised_data4_extend + test_original[:, self.histlen: self.histlen + 1, :, :, :]

            return revised_data

    def discriminator(self, data, reuse = False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            m = 2  # lyq add
            conv1 = tf.layers.conv2d(data,64/m,kernel_size=[5,5],strides=[1,1],padding="VALID",reuse=reuse,name="dis1")
            conv1 = lrelu(conv1)  # No BN after the conv layer
            conv2 = tf.layers.conv2d(conv1,128/m,kernel_size=[3,3],strides=[1,1],padding="VALID",reuse=reuse,name="dis2")
            conv2 = lrelu(conv2)
            conv3 = tf.layers.conv2d(conv2,128/m,kernel_size=[3,3],strides=[1,1],padding="VALID",reuse=reuse,name="dis3")
            conv3 = lrelu(conv3)
            conv4 = tf.layers.conv2d(conv3,64/m,kernel_size=[3,3],strides=[1,1],padding="VALID",reuse=reuse,name="dis4") #lyq 7,7->5,5
            conv4 = lrelu(conv4)  # lyq add
            conv5 = tf.layers.conv2d(conv4, 1, kernel_size=[3, 3], strides=[1, 1], padding="VALID", reuse=reuse,name="dis5") # lyq add
            # No BN at the last layer
            return conv5 # lyq add

