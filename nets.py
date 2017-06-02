from __future__ import division

import gym
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.keras as keras
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.misc
import os
import pdb

import re

import cPickle

import pyemd

from PIL import Image
import cv2

def local_blob(size):
    blob = np.zeros((size, size))
    middle = (size-1)/2
    for i in range(size):
        for j in range(size):
            blob[i,j] = 1./(1+((i-middle)**2 + (j-middle)**2))
    return blob

class protoQnetwork():
    def __init__(self, env, h_size, mover_prototypes, mover_disps, blob_size,
                 model_name, dueling=True, lr=0.001, eps=1e-3):
        self.model_name = model_name
        self.blob_size = blob_size
        self.lr = lr
        self.eps = eps
        self.scalarInput =  tf.placeholder(shape=[None,210*160*3*2],dtype=tf.float32)
        self.imageIn = tf.reshape(self.scalarInput/255.,shape=[-1,210,160,3*2])

        self.mover_conv_list = []
        self.disp_conv_list = []

        for i, proto in enumerate(mover_prototypes):
            self.mover_conv_list.append(self.get_conv_mover(proto, i))

        self.conv_movers = tf.concat(self.mover_conv_list,3)

        for i, disps in enumerate(mover_disps):
            self.disp_conv_list.append(self.get_conv_disp(disps, i))

        self.conv_disps = tf.concat(self.disp_conv_list,3)

        self.conv_dm = tf.concat([self.conv_disps,
                                            self.conv_movers],3)

        self.conv_dm_pool = slim.pool(self.conv_dm, [4,4], \
                                         'MAX', 'VALID', stride=[4,4])

        #Leaky1 = keras.layers.LeakyReLU(0.1)
        self.conv1 = slim.conv2d( \
            inputs=self.conv_dm_pool,num_outputs=16,kernel_size=[4,4],stride=[2,2],
                                 padding='VALID', biases_initializer=None)

        self.conv1_pool = slim.pool(self.conv1, [3,3], \
                                         'MAX', 'VALID', stride=[3,3])

        #Leaky2 = keras.layers.LeakyReLU(0.1)

        # for width 250: kernel_size=[10,6]
        # for width 210: kernel_size=[8,6]
        self.conv2 = slim.conv2d( \
            inputs=self.conv1_pool,num_outputs=h_size,kernel_size=[8,6],stride=[1,1],
                                 padding='VALID', biases_initializer=None)
        #We take the output from the final convolutional layer and split it into separate advantage and value streams.

        self.streamAC,self.streamVC = tf.split(self.conv2,2,3)
        self.streamA = slim.flatten(self.streamAC)
        self.streamV = slim.flatten(self.streamVC)
        xavier_init = tf.contrib.layers.xavier_initializer()
        self.AW = tf.Variable(xavier_init([h_size//2,env.action_space.n]))
        self.VW = tf.Variable(xavier_init([h_size//2,1]))

        self.Advantage = tf.matmul(self.streamA,self.AW)
        self.Value = tf.matmul(self.streamV,self.VW) + 0. # experimental

        #Then combine them together to get our final Q-values.
        if dueling:
            self.Qout = self.Value + tf.subtract(self.Advantage,tf.reduce_mean(self.Advantage,axis=1,keep_dims=True))
        else:
            self.Qout = self.Advantage
        self.predict = tf.argmax(self.Qout,1)

        #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions,env.action_space.n,dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)

        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=self.lr,
                                             epsilon=self.eps)
        self.trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                 scope='(?!' + self.model_name +
                               '/piaget)')
        self.updateModel = self.trainer.minimize(self.loss,
                                                 var_list=self.trainables)
        self.grads = self.trainer.compute_gradients(self.loss,
                                                 var_list=self.trainables)

    def get_conv_mover(self, proto, ind):

        p0 = proto/(255.)

        p0_mean = np.mean(p0,(0,1))
        p0 = p0 - p0_mean

        p0_norm = np.sqrt(np.sum(p0**2))
        p0 = p0 / p0_norm

        p0_frame0 = np.concatenate([p0, np.zeros_like(p0)],2)
        p0_frame1 = np.concatenate([np.zeros_like(p0), p0],2)

        p0_multi_frame = np.stack([p0_frame0, p0_frame1],3)

        with tf.variable_scope(self.model_name +
                               "/piaget/prototypes/proto" + str(ind)):
            self.kernel = tf.get_variable(name='kernel',
                                        shape=p0_multi_frame.shape,
                                       initializer=tf.constant_initializer(
                                           p0_multi_frame))
            self.conv = tf.nn.conv2d(input=self.imageIn/p0_norm,
                               filter=self.kernel,
                               strides=[1,1,1,1],
                               padding='SAME'
                              )
            self.biases = tf.get_variable(name='bias',shape=(2),
                                    initializer=tf.constant_initializer(
                                        -0.75))
            self.bias = tf.nn.bias_add(self.conv, self.biases)
            self.conv_p0 = tf.nn.relu(self.bias)
            return self.conv_p0

    def get_conv_disp(self, disps, ind):
        blob = local_blob(self.blob_size)

        blob_norm = np.sqrt(np.sum(blob**2))
        blob = blob / blob_norm

        conv_disps = []
        for j, disp in enumerate(disps):
            with tf.variable_scope(self.model_name +
                   "/piaget/mover_disps/mover" + str(ind) +
                    "/disp" + str(j)):

                dx, dy = disp
                blob_frame0 = np.pad(blob,(\
                                             (abs(dy)*(dy<0), abs(dy)*(dy>0)),\
                                             (abs(dx)*(dx<0), abs(dx)*(dx>0))\
                                            ),
                                       'constant')
                blob_frame1 = np.pad(blob,(\
                             (abs(dy)*(dy>0), abs(dy)*(dy<0)),\
                             (abs(dx)*(dx>0), abs(dx)*(dx<0))\
                            ),
                       'constant')

                blob_multi_frame = np.stack(\
                                            [blob_frame0, \
                                             blob_frame1]\
                                            ,2)
                blob_multi_frame = np.expand_dims(blob_multi_frame, 3)

                self.kernel = tf.get_variable(name='kernel',
                                        shape=blob_multi_frame.shape,
                                       initializer=tf.constant_initializer(
                                           blob_multi_frame))
                self.conv = tf.nn.conv2d(input=self.mover_conv_list[ind],
                   filter=self.kernel,
                   strides=[1,1,1,1],
                   padding='SAME'
                  )
                self.biases = tf.get_variable(name='bias',shape=(1),
                                        initializer=tf.constant_initializer(
                                            -0.))
                self.bias = tf.nn.bias_add(self.conv, self.biases)
                conv_disp = tf.nn.relu(self.bias)
                conv_disps.append(conv_disp)
        return tf.concat(conv_disps, 3)



class protoModelnetwork():
    def __init__(self, env, h_size, mover_prototypes, mover_disps, blob_size,
                 model_name, dueling=True, lr=0.001, eps=1e-3):
        self.model_name = model_name
        self.blob_size = blob_size
        self.lr = lr
        self.eps = eps
        self.scalarInput =  tf.placeholder(shape=[None,210*160*3*2],dtype=tf.float32)
        self.imageIn = tf.reshape(self.scalarInput/255.,shape=[-1,210,160,3*2])

        self.mover_conv_list = []
        self.disp_conv_list = []

        for i, proto in enumerate(mover_prototypes):
            self.mover_conv_list.append(self.get_conv_mover(proto, i))

        self.conv_movers = tf.concat(self.mover_conv_list,3)

        for i, disps in enumerate(mover_disps):
            self.disp_conv_list.append(self.get_conv_disp(disps, i))

        self.conv_disps = tf.concat(self.disp_conv_list,3)

        self.conv_dm = tf.concat([self.conv_disps,
                                            self.conv_movers],3)

        # want to predict this for next frame
        self.conv_dm_pool = slim.pool(self.conv_dm, [4,4], \
                                         'MAX', 'VALID', stride=[4,4])


        # next frame prediction stuff
        self.out_shape = self.conv_dm_pool.get_shape().as_list()[1:3]
        self.n_ch = self.conv_dm_pool.get_shape().as_list()[-1]
        self.VA_n_ch = (1 + env.action_space.n)

        self.conv1_model = slim.conv2d( \
            inputs=self.conv_dm_pool,num_outputs=self.n_ch,\
                                 kernel_size=[6,6],stride=[1,1],\
                                 padding='SAME')

        Leaky1 = keras.layers.LeakyReLU(0.01)
        self.conv2_model = slim.conv2d( \
            inputs=self.conv1_model,num_outputs=self.VA_n_ch*self.n_ch, \
                                       kernel_size=[6,6],stride=[1,1], \
                                       padding='SAME',
                                      activation_fn=Leaky1)

        self.streams = tf.split(self.conv2_model,
                                             self.VA_n_ch,
                                             3)
        self.streamV = self.streams[0]
        self.streamA = tf.stack(self.streams[1:],4)
        self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions,env.action_space.n,dtype=tf.float32)

        self.pred_pool = self.streamV +\
            (tf.einsum('abcde,ae->abcd',self.streamA,self.actions_onehot))

        target_shape = self.conv_dm_pool.get_shape().as_list()
        self.target_pool = tf.placeholder(shape=target_shape,dtype=tf.float32)


        self.td_error = tf.square(tf.reshape(
            (self.target_pool - self.pred_pool),
                                 [-1]))
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=self.lr,
                                             epsilon=self.eps)
        self.trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                 scope='(?!' + self.model_name +
                               '/piaget)')
        self.updateModel = self.trainer.minimize(self.loss,
                                                 var_list=self.trainables)

        # reward model -- here we're using the same layers as the
        # Q model above, but only one output
        self.conv1_reward = slim.conv2d( \
            inputs=self.conv_dm_pool,num_outputs=16,kernel_size=[4,4],stride=[2,2],
                                 padding='VALID', biases_initializer=None)

        self.conv1_reward_pool = slim.pool(self.conv1_reward, [3,3], \
                                         'MAX', 'VALID', stride=[3,3])


        # for width 250: kernel_size=[10,6]
        # for width 210: kernel_size=[8,6]
        self.conv2_reward = slim.conv2d( \
            inputs=self.conv1_reward_pool,num_outputs=1,kernel_size=[8,6],stride=[1,1],
                                 padding='VALID', biases_initializer=None)

        self.target_reward = tf.placeholder(shape=[None],dtype=tf.float32)

        self.pred_reward = self.conv2_reward[:,0,0,0]

        self.reward_loss = tf.reduce_mean(
            tf.square(self.target_reward - self.pred_reward))

        self.reward_trainer = tf.train.AdamOptimizer(learning_rate=0.01,
                                             epsilon=1e-8)
        self.trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
         scope='(?!' + self.model_name +
                       '/piaget)')

        self.reward_updateModel = self.reward_trainer.minimize(self.reward_loss,
                                                var_list=self.trainables)
    def get_conv_mover(self, proto, ind):

        p0 = proto/(255.)

        p0_mean = np.mean(p0,(0,1))
        p0 = p0 - p0_mean

        p0_norm = np.sqrt(np.sum(p0**2))
        p0 = p0 / p0_norm

        p0_frame0 = np.concatenate([p0, np.zeros_like(p0)],2)
        p0_frame1 = np.concatenate([np.zeros_like(p0), p0],2)

        p0_multi_frame = np.stack([p0_frame0, p0_frame1],3)

        with tf.variable_scope(self.model_name +
                               "/piaget/prototypes/proto" + str(ind)):
            self.kernel = tf.get_variable(name='kernel',
                                        shape=p0_multi_frame.shape,
                                       initializer=tf.constant_initializer(
                                           p0_multi_frame))
            self.conv = tf.nn.conv2d(input=self.imageIn/p0_norm,
                               filter=self.kernel,
                               strides=[1,1,1,1],
                               padding='SAME'
                              )
            self.biases = tf.get_variable(name='bias',shape=(2),
                                    initializer=tf.constant_initializer(
                                        -0.75))
            self.bias = tf.nn.bias_add(self.conv, self.biases)
            self.conv_p0 = tf.nn.relu(self.bias)
            return self.conv_p0

    def get_conv_disp(self, disps, ind):
        blob = local_blob(self.blob_size)

        blob_norm = np.sqrt(np.sum(blob**2))
        blob = blob / blob_norm

        conv_disps = []
        for j, disp in enumerate(disps):
            with tf.variable_scope(self.model_name +
                   "/piaget/mover_disps/mover" + str(ind) +
                    "/disp" + str(j)):

                dx, dy = disp
                blob_frame0 = np.pad(blob,(\
                                             (abs(dy)*(dy<0), abs(dy)*(dy>0)),\
                                             (abs(dx)*(dx<0), abs(dx)*(dx>0))\
                                            ),
                                       'constant')
                blob_frame1 = np.pad(blob,(\
                             (abs(dy)*(dy>0), abs(dy)*(dy<0)),\
                             (abs(dx)*(dx>0), abs(dx)*(dx<0))\
                            ),
                       'constant')

                blob_multi_frame = np.stack(\
                                            [blob_frame0, \
                                             blob_frame1]\
                                            ,2)
                blob_multi_frame = np.expand_dims(blob_multi_frame, 3)

                self.kernel = tf.get_variable(name='kernel',
                                        shape=blob_multi_frame.shape,
                                       initializer=tf.constant_initializer(
                                           blob_multi_frame))
                self.conv = tf.nn.conv2d(input=self.mover_conv_list[ind],
                   filter=self.kernel,
                   strides=[1,1,1,1],
                   padding='SAME'
                  )
                self.biases = tf.get_variable(name='bias',shape=(1),
                                        initializer=tf.constant_initializer(
                                            -0.))
                self.bias = tf.nn.bias_add(self.conv, self.biases)
                conv_disp = tf.nn.relu(self.bias)
                conv_disps.append(conv_disp)
        return tf.concat(conv_disps, 3)
