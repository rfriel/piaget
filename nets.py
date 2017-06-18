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
from itertools import combinations
import cPickle

from pyemd import emd

from PIL import Image
import cv2

from scipy.spatial import distance_matrix

from piaget import *

def local_blob(size):
    blob = np.zeros((size, size))
    middle = (size-1)/2
    for i in range(size):
        for j in range(size):
            blob[i,j] = 1.#/(1+((i-middle)**2 + (j-middle)**2))
    return blob

class protoQnetwork():
    def __init__(self, env, h_size, mover_prototypes, mover_disps,
                 md_equiv_classes, blob_size,
                 model_name, n_base_movers,
                 dueling=True, lr=0.001, eps=1e-3):
        self.model_name = model_name
        self.blob_size = blob_size
        self.lr = lr
        self.eps = eps
        self.n_act = env.action_space.n
        self.n_base_movers = n_base_movers

        self.scalarInput =  tf.placeholder(shape=[None,210*160*3*2],dtype=tf.float32)
        self.imageIn = tf.reshape(self.scalarInput/255.,shape=[-1,210,160,3*2])

        self.mover_conv_list = []
        self.disp_conv_list = []

        for i, proto in enumerate(mover_prototypes):
            self.mover_conv_list.append(self.get_conv_mover(proto, i))

        self.conv_movers = tf.concat(self.mover_conv_list,3)
        self.mover_conv_list_frame1 = [
            t[:,...,1:] for t in self.mover_conv_list]
        self.conv_movers_frame1 = tf.concat(self.mover_conv_list_frame1,3)

        for i, disps in enumerate(mover_disps):
            self.disp_conv_list.append(self.get_conv_disp(disps, i))

        self.conv_disps = tf.concat(self.disp_conv_list,3)

        self.n_disps = self.conv_disps.get_shape().as_list()[-1]

        eq_tensors = []
        channels = tf.split(self.conv_disps, self.n_disps, 3)
        m_id_shift = 0
        for m_id, m_eq in enumerate(md_equiv_classes):
            for eq in m_eq:
                eq_tensors.append(tf.reduce_max(
                    tf.stack(
                        [channels[ch_ind + m_id_shift] for ch_ind in eq],3),
                    3))
            m_id_shift += len(mover_disps[m_id])


        self.cd_equiv = tf.concat(eq_tensors, 3)

        self.conv_dm = tf.concat([self.cd_equiv,
                                            self.conv_movers_frame1],3)

        self.n_ch = self.conv_dm.get_shape().as_list()[-1]

        self.conv_dm_pool = slim.pool(self.conv_dm, [4,4], \
                                         'MAX', 'VALID', stride=[4,4])


        Leaky1 = keras.layers.LeakyReLU(0.1)
        self.conv1 = slim.conv2d( \
            inputs=self.conv_dm_pool,num_outputs=self.n_ch,kernel_size=[3,3],stride=[2,2],
                                 padding='VALID', biases_initializer=None,
                                 activation_fn=Leaky1)

        # self.conv1 = slim.pool( \
        #     self.conv_dm,[4,4],'MAX','VALID',stride=[2,2])


        self.conv1_pool = slim.pool(self.conv1, [3,3], \
                                         'MAX', 'VALID', stride=[3,3])

        Leaky2 = keras.layers.LeakyReLU(0.1)

        # for width 250: kernel_size=[10,6]
        # for width 210: kernel_size=[8,6]
        self.conv2 = slim.conv2d( \
            inputs=self.conv1_pool,num_outputs=h_size,kernel_size=[8,6],stride=[1,1],
                                 padding='VALID', biases_initializer=None,
                                 activation_fn=Leaky2)

        #We take the output from the final convolutional layer and split it into separate advantage and value streams.

        self.streamAC,self.streamVC = tf.split(self.conv2,[3*h_size//4,h_size//4],3)
        self.streamA = slim.flatten(self.streamAC)
        self.streamV = slim.flatten(self.streamVC)
        xavier_init = tf.contrib.layers.xavier_initializer()
        self.AW = tf.Variable(xavier_init([3*h_size//4,self.n_act]))
        self.VW = tf.Variable(xavier_init([h_size//4,1]))

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
        self.actions_onehot = tf.one_hot(self.actions,self.n_act,dtype=tf.float32)

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
                                        -0.9))
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
                                            -0.2))
                self.bias = tf.nn.bias_add(self.conv, self.biases)
                conv_disp = tf.nn.relu(self.bias)
                conv_disps.append(conv_disp)
        return tf.concat(conv_disps, 3)



class protoModelnetwork():
    def __init__(self, env, game_id, model_name,
                 blob_size = 1,
                 dueling=True, lr=0.001, eps=1e-3,
                 frame_h = 210, frame_w = 160,
                 mean_reward_pool=None, dists=None):

        self.game_id = game_id
        self.pt = Prototyper(self.game_id)

        self.model_name = model_name
        self.n_base_movers = self.pt.mt.n_base_movers
        self.blob_size = blob_size
        self.lr = lr
        self.eps = eps
        self.frame_h = frame_h
        self.frame_w = frame_w
        self.n_act = (env.action_space.n)**2

        self.mean_reward_pool = mean_reward_pool
        self.dists = dists

        self.scalarInput =  tf.placeholder(shape=[None,frame_h*frame_w*3*2],dtype=tf.float32)
        self.imageIn = tf.reshape(self.scalarInput/255.,shape=[-1,frame_h,frame_w,3*2])

        self.mover_conv_list = []
        self.disp_conv_list = []
        self.disp_filter_list = []

        for i, proto in enumerate(self.pt.mover_prototypes):
            self.mover_conv_list.append(self.get_conv_mover(proto, i))

        self.conv_movers = tf.concat(self.mover_conv_list,3)
        self.mover_conv_list_frame1 = [
            self.conv_movers[:,...,2*i+1:2*i+2]
            for i in range(len(self.pt.mover_prototypes))]
        self.conv_movers_frame1 = tf.concat(self.mover_conv_list_frame1,3)

        self.n_movers = self.conv_movers_frame1.get_shape().as_list()[-1]

        for i, disps in enumerate(self.pt.mover_disps):
            self.disp_conv_list.append(self.get_conv_disp(disps, i))

        self.conv_disps = tf.concat(self.disp_conv_list,3)

        # self.conv_dm = tf.concat([self.conv_disps,
        #                                      self.conv_movers],3)

        # next frame prediction stuff
        self.n_disps = self.conv_disps.get_shape().as_list()[-1]

        d_id_shift = 0
        for i, disps in enumerate(self.pt.mover_disps):
            self.disp_filter_list.append(self.get_filter_disp(disps, i,
                                                              d_id_shift,
                                                              (11,11)))
            d_id_shift += len(disps)
        self.filter_disps = np.concatenate(self.disp_filter_list, 3)
        # eq_tensors = []
        # channels = tf.split(self.conv_disps, self.n_disps, 3)
        # m_id_shift = 0
        # for m_id, m_eq in enumerate(md_equiv_classes):
        #     for eq in m_eq:
        #         eq_tensors.append(tf.reduce_max(
        #             tf.stack(
        #                 [channels[ch_ind + m_id_shift] for ch_ind in eq],3),
        #             3))
        #     m_id_shift += len(self.pt.mover_disps[m_id])
        # self.cd_equiv = tf.concat(eq_tensors,3)
        self.cd_equiv = tf.concat(self.conv_disps,3)
        self.conv_dm = tf.concat([self.cd_equiv,
                                            self.conv_movers],3)
        #self.cdp_equiv = self.conv_dm_pool
        self.out_shape = self.conv_movers_frame1.get_shape().as_list()
        self.out_shape[-1] = self.n_base_movers
        self.n_ch_out = self.n_base_movers#self.conv_movers_frame1.get_shape().as_list()[-1]

        f0x, f1x = self.get_position_filters(0,2)
        f0y, f1y = self.get_position_filters(1,2)
        f2x, f3x = self.get_position_filters(0,4)
        f2y, f3y = self.get_position_filters(1,4)
        f4x, f5x = self.get_position_filters(0,8)
        f4y, f5y = self.get_position_filters(1,8)
        f6x, f7x = self.get_position_filters(0,16)
        f6y, f7y = self.get_position_filters(1,16)
        fT, fB, fL, fR = self.get_edge_filters()
        # self.pos_filters = tf.stack([f0x, f1x, f0y, f1y, \
        #                     f2x, f3x, f2y, f3y, \
        #                     f4x, f5x, f4y, f5y, \
        #                     f6x, f7x, f6y, f7y, \
        #                     fT, fB, fL, fR],
        #                             3)

        # self.pos_filters = tf.stack([f0x, f1x, f0y, f1y, \
        #                     f2x, f3x, f2y, f3y, \
        #                     fT, fB, fL, fR],
        #                             3)

        self.pos_filters = tf.stack([
                            fT, fB, fL, fR],
                                    3)

        #self.conv_collisions = self.get_conv_collisions()

        # self.cd_with_pos = tf.concat([self.conv_dm,
        #                                 self.pos_filters,
        #                                 self.conv_collisions],3)
        self.cd_with_pos = tf.concat([self.conv_dm,
                                        self.pos_filters],3)

        self.n_ch_in = self.cd_with_pos.get_shape().as_list()[-1]
        self.VA_n_ch = (1 + self.n_act)

        # self.conv1_model = self.cd_with_pos
        Leaky1 = keras.layers.LeakyReLU(0.1)
        self.conv1_model = slim.conv2d( \
            inputs=self.cd_with_pos,num_outputs=self.n_ch_in,\
                                 kernel_size=[5,5],stride=[1,1],\
                                 padding='SAME',
                                activation_fn=Leaky1)


        Leaky2 = keras.layers.LeakyReLU(0.1)
        self.conv2_model = slim.conv2d( \
            inputs=self.conv1_model,num_outputs=self.VA_n_ch*self.n_ch_out, \
                                       kernel_size=[5,5],stride=[1,1], \
                                       padding='SAME',
                                      activation_fn=Leaky2)

        self.streams = tf.split(self.conv2_model,
                                             self.VA_n_ch,
                                             3)
        self.streamV = self.streams[0]
        self.streamA = tf.stack(self.streams[1:],4)
        self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
        self.old_actions = tf.placeholder(shape=[None],dtype=tf.int32)

        self.actions_ind = self.actions*(env.action_space.n) + self.old_actions
        self.actions_onehot = tf.one_hot(self.actions_ind,self.n_act,dtype=tf.float32)

        self.pred = self.streamV +\
            (tf.einsum('abcde,ae->abcd',self.streamA,self.actions_onehot))

        self.target = tf.placeholder(shape=self.out_shape,dtype=tf.float32)

        self.cm_pool = self.conv_movers_frame1[:,...,:self.n_base_movers]
        self.pred_pool = self.pred
        self.target_pool = self.target

        # self.cm_pool = slim.pool(self.conv_movers_frame1, [4,4], \
        #                            'MAX', 'VALID', stride=[4,4])
        #
        # self.pred_pool = slim.pool(self.pred, [4,4], \
        #                            'MAX', 'VALID', stride=[4,4])
        #
        # self.target_pool = slim.pool(self.target, [4,4], \
        #                            'MAX', 'VALID', stride=[4,4])

        # MSE loss
        # self.td_error = tf.square(tf.reshape(
        #     (self.target_pool - self.pred_pool),
        #                          [-1]))
        # self.loss = tf.reduce_mean(self.td_error)

        # log loss
        labels = tf.cast(self.target_pool>0.1,'int32')
        logits = tf.stack(\
                          [10*tf.ones_like(self.pred_pool),
                           self.pred_pool],
                          4)
        batch_size = tf.shape(labels)[0]
        raw_log_loss_list = tf.split(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits),
            self.n_base_movers, 3)
        self.mover_log_losses = tf.stack([tf.reduce_sum(
            tf.reshape(m_loss, [batch_size,-1]),1)
                  for m_loss in raw_log_loss_list],1)
        # self.log_losses = tf.reshape(
        #     tf.nn.sparse_softmax_cross_entropy_with_logits(
        #     labels=labels, logits=logits),
        #     [batch_size,-1])
        # self.frame_losses = tf.reduce_sum(self.log_losses,1)
        self.loss = tf.reduce_sum(self.mover_log_losses)

        self.trainer = tf.train.AdamOptimizer(learning_rate=self.lr,
                                             epsilon=self.eps)
        self.trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                 scope='(?!' + self.model_name +
                               '/piaget)')
        self.updateModel = self.trainer.minimize(self.loss,
                                                 var_list=self.trainables)

        # emd-based q estimates
        if self.mean_reward_pool is not None:
            if self.dists is None:
                self.make_distance_matrix()

            self.demands = []
            for ii in range(self.n_ch):
                demand = np.asarray(np.reshape(mean_reward_pool[:,:,ii], (52*40,))\
                                .astype('float64'), order='C')
                demand = demand / np.sum(demand)
                self.demands.append(demand)

            self.act_frames = self.streamA + \
            tf.stack([self.streamV for i in range(self.n_act)],4)

    def get_Q(self, batch):

        self.act_emd_batch = [self.get_emds(act_frame) for act_frame in batch]

        self.Qout = [self.emd_to_q(np.mean(act_emds,axis=1)) for act_emds \
                     in self.act_emd_batch]
        self.predict = [np.argmax(Q) for Q in self.Qout]

    def get_emds(self, act_frame):
        act_emds = np.zeros((self.n_act, self.n_ch))
        for act_ind in range(self.n_act):
            for ch_ind in range(self.n_ch):
                #print(act_ind, ch_ind)

                supply = \
                np.asarray(np.reshape(act_frame[:,:,ch_ind,act_ind], \
                                      (52*40,)
                                      ).astype('float64'), order='C')
                supply = (supply + np.abs(supply)) / 2
                demand = self.demands[ch_ind]

                if min(np.sum(demand), np.sum(supply)) > 0:
                    supply = supply / np.sum(supply)

                    act_emd = emd(supply, demand, self.dists)
                    act_emds[act_ind, ch_ind] = act_emd

        for act_ind in range(self.n_act):
            for ch_ind in range(self.n_ch):
                if act_emds[act_ind, ch_ind] == 0:
                    act_emds[act_ind, ch_ind] = \
                    np.mean(act_emds[act_ind, :][act_emds[act_ind, :] > 0])

        return act_emds

    def emd_to_q(self, emds):
        return 1./emds

    def make_distance_matrix(self):
        x_coords = np.array([[j for j in range(40)] for i in range(52)])
        y_coords = np.array([[i for j in range(40)] for i in range(52)])

        x_flat = np.reshape(x_coords,(52*40,))
        y_flat = np.reshape(y_coords,(52*40,))
        coords_flat = [[x,y] for x, y in zip(x_flat, y_flat)]

        dists = distance_matrix(coords_flat,coords_flat)
        self.dists = np.asarray(dists, order='C')

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
                                        -0.9))
            self.bias = tf.nn.bias_add(self.conv, self.biases)
            self.conv_p0 = tf.nn.relu(self.bias)
            self.conv_p0 = tf.cast(self.conv_p0>0.1,'float32')
            return self.conv_p0

    def get_position_filters(self, axis, reps, weight=0.25):
        filter0 = np.zeros(self.out_shape[1:3],dtype='float32')
        filter1 = np.zeros(self.out_shape[1:3],dtype='float32')

        if axis == 1:
            filter0 = filter0.T
            filter1 = filter1.T

        len_to_fill = filter0.shape[0]
        len_per_band = len_to_fill // reps

        for ii in range(len_to_fill):
            if (ii // len_per_band) % 2 == 0:
                filter0[ii,:] = weight
            else:
                filter1[ii,:] = weight

        if axis == 1:
            filter0 = filter0.T
            filter1 = filter1.T

        f0_tensor = tf.zeros_like(self.conv_disps[:,...,0]) + tf.convert_to_tensor(filter0)
        f1_tensor = tf.zeros_like(self.conv_disps[:,...,0]) + tf.convert_to_tensor(filter1)

        return f0_tensor, f1_tensor

    def get_edge_filters(self, weight=0.25, extent=0.1):
        filterT = np.zeros(self.out_shape[1:3],dtype='float32')
        filterB = np.zeros(self.out_shape[1:3],dtype='float32')
        filterL = np.zeros(self.out_shape[1:3],dtype='float32')
        filterR = np.zeros(self.out_shape[1:3],dtype='float32')

        extent_h = int(self.frame_h * extent)
        extent_w = int(self.frame_w * extent)

        filterT[:extent_h,:] = np.expand_dims(weight * np.linspace(1,0,extent_h)
                                              ,1)
        filterB[-extent_h:,:] = np.expand_dims(weight * np.linspace(0,1,extent_h)
                                               ,1)
        filterL[:,:extent_w] = np.expand_dims(weight * np.linspace(1,0,extent_w)
                                              ,0)
        filterR[:,-extent_w:] = np.expand_dims(weight * np.linspace(0,1,extent_w)
                                                ,0)

        fT_tensor = tf.zeros_like(self.conv_disps[:,...,0]) + tf.convert_to_tensor(filterT)
        fB_tensor = tf.zeros_like(self.conv_disps[:,...,0]) + tf.convert_to_tensor(filterB)
        fL_tensor = tf.zeros_like(self.conv_disps[:,...,0]) + tf.convert_to_tensor(filterL)
        fR_tensor = tf.zeros_like(self.conv_disps[:,...,0]) + tf.convert_to_tensor(filterR)

        return fT_tensor, fB_tensor, fL_tensor, fR_tensor

    def get_conv_collisions(self):
        mover_pairs = combinations(range(self.n_movers),2)
        conv_collisions = []

        for pair in mover_pairs:
            with tf.variable_scope(self.model_name +
                   "/piaget/collisions/mover" + str(pair[0]) +
                    "mover" + str(pair[1])):
                coll_kernel = np.zeros((2,2,self.n_movers))
                coll_kernel[:,:,pair[0]] = 1
                coll_kernel[:,:,pair[1]] = 1
                coll_kernel = np.expand_dims(coll_kernel,3)

                kernel = tf.get_variable(name='kernel',
                                        shape=coll_kernel.shape,
                                       initializer=tf.constant_initializer(
                                           coll_kernel))

                conv = tf.nn.conv2d(input=self.conv_movers_frame1,
                   filter=kernel,
                   strides=[1,1,1,1],
                   padding='SAME'
                  )
                self.biases = tf.get_variable(name='bias',shape=(1),
                                        initializer=tf.constant_initializer(
                                            -0.2))
                self.bias = tf.nn.bias_add(self.conv, self.biases)
                conv_collision = tf.nn.relu(self.bias)
                conv_collisions.append(conv_collision)
            return tf.concat(conv_collisions,3)

    def get_conv_disp(self, disps, ind):
        blob = local_blob(self.blob_size)

        blob_norm = np.sqrt(np.sum(blob**2))
        #blob = blob / blob_norm

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

                # special treatment for (0,0) motions bc they indicate
                # movers appearing and disappearing
                appear_disappear = ((dx == 0) and (dy == 0))
                if not appear_disappear:
                    blob_multi_frame = np.stack([blob_frame0, blob_frame1],2)
                    # blob_multi_frame = np.stack(\
                    #                             [blob_frame0-blob_frame1, \
                    #                              blob_frame1-blob_frame0]\
                    #                             ,2)
                    blob_multi_frame = np.expand_dims(blob_multi_frame, 3)
                    init_bias = np.array([-1.])
                else:
                    blob_multi_frame_appear = np.stack(
                        [-blob_frame0, blob_frame1],2)
                    blob_multi_frame_disappear = np.stack(
                        [blob_frame0, -blob_frame1],2)
                    blob_multi_frame = np.stack([blob_multi_frame_appear,
                                                 blob_multi_frame_disappear],
                                                3)
                    init_bias = np.array([0.,0])
                self.kernel = tf.get_variable(name='kernel',
                                        shape=blob_multi_frame.shape,
                                       initializer=tf.constant_initializer(
                                           blob_multi_frame))
                mover = self.conv_movers[:,...,2*ind:2*ind+2]
                self.conv = tf.nn.conv2d(input=mover,
                   filter=self.kernel,
                   strides=[1,1,1,1],
                   padding='SAME'
                  )
                self.biases = tf.get_variable(name='bias',shape=init_bias.shape,
                                        initializer=tf.constant_initializer(
                                            init_bias))
                self.bias = tf.nn.bias_add(self.conv, self.biases)
                conv_disp = tf.nn.relu(self.bias)
                conv_disps.append(conv_disp)
        return tf.concat(conv_disps, 3)

    def get_filter_disp(self, disps, ind, d_id_shift, filter_shape, weight=10):
        filter_disps = []
        for i, disp in enumerate(disps):
            filter_disp = np.zeros(filter_shape + (self.n_disps, 1),
                                   dtype='float32')

            source_x = filter_shape[0]//2 - disp[0]
            source_y = filter_shape[1]//2 - disp[1]

            filter_disp[source_x,source_y,i+d_id_shift,0] = weight
            filter_disps.append((filter_disp))
        filter_disps_combined = np.sum(filter_disps,0)
        return filter_disps_combined
