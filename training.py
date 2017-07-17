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
from nets import *

# note: much of this code is adapted from Arthur Juliani's DQN code
# see README and demo notebook for details

def softmax20(x):
    e_x = np.exp(x)
    return e_x / (e_x + np.exp(20))

# experience_buffer class below is Juliani's code but with some
# experimental modifications

class experience_buffer():
    def __init__(self, buffer_size = 50000):
        self.buffer = []
        self.buffer_size = buffer_size
        self.reward_prop_frames = 6

    def add(self,experience,reward_prop):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size] = []
        if reward_prop and len(self.buffer) > self.reward_prop_frames:
            for i in range(1,self.reward_prop_frames+1):
                self.buffer[-i][2] += 2**(-i)*experience[0,2]
        self.buffer.extend(experience)

    def sample(self,size,attention=False,rate=1):
        if attention:
            cont_ind = np.random.rand(size)
            cont_ind_exp = (np.log(1+rate*cont_ind))*(len(self.buffer)/(np.log(1+rate)))
            disc_ind_exp = cont_ind_exp.astype('int32')

            np_buffer = np.array(self.buffer)
            sort_inds = np.argsort(np_buffer[:,6])
            sample_exps = np_buffer[sort_inds][disc_ind_exp]
            return sample_exps, sort_inds[disc_ind_exp]
        else:
            return np.reshape(np.array(random.sample(self.buffer,size)),[size,7])
        #TESTING continguous samples -- this is dangerous!
        #start_index = random.randint(0,len(self.buffer)-size)
        #return np.array(myBuffer.buffer[start_index:start_index+128])

    def set_losses(self,indices,losses):
        for ind, loss in zip(indices, losses):
            self.buffer[ind][6] = loss

def updateTargetGraph(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
    return op_holder

def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)

def processState(states):
    return np.reshape(states,[np.product(states.shape)])

def initial_obs(env, breakout=False):
    env.reset()
    if breakout:
        s, r, d, info = env.step(1) # request next ball in breakout
    else:
        s, r, d, info = env.step(np.random.randint(0,env.action_space.n))
    s1, r1, d1, info1 = env.step(np.random.randint(0,env.action_space.n))

    s_list = [s, s1]

    return s_list, r1, d1, info1


def train_protoModelNetwork(env, pt, breakout=False,
                        batch_size=32, update_freq=4, y=.75, startE=1,
                        endE=0.1, anneling_steps=1000000.,
                        num_episodes=10000, max_epLength=5000,
                        pre_train_steps=5000, load_model=False,
                        path="./dqn/piaget", h_size=24, reset_freq=5000,
                        reward_prop=False, learning_rate=0.0025,
                        adam_eps=1e-5,
                        act_repeat_len=0, act_init_len=0,
                        attn_start=1e9, attn_rate=100,
                        burnin_batches_init=250, burnin_batches_new_mover=250,
                        outlier_sample_min_batches=25, new_mover_thresh=5,
                        loss_thresh=0, n_free=0,
                        free_kernel_batches=15000000):

    # making params that depend on batch_size
    burnin_frames_init = batch_size * burnin_batches_init
    burnin_frames_new_mover = batch_size * burnin_batches_new_mover
    outlier_sample_min_size = batch_size * outlier_sample_min_batches
    free_kernel_frames = batch_size * free_kernel_batches

    tf.reset_default_graph()

    myBuffer = experience_buffer(buffer_size=50000)
    buffer_pushes = 0

    #Set the rate of random action decrease.
    e = startE
    stepDrop = (startE - endE)/anneling_steps

    #create lists to contain total rewards and steps per episode
    jList = []
    rList = []
    total_steps = 0

    #statistics on Q
    Qpred_list = []
    Qtarget_list = []
    QrList = []

    frame_err_list = np.zeros((0, pt.mt.n_base_movers))
    reference_err_list = []
    reward_err_list = []

    mean_reward_pool = None

    #Make a path for our model to be saved in.
    if not os.path.exists(path):
        os.makedirs(path)

    sess = tf.Session()

    QN_index = 0

    if breakout:
        frame_h = 210
    else:
        frame_h = 250

    mainQN = protoModelnetwork(env, pt, 'mainQN',
                dueling=True,
                lr=learning_rate, eps=adam_eps,
                               bg=None,
                              n_free_kernels=n_free,
                              frame_h=frame_h)

    init = tf.global_variables_initializer()
    sess.run(init)

    saver = tf.train.Saver()

    if load_model == True:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess,ckpt.model_checkpoint_path)

    #updateTarget(targetOps,sess) #Set the target network to be equal to the primary network.
    for i in range(num_episodes):
        episodeBuffer = experience_buffer()
        #Reset environment and get first new observation(s)
        s_list, r, d, info = initial_obs(env, breakout)
        s_stack = np.dstack(s_list)
        s = processState(s_stack)
        if breakout:
            a = 1
        else:
            # fix this
            a = 0

        d = False
        rAll = 0
        j = 0

        # periodically evaluate
        if False:#i % 5 == 0 and total_steps  > pre_train_steps:
            eval_episode = True
            saved_e = e
            e = 0.1
        else:
            eval_episode = False

        act_repeat_countdown = 0
        breakout_requesting = False
        while j < max_epLength: #If the agent takes longer than 200 moves to reach either of the blocks, end the trial.
            j+=1

            save_to_buffer = True
            old_a = a

            if breakout and breakout_requesting:
                # prevent from saving the frame pair where the ball appears
                save_to_buffer = False
                breakout_requesting = False
            if breakout and np.array_equal(s_list[0],s_list[1]):
                # breakout: requesting next ball
                a = 1
                save_to_buffer = False
                breakout_requesting = True
            elif act_repeat_countdown == 0:
                a = np.random.randint(0,env.action_space.n)
                for _ in range(act_init_len):
                    s_next,r,d,info = env.step(a)
                    s_list.pop(0)
                    s_list.append(s_next)
                    s_stack = np.dstack(s_list)
                    s = processState(s_stack)
                    old_a = a
                act_repeat_countdown = act_repeat_len
            else:
                act_repeat_countdown -= 1

            s_next,r,d,info = env.step(a)

            s_list.pop(0)
            s_list.append(s_next)

            s_stack = np.dstack(s_list)
            s1 = processState(s_stack)

            total_steps += 1

    #         target_pool = sess.run(mainQN.cm_pool,feed_dict={mainQN.scalarInput:[s1]})
    #         loss = sess.run(mainQN.loss, \
    #             feed_dict={mainQN.scalarInput:[s],
    #                        mainQN.target:target_pool,
    #                        mainQN.actions:[a],
    #                        mainQN.old_actions:[old_a]})

            if (r > 0) or d:
                buffer_repeat = 1
            else:
                buffer_repeat = 1
            if save_to_buffer:
                buffer_pushes += 1
                for dummy in range(buffer_repeat):
                    episodeBuffer.add(np.reshape(np.array([s,a,r,s1,d,old_a,0]),
                                                 [1,7]),reward_prop) #7th entry is loss if we are tracking it for every frame

            if total_steps > pre_train_steps and not eval_episode:
                if mean_reward_pool is None:

    #                 bootstrapQN = protoModelnetwork(env, h_size, mover_prototypes, \
    #                        mover_disps, md_equiv_classes, 5, 'bootstrapQN', dueling=True,
    #                           lr=0.001, eps=1e-3, mean_reward_pool=None)
    #                 init = tf.global_variables_initializer()
    #                 sess.run(init)

                    #reward_frames = [f[3] for f in myBuffer.buffer if f[2]>0.5]
                    #reward_pools = sess.run(bootstrapQN.conv_dm,feed_dict={bootstrapQN.scalarInput:reward_frames})

                    #mean_reward_pool = np.mean(reward_pools,axis=0)
                    mean_reward_pool = True # hack, get rid of this once we care

                if e > endE:
                    e -= stepDrop

                if total_steps % (update_freq) == 0:
                    trainBatchFull = np.zeros((0, 7))
                    targetPoolFull = np.zeros((0, frame_h, 160, pt.mt.n_base_movers))
                    while trainBatchFull.shape[0] < batch_size:
                        if True:#total_steps < attn_start:
                            trainBatch = myBuffer.sample(batch_size, attention=False)
                        else:
                            trainBatch, _ = myBuffer.sample(batch_size, attention=True,
                                                                   rate=attn_rate)
                        if QN_index == 0:
                            target_pool = sess.run(mainQN.cm_pool,feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,3])})
                        else:
                            target_pool = sess.run(mainQN.free_kernels,feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,3])})

                        mover_log_losses = sess.run(mainQN.mover_log_losses, \
                                feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,0]),
                                           mainQN.target:target_pool,
                                           mainQN.actions:trainBatch[:,1],
                                           mainQN.old_actions:trainBatch[:,5]})
                        train_inds = (np.max(mover_log_losses - loss_thresh,1) > 0)
                        trainBatch = trainBatch[train_inds,:]
                        target_pool = target_pool[train_inds,:,...,:]
                        trainBatchFull = np.concatenate([trainBatchFull,
                                                         trainBatch[:(batch_size-trainBatchFull.shape[0]),:]],0)
                        targetPoolFull = np.concatenate([targetPoolFull,
                                         target_pool[:(batch_size-targetPoolFull.shape[0]),:,...,:]],0)

                    trainBatch = trainBatchFull
                    target_pool = targetPoolFull
                    frame_err_list = np.concatenate([frame_err_list, mover_log_losses])

                    if False:#len(frame_err_list) > free_kernel_frames and QN_index == 0:
                        QN_index += 1
                        prev_filter_counts = (mainQN.n_movers, [len(d) for d in pt.mover_disps])

                        mainQN = protoModelnetwork(env, pt, 'mainQN',
                        dueling=True,
                        lr=learning_rate, eps=adam_eps,
                           mean_reward_pool=None,
                        n_free_kernels=n_free,
                        existing_filters_counts=prev_filter_counts,
                        net_index=QN_index,
                        train_free_kernels=False)

                        tf_vars = tf.global_variables()
                        tf_initialized = sess.run([tf.is_variable_initialized(v) for v in tf_vars])
                        tf_vars_uninitialized = [v for i, v in enumerate(tf_vars) if not tf_initialized[i]]
                        sess.run(tf.variables_initializer(tf_vars_uninitialized))


                    if False:#len(frame_err_list) > (burnin_frames_init + QN_index*burnin_frames_new_mover):
                        worst_mover = np.random.randint(mainQN.n_base_movers)#worst_frame_ind[1]
                        worst_frame_ind = np.argmax(mover_log_losses[:,worst_mover])
    #                     worst_frame_ind = np.unravel_index(np.argmax(mover_log_losses),
    #                                                        mover_log_losses.shape)
                        worst_frame = trainBatch[worst_frame_ind,:]
                        current_thresh = np.median(frame_err_list[-outlier_sample_min_size:,worst_mover]) +\
                        new_mover_thresh*np.std(frame_err_list[-outlier_sample_min_size:,worst_mover])
                        if mover_log_losses[worst_frame_ind, worst_mover] > current_thresh:
                            print 'threshold exceeded'
                            plt.figure(figsize=(12,4))
                            wf_s0 = np.reshape(worst_frame[0],(frame_h,160,6))
                            wf_s1 = np.reshape(worst_frame[3],(frame_h,160,6))
                            plt.subplot(131)
                            plt.imshow(wf_s0[:,:,3:] - wf_s0[:,:,:3])
                            plt.subplot(132)
                            plt.imshow(wf_s0[:,:,:3])
                            plt.subplot(133)
                            plt.imshow(wf_s0[:,:,3:])
                            plt.show()
                            print ('loss',mover_log_losses[worst_frame_ind, worst_mover],'threshold', current_thresh)
                            # take snapshots of outliers
                            n_movers_before = len(pt.mover_prototypes)
                            prev_filter_counts = (mainQN.n_movers, [len(d) for d in pt.mover_disps])

                            print 'mt has %d movers, QN has %d' % (n_movers_before, mainQN.n_movers)
                            wf_fp0 = FramePair(wf_s0[:,:,:3],wf_s0[:,:,3:],0,0)
                            wf_fp1 = FramePair(wf_s1[:,:,:3],wf_s1[:,:,3:],0,0)
                            pt.mt.process_frame_pair(wf_fp0, base_movers=False)
                            pt.mt.process_frame_pair(wf_fp1, base_movers=False)

                            with open(pt.mt_filename,'w') as mt_file:
                                cPickle.dump(pt.mt, mt_file)

                            pt.prototype_game(game_id)
                            n_movers_after = len(pt.mover_prototypes)
                            if (n_movers_after - n_movers_before) > 0:
                                QN_index += 1

                                print 'found %d new movers' % (n_movers_after - n_movers_before)

                                mainQN = protoModelnetwork(env, pt, 'mainQN',
                                dueling=True,
                                lr=learning_rate, eps=adam_eps,
                                   mean_reward_pool=None,
                                n_free_kernels=n_free,
                                existing_filters_counts=prev_filter_counts,
                                net_index=QN_index)

                                tf_vars = tf.global_variables()
                                tf_initialized = sess.run([tf.is_variable_initialized(v) for v in tf_vars])
                                tf_vars_uninitialized = [v for i, v in enumerate(tf_vars) if not tf_initialized[i]]
                                sess.run(tf.variables_initializer(tf_vars_uninitialized))

                    #frame_err_list.append(sum(((target_pool-pred_pool)**2).flatten()))
                    #reference_err_list.append(sum(((target_pool-previous_pool)**2).flatten()))
                    #update_ops = [mainQN.reward_loss, mainQN.updateModel, mainQN.reward_updateModel]

                    if ((len(frame_err_list) // free_kernel_frames) % 2) == 1:
                        update_ops = [mainQN.updateModel_free]
                    else:
                        update_ops = [mainQN.updateModel]
                    if trainBatch.shape[0] > 0:
                        (_) = \
                        sess.run(update_ops, \
                            feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,0]),
                                       mainQN.target:target_pool,
                                       mainQN.actions:trainBatch[:,1],
                                       mainQN.old_actions:trainBatch[:,5]})
                    #reward_err_list.append(reward_loss)

            rAll += r
            s = s1

            if d == True:
                break
        if not eval_episode:
            myBuffer.add(episodeBuffer.buffer,False)
            jList.append(j)
            rList.append(rAll)
        else:
            print("Evaluated model")
            print(j, rAll)
            e = saved_e
        #Periodically save the model.
        if i % 100 == 0 and i > 0:
            saver.save(sess,path+'/model-'+str(i)+'.cptk')
            print("Saved Model")
        print_rate = 1
        if len(rList) % print_rate == 0 and total_steps > pre_train_steps:
            print 'total_steps: %d' % total_steps
            print 'mean log loss (last 100 training frames): %d ' %
            (total_steps, np.mean(frame_err_list[-batch_size*100:,:]))
            n_example_frames = 2
            print 'Displaying model performance on %d random frames from buffer...'  % n_example_frames
            for _ in range(n_example_frames):
                displayBatch = myBuffer.sample(1, attention=False)
                target_pool = sess.run(mainQN.cm_pool,feed_dict={mainQN.scalarInput:np.vstack(displayBatch[:,3])})
                pred_pool = sess.run(mainQN.pred_pool,\
                                     feed_dict={mainQN.scalarInput:np.vstack(displayBatch[:,0]),
                                                mainQN.actions:displayBatch[:,1],
                                                mainQN.old_actions:displayBatch[:,5]})
                previous_pool = sess.run(mainQN.cm_pool,feed_dict={mainQN.scalarInput:np.vstack(displayBatch[:,0])})

                pred_V = sess.run(mainQN.streamV,\
                         feed_dict={mainQN.scalarInput:np.vstack(displayBatch[:,0]),
                                    mainQN.actions:displayBatch[:,1],
                                    mainQN.old_actions:displayBatch[:,5]})
                pred_A = sess.run(tf.einsum('abcde,ae->abcd',mainQN.streamA,mainQN.actions_onehot),\
                                     feed_dict={mainQN.scalarInput:np.vstack(displayBatch[:,0]),
                                                mainQN.actions:displayBatch[:,1],
                                                mainQN.old_actions:displayBatch[:,5]})


                #print(total_steps, np.mean(reference_err_list[-100:]), np.mean(frame_err_list[-100:]))
                print 'action: %d, previous action: %d' % (displayBatch[0][1],
                                                           displayBatch[0][5])

                plt.figure(figsize=(12,4))
                s0 = np.reshape(displayBatch[0,0],(frame_h,160,6))
                s1 = np.reshape(displayBatch[0,3],(frame_h,160,6))
                plt.subplot(131)
                plt.imshow(s0[:,:,3:] - s0[:,:,:3])
                plt.title('Difference');
                plt.subplot(132)
                plt.imshow(s0[:,:,:3])
                plt.title('First frame');
                plt.subplot(133)
                plt.imshow(s0[:,:,3:])
                plt.title('Second frame');

                i_max = target_pool.shape[3]
                j_max = 5
                for ii in range(i_max):
                    target_img = (target_pool[0,:,:,ii]>0.)
                    pred_img = pred_pool[0,:,:,ii]
                    previous_img = (previous_pool[0,:,:,ii]>0.)

                    predV_img = pred_V[0,:,:,ii]
                    predA_img = pred_A[0,:,:,ii]

                    vis_center = np.unravel_index(np.argmax(target_img), target_img.shape)
                    vis_center = (max(vis_center[0],5), max(vis_center[1],5))
                    target_img = target_img[vis_center[0]-5:vis_center[0]+5,
                                           vis_center[1]-5:vis_center[1]+5]
                    pred_img = pred_img[vis_center[0]-5:vis_center[0]+5,
                                           vis_center[1]-5:vis_center[1]+5]
                    previous_img = previous_img[vis_center[0]-5:vis_center[0]+5,
                                           vis_center[1]-5:vis_center[1]+5]
                    predV_img = predV_img[vis_center[0]-5:vis_center[0]+5,
                       vis_center[1]-5:vis_center[1]+5]
                    predA_img = predA_img[vis_center[0]-5:vis_center[0]+5,
                                           vis_center[1]-5:vis_center[1]+5]

                    #cmap_max = max(np.max(previous_img), np.max(target_img))
                    #cmap_min = min(np.min(previous_img), np.min(target_img))
                    cmap_max = 1; cmap_min = 0
                    cmap_max_VA = max(np.max(predV_img), np.max(predA_img))
                    cmap_min_VA = min(np.min(predV_img), np.min(predA_img))

                    plt.figure(figsize=(12,2*2))
                    plt.subplot(151)
                    plt.imshow(s0[vis_center[0]-5:vis_center[0]+5,
                                  vis_center[1]-5:vis_center[1]+5,:3],
                              interpolation='nearest')
                    plt.xticks(np.linspace(-0.5,9.5,11),range(10))
                    plt.yticks(np.linspace(-0.5,9.5,11),range(10))
                    plt.grid(color='w',lw=1,ls='-',alpha=0.5)

                    plt.figure(figsize=(12,2*2))
                    plt.subplot(151)
                    plt.imshow(s0[vis_center[0]-5:vis_center[0]+5,
                                  vis_center[1]-5:vis_center[1]+5,3:],
                              interpolation='nearest')
                    plt.xticks(np.linspace(-0.5,9.5,11),range(10))
                    plt.yticks(np.linspace(-0.5,9.5,11),range(10))
                    plt.grid(color='w',lw=1,ls='-',alpha=0.5)
                    plt.subplot(152)
                    plt.imshow(s1[vis_center[0]-5:vis_center[0]+5,
                                  vis_center[1]-5:vis_center[1]+5,3:],
                              interpolation='nearest')
                    plt.xticks(np.linspace(-0.5,9.5,11),range(10))
                    plt.yticks(np.linspace(-0.5,9.5,11),range(10))
                    plt.grid(color='w',lw=1,ls='-',alpha=0.5)
                    plt.show()

                    plt.figure(figsize=(12,2*i_max))
                    for jj in range(j_max//5):


                        plt.subplot(i_max,j_max,3*jj+(ii*j_max)+1)
                        plt.imshow(previous_img,cmap='gray',interpolation='nearest',
                                  vmin=cmap_min, vmax=cmap_max)
                        plt.xticks(np.linspace(-0.5,9.5,11),range(10))
                        plt.yticks(np.linspace(-0.5,9.5,11),range(10))
                        plt.grid(color='w',lw=1,ls='-',alpha=0.5)

                        plt.subplot(i_max,j_max,3*jj+(ii*j_max)+2)
                        plt.imshow(target_img,cmap='gray',interpolation='nearest',
                                  vmin=cmap_min, vmax=cmap_max)
                        plt.xticks(np.linspace(-0.5,9.5,11),range(10))
                        plt.yticks(np.linspace(-0.5,9.5,11),range(10))
                        plt.grid(color='w',lw=1,ls='-',alpha=0.5)

                        plt.subplot(i_max,j_max,3*jj+(ii*j_max)+3)
                        plt.imshow(softmax20(pred_img),cmap='gray',interpolation='nearest',
                                  vmin=0, vmax=1)
                        plt.xticks(np.linspace(-0.5,9.5,11),range(10))
                        plt.yticks(np.linspace(-0.5,9.5,11),range(10))
                        plt.grid(color='w',lw=1,ls='-',alpha=0.5)

                        plt.subplot(i_max,j_max,3*jj+(ii*j_max)+4)
                        plt.imshow(predV_img,cmap='gray',interpolation='nearest',
                                  vmin=cmap_min_VA, vmax=cmap_max_VA)
                        plt.xticks(np.linspace(-0.5,9.5,11),range(10))
                        plt.yticks(np.linspace(-0.5,9.5,11),range(10))
                        plt.grid(color='w',lw=1,ls='-',alpha=0.5)

                        plt.subplot(i_max,j_max,3*jj+(ii*j_max)+5)
                        plt.imshow(predA_img,cmap='gray',interpolation='nearest',
                                  vmin=cmap_min_VA, vmax=cmap_max_VA)
                        plt.xticks(np.linspace(-0.5,9.5,11),range(10))
                        plt.yticks(np.linspace(-0.5,9.5,11),range(10))
                        plt.grid(color='w',lw=1,ls='-',alpha=0.5)

                        print(np.max(previous_img),np.max(target_img),softmax20(pred_img[5,5]))

                        dcl = sess.run(mainQN.disp_conv_list,\
                         feed_dict={mainQN.scalarInput:np.vstack(displayBatch[:,0]),
                                    mainQN.actions:displayBatch[:,1],
                                    mainQN.old_actions:displayBatch[:,5]})
                plt.show()

                if n_free > 0:
                    fk = sess.run(mainQN.free_kernels,feed_dict={mainQN.scalarInput:np.vstack(displayBatch[:,3])})

                    all_vars = tf.trainable_variables()
                    free_weights=[v for v in all_vars \
                     if v.name == 'mainQN/pg_free/weights:0'][0]\
                    .eval(session=sess)

                    i_max=2
                    j_max=mainQN.n_free_kernels
                    plt.figure(figsize=(j_max, i_max))
                    for ii in range(i_max):
                        target_img = (target_pool[0,:,:,ii])
                        vis_center = np.unravel_index(np.argmax(target_img), target_img.shape)
                        vis_center = (max(vis_center[0],5), max(vis_center[1],5))
                        for jj in range(j_max):
                            plt.subplot(i_max,j_max,jj+(ii*j_max)+1)
                            plt.xticks([])
                            plt.yticks([])
                            fw = free_weights[:,:,(ii*3):((ii+1)*3),jj]
                            fw = fw - np.min(fw)
                            fw = fw / np.max(fw)
                            plt.imshow(fw, interpolation='nearest')
    #                         fk_img = fk[0,:,:,jj]

    #                         fk_img = fk_img[vis_center[0]-5:vis_center[0]+5,
    #                                            vis_center[1]-5:vis_center[1]+5]
    #                         plt.imshow(fk_img,cmap='gray',interpolation='nearest')
                    plt.show()

            avg_window = batch_size*300
            if frame_err_list.shape[0] > 3*avg_window:
                plt.figure()

                for m_id in range(frame_err_list.shape[1]):
                    sqMat = np.resize(frame_err_list[:,m_id],
                                      [frame_err_list.shape[0]//avg_window,
                                                       avg_window])
                    QsqAvgs = np.average(sqMat,1)
                    q95 = np.percentile(sqMat,95,1)
                    plt.plot(QsqAvgs[1:],label=('mean ' + str(m_id)))
                    plt.plot(q95[1:],label=('95% ' + str(m_id)))
                plt.show()

    saver.save(sess,path+'/model-'+str(i)+'.cptk')


# this needs work -- focusing on protoModelnetwork for now
def train_protoQnetwork(env, pt,
                        batch_size=32, update_freq=4, y=.75, startE=1,
                        endE=0.1, anneling_steps=1000000.,
                        num_episodes=10000, max_epLength=5000,
                        pre_train_steps=5000, load_model=False,
                        path="./dqn/piaget", h_size=24, reset_freq=5000,
                        reward_prop=False, learning_rate=0.0025,
                        adam_eps=1e-5):
    tf.reset_default_graph()
    mainQN = protoQnetwork(env, h_size, pt.mover_prototypes, \
                           pt.mover_disps, md_equiv_classes,
                           1, 'mainQN', True,
                          lr=0.01, eps=1e-4)
    targetQN = protoQnetwork(env, h_size, mover_prototypes, \
                           mover_disps, md_equiv_classes,
                             1, 'targetQN', True)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    trainables = tf.trainable_variables()

    #targetOps = updateTargetGraph(trainables,tau)
    targetOps = updateTargetGraph(trainables,1)

    myBuffer = experience_buffer()

    #Set the rate of random action decrease.
    e = startE
    stepDrop = (startE - endE)/anneling_steps

    #create lists to contain total rewards and steps per episode
    jList = []
    rList = []
    total_steps = 0
    n_resets = 0

    #statistics on Q
    Qpred_list = []
    Qtarget_list = []
    QrList = []

    #Make a path for our model to be saved in.
    if not os.path.exists(path):
        os.makedirs(path)

    sess = tf.Session()
    #with tf.Session() as sess:
    sess.run(init)
    if load_model == True:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess,ckpt.model_checkpoint_path)

    updateTarget(targetOps,sess) #Set the target network to be equal to the primary network.
    for i in range(num_episodes):
        episodeBuffer = experience_buffer()
        #Reset environment and get first new observation(s)
        s_list, r, d, info = initial_obs(env, breakout)

        s_stack = np.dstack(s_list)
        s = processState(s_stack)

        d = False
        rAll = 0
        j = 0
        # periodically evaluate
        if i % 5 == 0 and total_steps  > pre_train_steps:
            eval_episode = True
            saved_e = e
            e = 0.1
        else:
            eval_episode = False
        #The Q-Network
        while j < max_epLength: #If the agent takes longer than 200 moves to reach either of the blocks, end the trial.
            j+=1

            #plt.imshow(env.render('rgb_array'))
            #plt.show()

            #Choose an action by greedily (with e chance of random action) from the Q-network
            if np.random.rand(1) < e or total_steps < pre_train_steps:
                a = np.random.randint(0,env.action_space.n)
            else:
                a = sess.run(mainQN.predict,feed_dict={mainQN.scalarInput:[s]})[0]

            if breakout and np.array_equal(s_list[0],s_list[1]):
                # breakout: requesting next ball
                a = 1

            s_next,r,d,info = env.step(a)
            #s_next = cv2.cvtColor(s_next, cv2.COLOR_BGR2GRAY)

            s_list.pop(0)
            s_list.append(s_next)

            s_stack = np.dstack(s_list)
            s1 = processState(s_stack)

            total_steps += 1
            if (r > 0) or d:
                buffer_repeat = 1
            else:
                buffer_repeat = 1
            for dummy in range(buffer_repeat):
                episodeBuffer.add(np.reshape(np.array([s,a,r,s1,d]),[1,5]),reward_prop) #Save the experience to our episode buffer.

            if total_steps > pre_train_steps and not eval_episode:
                if e > endE:
                    e -= stepDrop

                if total_steps % (update_freq) == 0:
                    trainBatch = myBuffer.sample(batch_size) #Get a random batch of experiences.
                    #Below we perform the Double-DQN update to the target Q-values
                    Q1 = sess.run(mainQN.predict,feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,3])})
                    Q2 = sess.run(targetQN.Qout,feed_dict={targetQN.scalarInput:np.vstack(trainBatch[:,3])})
                    end_multiplier = -(trainBatch[:,4] - 1)
                    doubleQ = Q2[range(batch_size),Q1]
                    targetQ = trainBatch[:,2] + (y*doubleQ * end_multiplier)
                    Qtarget_list.extend(targetQ)
                    predQ = sess.run(mainQN.Q, \
                        feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,0]), mainQN.actions:trainBatch[:,1]})
                    Qpred_list.extend(predQ)
                    QrList.extend(trainBatch[:,2])
                    #Update the network with our target values.
                    _ = sess.run(mainQN.updateModel, \
                        feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,0]),mainQN.targetQ:targetQ, mainQN.actions:trainBatch[:,1]})
                    if (total_steps//update_freq) % reset_freq == 0:
                        n_resets += 1
                        updateTarget(targetOps,sess) #Set the target network to be equal to the primary network.
    #                 print('value')
    #                 print(np.sum([sess.run(mainQN.streamV,\
    #                  feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:1,0]), mainQN.actions:trainBatch[:1,1]})>0]))
    #                 print(np.max(sess.run(mainQN.streamV,\
    #                  feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:1,0]), mainQN.actions:trainBatch[:1,1]})))
    #                 print('action')
    #                 print(np.sum([sess.run(mainQN.streamA,\
    #                  feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:1,0]), mainQN.actions:trainBatch[:1,1]})>0]))
    #                 print(np.max(sess.run(mainQN.streamA,\
    #                  feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:1,0]), mainQN.actions:trainBatch[:1,1]})))

            rAll += r
            s = s1

            if d == True:
                break
        #print(j)
        if not eval_episode:
            myBuffer.add(episodeBuffer.buffer,False)
            jList.append(j)
            rList.append(rAll)
        else:
            print("Evaluated model")
            print(j, rAll)
            e = saved_e
        #Periodically save the model.
        if i % 100 == 0 and i > 0:
            saver.save(sess,path+'/model-'+str(i)+'.cptk')
            print("Saved Model")
        print_rate = 5
        if len(rList) % print_rate == 0 and total_steps > pre_train_steps:
            displayBatch = myBuffer.sample(1)

            print(total_steps,
                  np.mean(jList[-print_rate:]),
                  np.mean(rList[-print_rate:]),
                  e)
            all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

            test_weights=[v for v in all_vars \
             if v.name == 'Conv/weights:0'][0]\
            .eval(session=sess)

            q_weights=[v for v in all_vars \
             if v.name == 'Conv_1/weights:0'][0]\
            .eval(session=sess)

    #         q_weights=[v for v in all_vars \
    #          if v.name == 'Conv/weights:0'][0]\
    #         .eval(session=sess)

            for ch_ind in range(1) + range(h_size//2,h_size//2+3):
                i_max=1
                j_max=8
                plt.figure(figsize=(j_max, i_max))
                for ii in range(i_max):
                    for jj in range(j_max):
                        cmap_max = max(np.max(q_weights[:,:,:,ch_ind]),
                                      -np.min(q_weights[:,:,:,ch_ind]))
                        plt.subplot(i_max,j_max,jj+(ii*j_max)+1)
                        plt.xticks([])
                        plt.yticks([])
                        plt.imshow(q_weights[:,:,jj+(ii*j_max),ch_ind],cmap='gray',
                        interpolation='nearest',
                        vmin=-cmap_max,vmax=cmap_max)
                plt.show()

            for ch_ind in range(8):

                i_max=1
                j_max=8
                plt.figure(figsize=((4./6)*j_max, (4./8)*i_max))
                for ii in range(i_max):
                    for jj in range(j_max):
                        cmap_max = max(np.max(test_weights[:,:,:,ch_ind]),
                                      -np.min(test_weights[:,:,:,ch_ind]))
                        plt.subplot(i_max,j_max,jj+(ii*j_max)+1)
                        plt.xticks([])
                        plt.yticks([])
                        plt.imshow(test_weights[:,:,jj+(ii*j_max),ch_ind],cmap='gray',
                        interpolation='nearest',
                                  vmin=-cmap_max,vmax=cmap_max)

                plt.show()

            displayBatch = myBuffer.sample(100)
            print((1./100)*np.sum([sess.run(mainQN.streamV,\
                feed_dict={mainQN.scalarInput:np.vstack(displayBatch[:,0]), mainQN.actions:displayBatch[:,1]})>0]))
            print((1./100)*np.sum([sess.run(mainQN.streamA,\
                feed_dict={mainQN.scalarInput:np.vstack(displayBatch[:,0]), mainQN.actions:displayBatch[:,1]})>0]))
            print('n_resets', n_resets)
    saver.save(sess,path+'/model-'+str(i)+'.cptk')
