from __future__ import division

import gym
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.misc
import os

import cv2

import pdb
from time import sleep

import tensorflow.python.platform
from tensorflow.python.platform import gfile

from PIL import Image
import re

# classes

class Point():
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __repr__(self):
        return str((self.x, self.y))

    def sq_dist(self, p):
        return (self.x-p.x)**2 + (self.y-p.y)**2

class Mover():
    def __init__(self, mover_id, game_id, img_dir):
        self.trajectory = []
        self.id = mover_id
        self.img_dir = img_dir + 'mover' + str(self.id) + '/'
        os.mkdir(self.img_dir)
        self.features = None

    def add_observation(self, location, img):
        self.trajectory.append(location)
        img_for_save = Image.fromarray(img)
        img_for_save.save(self.img_dir + str(random.randint(0,1e6)) + '.jpg')

class MoverTracker():
    def __init__(self, game_id, img_dir, hyperparams):
        self.game_id = game_id
        self.n_movers = 0
        self.movers = []
        self.action_hist = []
        self.reward_hist = []
        self.img_dir = img_dir + str(game_id) + '/'
        os.mkdir(self.img_dir)
        self.hyperparams = hyperparams

    def process_frame_pair(self, frame_pair):
        self.action_hist.append(frame_pair.a)
        self.reward_hist.append(frame_pair.r)

        boxes = frame_pair.find_movers(force_square=self.hyperparams['force_square'])[0]
        if self.n_movers == 0:
            if len(boxes) > 1:
                # if we only found one mover, hard to track which is which
                # so wait until we see two
                for i, box in enumerate(boxes):
                    m = Mover(mover_id=i, game_id=self.game_id, img_dir=self.img_dir)
                    img = frame_pair.s0[box[0].y:box[1].y, box[0].x:box[1].x]
                    center = Point((box[0].x + box[1].x)/2, (box[0].y + box[1].y)/2)

                    m.add_observation(center, img)
                    self.movers.append(m)
                    self.n_movers += 1
        else:
            cur_positions = [m.trajectory[-1] for m in self.movers]
            internal_dists = self.get_internal_dists(cur_positions)
            for i, box in enumerate(boxes):
                # this should be the other way around --
                # rather than matching new points to old points,
                # we should try to account for all the old points,
                # then deal with any new points remaining
                img = frame_pair.s0[box[0].y:box[1].y, box[0].x:box[1].x]
                center = Point((box[0].x + box[1].x)/2, (box[0].y + box[1].y)/2)

                box_id = self.identify_mover(center, cur_positions, internal_dists)
                if box_id >= self.n_movers:
                    # new mover
                    m = Mover(mover_id=box_id, game_id=self.game_id, img_dir=self.img_dir)
                    m.add_observation(center, img)
                    self.movers.append(m)
                    self.n_movers += 1
                else:
                    self.movers[box_id].add_observation(center, img)


    def identify_mover(self, center, cur_positions, internal_dists):
        dists = [center.sq_dist(p) for p in cur_positions]
        ind_min = np.argmin(dists)
        if dists[ind_min] > np.percentile(internal_dists,0.33):
            # above test is crude, refine later
            # new mover
            return self.n_movers
        else:
            return ind_min

    def get_internal_dists(self, cur_positions):
        internal_dists = []
        for i, p1 in enumerate(cur_positions):
            for j, p2 in enumerate(cur_positions[:i]):
                internal_dists.append(p1.sq_dist(p2))
        return internal_dists

class FramePair():
    def __init__(self, s0, s1, a, r):
        self.s0 = s0
        self.s1 = s1
        self.a = a
        self.r = r

    def find_movers(self, force_square=True):
        movers = []
        frame_diff = self.s1 - self.s0
        fd_grey = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(fd_grey,1,255,cv2.THRESH_BINARY)[1]
        thresh_dilated = cv2.dilate(thresh,None,iterations=1)
        (cnts, _) = cv2.findContours(thresh_dilated.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        for i, c in enumerate(cnts):
            if cv2.contourArea(c) > 41:
                (x, y, w, h) = cv2.boundingRect(c)
                if force_square:
                    if w < h:
                        deltaW = h - w
                        w = h
                        x = x - int(deltaW/2)
                    elif h < w:
                        deltaH = w - h
                        h = w
                        y = y - int(deltaH/2)
                if (x >= 0) and (y >= 0):
                    point0 = Point(x,y)
                    point1 = Point(x+w, y+h)
                    movers.append((point0, point1))
        return movers, thresh_dilated


# functions: gym

def init_env(env,n_steps):
    env.reset()
    # e.g. nothing happens in first 100 steps of ms pacman
    for i in range(n_steps):
        s,r,d,info = env.step(env.action_space.sample()) # take a random action
    return s,info

# functions: running the AI

def play(num_steps, env, img_dir, init_steps, debug=False, hyperparams={'force_square':True}):

    game_id = str(random.randint(0,1e6))

    mover_tracker = MoverTracker(game_id, img_dir, hyperparams)

    s0, info = init_env(env, init_steps)

    for i in range(num_steps):

        if info['ale.lives'] == 0:
            s0, info = init_env(env, init_steps)

        a = env.action_space.sample()
        s1,r,d,info = env.step(a)

        #s0 = s0[:160,:]
        #s1 = s1[:160,:]

        frame_pair = FramePair(s0, s1, a, r)

        mover_tracker.process_frame_pair(frame_pair)

        s0 = s1

    return mover_tracker

# main

if __name__ == '__main__':
    env = gym.make('MsPacman-v0')
    num_steps = 100
    img_dir = 'img/'
    mt = play(num_steps, env, img_dir, 10)
    print 'Game ID (for image directory): ' + str(mt.game_id)
