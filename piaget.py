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
import re

import tensorflow.python.platform
from tensorflow.python.platform import gfile

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import trackpy as tp
import pandas as pd

from sklearn.linear_model import LogisticRegression
# classes

class Point():
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __repr__(self):
        return str((self.x, self.y))
    def __add__(self, p):
        return Point(self.x + p.x, self.y + p.y)
    def __sub__(self, p):
        return Point(self.x - p.x, self.y - p.y)

    def sq_dist(self, p):
        return (self.x-p.x)**2 + (self.y-p.y)**2

    def sq_norm(self):
        return (self.x)**2 + (self.y)**2

class Mover():
    def __init__(self, mover_id, game_id, img_dir, trajectory=[]):
        self.trajectory = trajectory
        self.displacements = []
        self.id = mover_id
        self.img_dir = img_dir + 'mover' + str(self.id) + '/'
        os.mkdir(self.img_dir)
        self.features = None
        self.visible = True

    def update(self, latest_location, box):
        self.trajectory.append(latest_location)
        cur_frame = int(self.trajectory[-1][0])
        img_for_save = Image.fromarray(box.img)
        img_for_save.save(self.img_dir + 'frame' + str(cur_frame) + '.jpg')

class MoverTracker():
    def __init__(self, game_id, img_dir, hyperparams, debug=False):

        self.game_id = game_id
        self.img_dir = img_dir

        self.stage = 'init_movement_tracking'

        self.n_movers = 0
        self.movers = []
        self.frame_pairs = []
        self.action_hist = []
        self.reward_hist = []

        self.tp_link = pd.DataFrame()

        self.hyperparams = hyperparams

    def process_frame_pair(self, frame_pair):
        frame_pair.find_movers(force_square=self.hyperparams['force_square'])

        self.action_hist.append(frame_pair.a)
        self.reward_hist.append(frame_pair.r)
        self.frame_pairs.append(frame_pair)
        self.cur_frame = len(self.frame_pairs)-1

        if self.stage == 'init_movement_tracking':
            tp_features = tp.locate(frame_pair.thresh, self.hyperparams['tp_part_size'])
            tp_features['frame'] = self.cur_frame
            self.tp_link = self.tp_link.append(tp_features)
            self.tp_link = tp.link_df(self.tp_link,
                            search_range=self.hyperparams['tp_max_disp'],
                            memory=self.hyperparams['tp_memory'])

            tp_link_cur_frame = self.tp_link[self.tp_link['frame']==
                                             self.cur_frame]
            tp_particles_ints = tp_link_cur_frame.loc[:,'particle'].astype('int')

            for i in tp_particles_ints.unique():
                particle_rows = tp_link_cur_frame[tp_particles_ints==i]
                latest_location = [[obs[1].frame, Point(obs[1].x, obs[1].y)]
                              for obs in particle_rows.iterrows()][0]
                if i > self.n_movers-1:
                    m = Mover(mover_id=i, game_id=self.game_id,
                              img_dir=self.img_dir,
                              trajectory=[])
                    self.movers.append(m)
                    self.n_movers += 1
                box_id = self.identify_particle(latest_location[1],
                                                frame_pair)
                box = frame_pair.boxes[box_id]
                self.movers[i].update(latest_location, box)

    def identify_particle(self, center, frame_pair):
        choices = [b.center for b in frame_pair.boxes]
        dists = [center.sq_dist(p) for p in choices]
        ind_min = np.argmin(dists)
        identification = ind_min
        return identification

class FramePair():
    def __init__(self, s0, s1, a, r):
        self.s0 = s0
        self.s1 = s1
        self.a = a
        self.r = r
        self.boxes = []
        self.thresh = None

    def find_movers(self, force_square=True):
        boxes = []
        frame_diff = self.s1 - self.s0
        fd_grey = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(fd_grey,1,255,cv2.THRESH_BINARY)[1]
        thresh_dilated = cv2.dilate(thresh,None,iterations=1)
        (cnts, _) = cv2.findContours(thresh_dilated.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        for i, c in enumerate(cnts):
            if cv2.contourArea(c) > 0:
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
                    lower_left = Point(x,y)
                    upper_right = Point(x+w, y+h)
                    box = Box(lower_left, upper_right)
                    box.add_image(self.s0)
                    boxes.append(box)
        self.boxes = boxes
        self.thresh = thresh
        return boxes, thresh

class Box():
    def __init__(self, lower_left, upper_right):
        self.ll = lower_left
        self.ur = upper_right
        self.width = (self.ur-self.ll).x
        self.height = (self.ur-self.ll).y
        self.center = Point((self.ll.x + self.ur.x)/2, (self.ll.y + self.ur.y)/2)

    def __repr__(self):
        return '(' + str(self.ll.x) + '-' + str(self.ur.x) + ', ' + \
        str(self.ll.y) + '-' + str(self.ur.y) + ')'

    def add_image(self, s):
        self.img = s[self.ll.y:self.ur.y, self.ll.x:self.ur.x]

class Categorizer():
    def __init__(self, game_id, img_dir, hyperparams):
        self.game_id = game_id
        self.img_dir = img_dir + str(game_id) + '/'
        self.hyperparams = hyperparams

    def categorize_movers(self):
        mover_dirs = [self.img_dir + d + '/'
                      for d in os.listdir(self.img_dir) if d.find('mover') == 0]

        self.features, self.labels = self.get_inception_features(mover_dirs)

        self.LR = LogisticRegression()
        self.LR.fit(self.features, self.labels)



    def get_inception_features(self, class_dirs):
        nb_features = 2048
        features = []
        labels = []

        self.create_inception_graph()

        with tf.Session() as sess:

            next_to_last_tensor = sess.graph.get_tensor_by_name('pool_3:0')
            ind = 0

            for class_dir in class_dirs:
                class_label = int(class_dir[class_dir.find('mover')+5:-1])
                list_images = [class_dir+f for f in os.listdir(class_dir) if re.search('jpg|JPG', f)]
                for image in list_images:
                    if (ind%10 == 0):
                        print('Processing %s...' % (image))
                    if not gfile.Exists(image):
                        tf.logging.fatal('File does not exist %s', image)

                    image_data = gfile.FastGFile(image, 'rb').read()
                    feature_vec = sess.run(next_to_last_tensor,
                                           {'DecodeJpeg/contents:0': image_data})
                    features.append(np.squeeze(feature_vec))
                    labels.append(class_label)
                    ind += 1
        return np.array(features), np.array(labels)

    def create_inception_graph(self):
      """Creates a graph from saved GraphDef file and returns a saver."""
      # Creates graph from saved graph_def.pb.
      with tf.gfile.FastGFile(os.path.join(
          './', 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


# functions: gym

def init_env(env,n_steps):
    env.reset()
    # e.g. nothing happens in first 100 steps of ms pacman
    for i in range(n_steps):
        s,r,d,info = env.step(env.action_space.sample()) # take a random action
    return s,d

# functions: running the AI

def play(num_steps, env, img_dir, init_steps, hyperparams={'force_square': True,
            'tp_part_size': 11,
            'tp_max_disp': 10,
            'tp_memory': 3},
         debug=False):

    game_id = str(random.randint(0,1e6))

    img_dir = img_dir + str(game_id) + '/'
    os.mkdir(img_dir)

    mover_tracker = MoverTracker(game_id, img_dir, hyperparams)

    s0, done = init_env(env, init_steps)

    for i in range(num_steps):

        if done:
            s0, done = init_env(env, init_steps)

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
