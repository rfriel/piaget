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
from trackpy.predict import NearestVelocityPredict
import pandas as pd

from sklearn.linear_model import LogisticRegression
from itertools import combinations

# classes

class Point():
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __repr__(self):
        return str((self.x, self.y))
    def is_none(self):
        return (self.x is None) or (self.y is None)
    def __add__(self, p):
        if self.is_none() or p.is_none():
            return Point(None, None)
        return Point(self.x + p.x, self.y + p.y)
    def __sub__(self, p):
        if self.is_none() or p.is_none():
            return Point(None, None)
        return Point(self.x - p.x, self.y - p.y)
    def __neg__(self):
        return Point(-self.x, -self.y)
    def __eq__(self, p):
        return (self.x == p.x) and (self.y == p.y)

    def sq_dist(self, p):
        if self.is_none() or p.is_none():
            return Point(None, None)
        return (self.x-p.x)**2 + (self.y-p.y)**2

    def sq_norm(self):
        if self.is_none():
            return Point(None, None)
        return (self.x)**2 + (self.y)**2

    def thresh(self, sign):
        if sign > 0:
            return Point(max(self.x,0),max(self.y,0))
        else:
            return Point(min(self.x,0),min(self.y,0))

class Mover():
    def __init__(self, mover_id, game_id, img_dir, trajectory=[]):
        self.trajectory = trajectory
        self.displacements = []
        self.id = mover_id
        self.img_dir = img_dir + 'mover' + str(self.id) + '/'
        os.mkdir(self.img_dir)
        self.features = None
        self.visible = True

    def update(self, box, cur_frame):
        self.trajectory.append([cur_frame, box])
        try:
            img_for_save = Image.fromarray(box.img)
        except ValueError:
            import pdb; pdb.set_trace()
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

        self.cur_frame = 0

        self.tp_link = pd.DataFrame()
        self.tp_pred = NearestVelocityPredict()

        self.hyperparams = hyperparams

    def process_frame_pair(self, frame_pair):

        self.action_hist.append(frame_pair.a)
        self.reward_hist.append(frame_pair.r)
        self.frame_pairs.append(frame_pair)

        if self.stage == 'init_movement_tracking':
            self.frame_pairs[-1].find_movers(force_square=self.hyperparams['force_square'])
            if self.cur_frame == 0:
                # initialize movers
                for i, box_pair in enumerate(frame_pair.mover_boxes):
                    m = Mover(mover_id=i, game_id=self.game_id,
                              img_dir=self.img_dir,
                              trajectory=[])
                    m.update(box_pair[0], 0)
                    m.update(box_pair[1], 1)
                    self.movers.append(m)
            else:
                self.identify_movers()
            #import pdb; pdb.set_trace()
            self.cur_frame = len(self.frame_pairs)+1
            self.n_movers = len(self.movers)


    def identify_movers(self):
        cur = self.frame_pairs[-1]
        for i, box_pair in enumerate(cur.mover_boxes):
            assigned = False
            for m in self.movers:
                if (m.trajectory[-1][1].center - box_pair[0].center).sq_norm() < 4:
                    m.update(box_pair[1], self.cur_frame)
                    assigned = True
            if not assigned:
                m = Mover(mover_id=self.n_movers+1,
                          game_id=self.game_id,
                          img_dir=self.img_dir,
                          trajectory=[])
                m.update(box_pair[1], self.cur_frame)
                self.movers.append(m)
                self.n_movers += 1

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

        self.finder = TranslationFinder(self, thresh_dilated, cnts)
        self.mover_boxes = self.finder.find_translations()

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
    def __sub__(self, b):
        return box(self.ll-b.ll, self.ur-bb.ur)
    def __eq__(self, b):
        return (self.ll == b.ll) and (self.ur == b.ur)

    def add_image(self, s):
        self.img = s[self.ll.y:self.ur.y, self.ll.x:self.ur.x]

class TranslationFinder():
    def __init__(self, frame_pair, thresh_dilated, cnts):
        self.frame_pair = frame_pair
        self.f0 = cv2.threshold(\
                                cv2.cvtColor(self.frame_pair.s0, \
                               cv2.COLOR_BGR2GRAY),
                                0,255,\
                                cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1].astype('float32')
        self.f1 = cv2.threshold(\
                                cv2.cvtColor(self.frame_pair.s1, \
                               cv2.COLOR_BGR2GRAY),
                                0,255,\
                                cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1].astype('float32')
        self.thresh_dilated = thresh_dilated
        self.cnts = cnts
        self.all_joins = self.make_join_combinations()
        self.cnts_joined = [self.join_cnts(inds) for inds in self.all_joins]

    def find_translations(self):
        plt.subplot(121)
        ax = plt.gca()
        plt.imshow(self.f0,cmap='gray')
        for cnt_id in range(len(self.cnts)):
            box = self.cnts_joined[cnt_id]
            ax.add_patch(patches.Rectangle((box.ll.x, box.ll.y),
                                    box.width, box.height,
                                    color='g',
                                    fill=False)
            )
            plt.text(box.ur.x+3, box.ur.y+3, str(cnt_id),color='w')
        plt.subplot(122)
        ax = plt.gca()
        for cnt_id in range(len(self.cnts)):
            box = self.cnts_joined[cnt_id]
            ax.add_patch(patches.Rectangle((box.ll.x, box.ll.y),
                                    box.width, box.height,
                                    color='g',
                                    fill=False)
            )
            plt.text(box.ur.x+3, box.ur.y+3, str(cnt_id),color='w')
        plt.imshow(self.f1,cmap='gray')
        plt.show()

        self.cnt_scores = {i: [] for i in range(len(self.cnts))}
        self.cnt_score_params = {i: [] for i in range(len(self.cnts))}
        for box_id, box in enumerate(self.cnts_joined):
            box0 = Box(box.ll, box.ur)
            box1 = Box(box.ll, box.ur)
            # grayscale
            box0.add_image(self.f0)
            box1.add_image(self.f1)

            box0_1channel = np.expand_dims(box0.img,2)
            box1_1channel = np.expand_dims(box1.img,2)
            #import pdb; pdb.set_trace()

            pc = cv2.phaseCorrelate(box0.img,box1.img)
            four_shifts = [[np.floor(pc[1]),np.floor(pc[0])],\
                           [np.ceil(pc[1]),np.floor(pc[0])],\
                           [np.floor(pc[1]),np.ceil(pc[0])],\
                           [np.ceil(pc[1]),np.ceil(pc[0])]
                          ]
            best_ratio = 1

            best_shift = four_shifts[0]
            for s in four_shifts:
                neg_s = [-s[0],-s[1]]
                synth_box1 = self.generate_translate(box0_1channel,box1_1channel,s)
                synth_box0 = self.generate_translate(box1_1channel,box0_1channel,neg_s)

                ref_score = self.score_gt(box0_1channel,box1_1channel,(0,0))
                ratio0 = self.score_gt(box0_1channel,box1_1channel,s)/ref_score
                ratio1 = self.score_gt(box1_1channel,box0_1channel,neg_s)/ref_score

                ratio_s = max(ratio0,ratio1)

                if ratio_s < best_ratio:
                    if True:#(len(self.all_joins[box_id]) == 1) or ratio_s < 0.1:
                        # note: make the 0.1s above a hyperparameter

                        # above statement applies more stringent condition to
                        # joins of more than one contour: we should be using
                        # those joins to *exactly* match extended objects.
                        # if we don't do something like this, then when a new
                        # object *appears* out of nowhere, we won't be able to
                        # get anywhere with it, so it will try to mitigate the
                        # failure by using a bigger box
                        best_shift = s
                        best_ratio = ratio_s

            s = best_shift
            neg_s = [-s[0],-s[1]]

            print 'Join ' + str(self.all_joins[box_id]) + ', Shift ' + str(s) + ', Score ' + str(best_ratio)

            synth_box1 = self.generate_translate(box0_1channel,box1_1channel,s)
            synth_box0 = self.generate_translate(box1_1channel,box0_1channel,neg_s)
            '''
            plt.subplot(121)
            plt.imshow(box0.img,cmap='gray')
            plt.subplot(122)
            plt.imshow(synth_box0.squeeze(),cmap='gray')
            plt.show()

            plt.subplot(121)
            plt.imshow(box1.img,cmap='gray')
            plt.subplot(122)
            plt.imshow(synth_box1.squeeze(),cmap='gray')
            plt.show()
            '''
            #if (len(self.cnts) > 2) and (len(self.all_joins[box_id]) > 1):
                #import pdb; pdb.set_trace()

            for ind in self.all_joins[box_id]:
                self.cnt_scores[ind].append(best_ratio)
                self.cnt_score_params[ind].append([self.all_joins[box_id], s])
        self.cnt_best_params = {}
        self.cnt_best_scores = {}
        cnt_scores_remaining = self.cnt_scores.copy()
        cnt_score_params_remaining = self.cnt_score_params.copy()

        print '\nStarting . . . \n\n'
        while len(cnt_scores_remaining) > 0:
            print 'self.cnt_best_params: ' + str(self.cnt_best_params)
            print 'len(self.cnts): ' + str(len(self.cnts))
            best_scores_cnts = {k: min(v) \
                                for k, v in cnt_scores_remaining.iteritems()}
            winning_index = np.argmin(best_scores_cnts.values())
            winner = best_scores_cnts.keys()[winning_index]
            winner_best_index = np.argmin(cnt_scores_remaining[winner])
            self.cnt_best_params[winner] = cnt_score_params_remaining[winner][winner_best_index]
            self.cnt_best_scores[winner] = cnt_scores_remaining[winner][winner_best_index]

            forbidden_inds = set(self.cnt_best_params[winner][0])
            print 'winner: ' + str(winner)
            print 'winner params: ' + str(self.cnt_best_params[winner])
            print 'forbidden_inds: ' + str(forbidden_inds)
            print 'cnt_scores_remaining :' + str(cnt_scores_remaining) + '\n'
            print 'cnt_score_params_remaining :' + str(cnt_score_params_remaining) + '\n'
            revised_scores = {k: [] for k in cnt_scores_remaining}
            revised_params = {k: [] for k in cnt_score_params_remaining}
            #import pdb; pdb.set_trace()
            for k in cnt_score_params_remaining:
                params_k = cnt_score_params_remaining[k]
                for j, par in enumerate(params_k):
                    join_k_j = par[0]
                    print 'Examining join ' + str(join_k_j) + ', for box ' + str(k)
                    if len(forbidden_inds & set(join_k_j)) > 0:
                        #del cnt_scores_remaining[k][j]
                        #del cnt_score_params_remaining[k][j]
                        print 'Deleted join ' + str(join_k_j) + ', for box ' + str(k)
                        print 'j: ' + str(j)
                        print 'len(params_k): ' + str(len(params_k))
                    else:
                        revised_scores[k].append(cnt_scores_remaining[k][j])
                        revised_params[k].append(cnt_score_params_remaining[k][j])
            cnt_scores_remaining = revised_scores.copy()
            cnt_score_params_remaining = revised_params.copy()
            for ind in forbidden_inds:
                cnt_scores_remaining.pop(ind)
                cnt_score_params_remaining.pop(ind)
            print 'cnt_scores_remaining :' + str(cnt_scores_remaining) + '\n'
            print 'cnt_score_params_remaining :' +str(cnt_score_params_remaining) + '\n'
        self.mover_joins = set()
        self.mover_shifts = {}
        self.mover_scores = {}
        for k, c in self.cnt_best_params.iteritems():
            self.mover_joins.add(c[0])
            self.mover_shifts[c[0]] = Point(int(c[1][1]),int(c[1][0]))
            self.mover_scores[c[0]] = self.cnt_best_scores[k]
        self.valid_boxes = {inds: self.join_cnts(inds) for inds in self.mover_joins}

        self.mover_boxes = []
        for ind in self.valid_boxes:
            box = self.valid_boxes[ind]
            shift = self.mover_shifts[ind]
            n_shift = -shift
            if self.mover_scores[ind] < 0.9:
                # note: make the 0.9 above a hyperparameter
                # the below crops the boxes if we're reasonably sure we've
                # identified an actual displacement
                shift = Point(shift.x % box.height, shift.y % box.width)
                n_shift = -shift
                box0 = Box(box.ll-shift.thresh(-1), box.ur-shift.thresh(1))
                box1 = Box(box.ll-n_shift.thresh(-1), box.ur-n_shift.thresh(1))
            else:
                box0 = Box(box.ll, box.ur)
                box1 = Box(box.ll, box.ur)
            box0.add_image(self.frame_pair.s0)
            box1.add_image(self.frame_pair.s1)
            self.mover_boxes.append([box0, box1])
        return self.mover_boxes

    def make_join_combinations(self):
        return [c \
                for l in range(1,len(self.cnts)+1)\
                for c in combinations(range(len(self.cnts)),l)\
                ]

    def join_cnts(self, inds):
        (x, y, w, h) = cv2.boundingRect(self.cnts[inds[0]])
        ll = Point(x,y)
        ur = Point(x+w,y+h)
        for ind in inds[1:]:
            (x, y, w, h) = cv2.boundingRect(self.cnts[ind])
            if x < ll.x:
                ll.x = x
            if x > ur.x:
                ur.x = x
            if y < ll.y:
                ll.y = y
            if y > ur.y:
                ur.y = y
        return Box(ll,ur)

    def generate_translate(self, f0, f1, shift, debug=False):
        #out = np.zeros(f0.shape)
        n_changed = 0
        s_x = int(shift[0])
        s_y = int(shift[1])
        '''
        pad_x = (max(s_x,0), max(-s_x,0))
        pad_y = (max(s_y,0), max(-s_y,0))

        f0_padded = np.pad(f0, (pad_x, pad_y, (0,0)), 'constant')

        x_start = max(-s_x,0)
        y_start = max(-s_y,0)

        out = f0_padded[x_start:(x_start + f0.shape[0]),\
                        y_start:(y_start + f0.shape[1]),\
                        :]
        '''

        out = np.roll(f0,(s_x,s_y),(0,1))

        '''
        for i in range(shifts.shape[0]):
            for j in range(shifts.shape[1]):
                try:
                    x_out_float = (i+shifts[i,j,0])# % out.shape[0]
                    y_out_float = (j+shifts[i,j,1])# % out.shape[1]

                    x_out = np.round(x_out_float).astype(int) % out.shape[0]
                    y_out = np.round(y_out_float).astype(int) % out.shape[1]

                    out[x_out,y_out,:] = f0[i,j,:]
                    n_changed += 1
                except IndexError:
                    pass
        '''
        if debug:
            return out, n_changed
        else:
            return out

    def score_gt(self, f0, f1, shift):
        err = f1 - self.generate_translate(f0, f1, shift)
        return np.sqrt(sum(err.flatten()**2))



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

def play(num_steps, env, img_dir, init_steps, \
         random_seed = None,\
         hyperparams={'force_square': False,
            'tp_part_size': 11,
            'tp_max_disp': 21,
            'tp_memory': 10},
         debug=False):

    game_id = str(random.randint(0,1e6))

    img_dir = img_dir + str(game_id) + '/'
    os.mkdir(img_dir)

    mover_tracker = MoverTracker(game_id, img_dir, hyperparams)

    s0, done = init_env(env, init_steps)
    if random_seed is not None:
        np.random.seed(seed=random_seed)
    for i in range(num_steps):

        if done:
            s0, done = init_env(env, init_steps)

        if i%1 == 0:
            # new action
            a = np.random.randint(env.action_space.n)#env.action_space.sample()
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
