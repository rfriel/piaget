# piaget
Visual reinforcement learner with "developmental stages"

Note: This is a personal project which is currently in the "trying lots of experiments to find out what works" stage.  The experimentation takes place in the Juypter Notebook files, while experimental code that works well enought to keep and build upon goes in the Python files.  This README doesn't intend to make everything in the code comprehensible (since I am continually changing it), but tries to explain the general idea and the components which are relatively well established (as of 6/18/17).

##  Motivation

There are at least two distinct metrics one can use to evaluate an algorithm that "learns" over time: asymptotic performance and data efficiency.  The former asks how well the computer can perform after a fixed amount of *computing* time, while the latter asks how well the computer can perform after a fixed length of experience with the task.

An interesting property of deep reinforcement learning algorithms (as pioneered by DeepMind) is that they can frequently match or surpass humans in asymptotic performence, but tend to be far, far more data-inefficient than humans.  For instance, in [DeepMind's 2014 paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) on deep Q-learning for Atari games, the algorithm could (using a single set of hyperparameters) ultimately play many games as well as a human or better -- but this asymptotic high performance was the result of 50 million frames of interaction per game, the equivalent of playing a each game for about 38 days straight.  By contrast, the human testers used to establish "human-level" game performance were allowed 2 hours of practice per game.

A natural objection to this comparison is that the human testers arrive with a lifetime of relevant experience -- in tasks involving vision, moving objects, real-world physics, etc. -- while the algorithm is a general learner, not allowed to "smuggle in" such domain-specific information.  However, this is only partially true.  The ConvNet architecture, a crucial piece of the algorithms under discussion, is domain-specific: it is designed to exploit data sets which have been sampled from known points in a space with a known metric (notion of distance), such as pixels in a 2D image, samples in a recorded sound wave, or spaces on a game board.  We would not expect ConvNets to perform well on data which has nothing like this structure (for instance, on images whose pixels have been rearranged in some deterministic-but-unknown way).

Moreover, the deep ConvNet architecture has close analogies to our best models of the mammalian visual system, an unusually well-understood part of the brain.  (Compared to the rest of the brain, early sensory processing is both unusually easy to probe experimentally and unsually feed-forward.)  And as Sarah Constantin [notes](https://srconstantin.wordpress.com/2017/01/28/performance-trends-in-ai/), deep learning has led to a discontinuous increase in performance on *visual and auditory* tasks, but not on tasks in *all* domains.  It's tempting 

***

Work in progress.  Designed for visual environments from the [OpenAI Gym](http://gym.openai.com/), like Atari games.  The central idea is to write a learner that "crawls before it can walk," e.g. figures out some basic things about the visual environment (what types of moving things are there, which can it control) first, then leverages that information to learn about the state and reward dynamics, etc.

Once a certain amount of information has been obtained, I'll probably use it to estimate values in something like Q-learning, and perhaps add a generic CNN that tries to correct for what this information misses.  I'm curious whether this approach will have some advantages over doing deep Q-learning from scrach, by leveraging some properties that first-time human players assume from background experience (visual states contain persistent objects, changes are composed of local object motions, the player can directly control some objects and not others).

Requires gym, TensorFlow, Pillow, and cv2.  Running piaget.py will (as of this writing, 5/9/17) play a bit of Ms. Pacman and dump images of the moving objects it identified.  In inception_tests.ipynb, I've done some tests with classifying these images via a linear classifier on Inception v3 features; I'm planning to have the model use this information to discern types of moving objects and to make object tracking more stable.
