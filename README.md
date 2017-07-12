# piaget
Visual reinforcement learner with "developmental stages"

Note: This is a personal project which is currently in the "trying lots of experiments to find out what works" stage.  The experimentation takes place in the Juypter Notebook files, while experimental code that works well enought to keep and build upon goes in the Python files.  This README doesn't intend to make everything in the code comprehensible (since I am continually changing it), but tries to explain the general idea and the components which are relatively well established (as of 6/18/17).

##  Motivation

### Data Efficiency

There are at least two distinct metrics one can use to evaluate an algorithm that "learns" over time: asymptotic performance and data efficiency.  The former asks how well the computer can perform after a fixed amount of *computing* time, while the latter asks how well the computer can perform after a fixed length of experience with the task.

An interesting property of deep reinforcement learning algorithms (as pioneered by DeepMind) is that they can frequently match or surpass humans in asymptotic performence, but tend to be far, far less data-efficient than humans.  For instance, in [DeepMind's 2014 paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) on deep Q-learning for Atari games, the algorithm could (using a single set of hyperparameters) ultimately play many games as well as a human or better -- but this asymptotic high performance was the result of 50 million frames of interaction per game, the equivalent of playing a each game for about 38 days straight.  By contrast, the human testers used to establish "human-level" game performance were allowed 2 hours of practice per game.  (N.B.: a [DeepMind paper from March 2017](https://arxiv.org/abs/1703.01988) proposes a fascinating new algorithm, Neural Episodic Control, motivated in part by these data efficiency concerns.)

A natural objection to the above comparison -- 38 days for the algorithm, 2 hours for the humans -- is that the human testers arrive with a lifetime of relevant experience, in tasks involving vision, moving objects, real-world physics, etc.  Meanwhile,  the algorithm is a general learner, not allowed to "smuggle in" such domain-specific information.  However, this latter claim is only partially true.  The ConvNet architecture, a crucial piece of the algorithms under discussion, is domain-specific: it is designed to exploit data sets which have been sampled from known points in a space with a known metric (notion of distance), such as pixels in a 2D image, samples in a recorded sound wave, or spaces on a game board.  We would not expect ConvNets to perform well on data which has nothing like this structure (for instance, on images whose pixels have been rearranged in some deterministic-but-unknown way).

Moreover, the deep ConvNet architecture has close analogies to our best models of the mammalian visual cortex, which contains sequential layers of topographicaly organized cells with local connectivity.

So there are reasons to believe that when deep reinforcement learners do well as humans, it is because they are using domain-specific information, and using it in some of the ways we do.

### Development

Humans are not built with all our domain-specific knowledge fully in place.  We metaphorically -- and literally! -- crawl before we can walk.  This is true even for very basic and dependable regularities in our environments.

For instance, cells in our primary visual cortex (relatively early in visual processing) are receptive to lines in specific orientations, much like the filters one might expect to see in an early layer of a visual ConvNet (cf. [Gabor filters](https://en.wikipedia.org/wiki/Gabor_filter)).  But these filters are not precisely pre-programmed by the genome.  Instead, the genome "initializes" a network which will develop the correct filters when "trained" in a natural environment in early life.  In several famous experiments (e.g. [this paper](https://pdfs.semanticscholar.org/5267/a2049b9c11d0deca2eb4a670e6f4aaa58dce.pdf)) , cats were raised in artificial environments containing only lines in one specific orientation, and they did not develop the usual full repertoire of orientation-selective cells.

In other words, just like deep neural nets, mammalian brains have to be "trained" for a while before they can get anywhere useful.  But once we *do* get something figured out, we tend *keep* it.  Cats and people do spend some amount of time, in childhood, wiring up the filters of their visual system, but after that, we never need to figure out how to see again.  When we sit down to play an Atari game for the first time, we can depend on the orientation-selective filters we developed as babies, and stuck with every since.  By contrast, standard artificial neural nets reinvent everything from scratch every time they are trained.  They re-learn to see every time we train them on a different game, or every time we vary a hyperparameter.

So while humans do have domain specific prior knowledge, we acquire that knowledge in a process of *learning* rather than mere hard-coding.  But we seem to do it in a sequential, planned manner -- building lower-level models, fixing them in place, and then building higher-level models on top of them, searching at each stage for the information most relevant to the type of model we are then constructing.

### Atari

Let's be a lot more concrete.  If I sit down to play an Atari game I've never seen before, I have a lot of expectations.  Not just that I'll be interacting with a two-dimensional spatial world (a bit of domain knowledge captured in the ConvNet architecture), but that motion is important, that there are persistent *objects* which move, that these objects are probably more important than static parts of the background, that it is probably important when these collide.

If you start with the not-especially-strong, physically motivated assumption that *moving things are persistent and important*, you can immediately extract a lot of information from the first few frames of an Atari game.  Take the difference between frames and identify the (typically small) regions where something has changed.  These usually correspond to moving objects (sprites).  Track which regions have similar positions as time elapses, take snapshots of those regions and apply some basic computer vision to them, and you've got reliable images of the most significant moving objects in the game, plus a sample trajectory for each one -- within as few as 5, 10, or 15 frames.  Use the snapshots to initialize the first layer of a ConvNet, and the observed motions to initialize the second layer, and after 15 frames you've got a set of filters that pinpoint the most important things in the game.  All of this assumes a fair amount about the visual/spatial structure of the environment -- but then, so do ConvNets alone.

***

Work in progress.  Designed for visual environments from the [OpenAI Gym](http://gym.openai.com/), like Atari games.  The central idea is to write a learner that "crawls before it can walk," e.g. figures out some basic things about the visual environment (what types of moving things are there, which can it control) first, then leverages that information to learn about the state and reward dynamics, etc.

Once a certain amount of information has been obtained, I'll probably use it to estimate values in something like Q-learning, and perhaps add a generic CNN that tries to correct for what this information misses.  I'm curious whether this approach will have some advantages over doing deep Q-learning from scrach, by leveraging some properties that first-time human players assume from background experience (visual states contain persistent objects, changes are composed of local object motions, the player can directly control some objects and not others).

Requires gym, TensorFlow, Pillow, and cv2.  Running piaget.py will (as of this writing, 5/9/17) play a bit of Ms. Pacman and dump images of the moving objects it identified.  In inception_tests.ipynb, I've done some tests with classifying these images via a linear classifier on Inception v3 features; I'm planning to have the model use this information to discern types of moving objects and to make object tracking more stable.
