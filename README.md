# piaget
Visual reinforcement learner with "developmental stages"

Work in progress.  Designed for visual environments from the [OpenAI Gym](http://gym.openai.com/), like Atari games.  The central idea is to write a learner that "crawls before it can walk," e.g. figures out some basic things about the visual environment (what types of moving things are there, which can it control) first, then leverages that information to learn about the state and reward dynamics, etc.

Once a certain amount of information has been obtained, I'll probably use it to estimate values in something like Q-learning, and perhaps add a generic CNN that tries to correct for what this information misses.  I'm curious whether this approach will have some advantages over doing deep Q-learning from scrach, by leveraging some properties that first-time human players assume from background experience (visual states contain persistent objects, changes are composed of local object motions, the player can directly control some objects and not others).

Requires gym, TensorFlow, Pillow, and cv2.  Running piaget.py will (as of this writing, 5/9/17) play a bit of Ms. Pacman and dump images of the moving objects it identified.  In inception_tests.ipynb, I've done some tests with classifying these images via a linear classifier on Inception v3 features; I'm planning to have the model use this information to discern types of moving objects and to make object tracking more stable.
