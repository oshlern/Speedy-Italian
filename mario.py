# PYTHON 3 (sadly)
import numpy
import gym
import gym_pull
gym_pull.pull('github.com/ppaquette/gym-super-mario')        # Only required once, envs will be loaded with import gym_pull afterwards
env = gym.make('ppaquette/SuperMarioBros-1-1-v0')
