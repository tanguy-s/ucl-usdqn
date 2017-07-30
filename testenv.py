import gym
import numpy as np

from usdqn_tf.envs.envs import Continuous_UsdqnOneDoFEnv

env = Continuous_UsdqnOneDoFEnv(True)
env.reset()

print(env.action_space.sample())

for _ in range(1000):
    print(env.action_space.sample())
    env.render()
    env.step(env.action_space.sample()) # take a random action




