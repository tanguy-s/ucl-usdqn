import os
import sys
import gzip
import pickle
import logging
import argparse

import numpy as np

import gym

from models.models import UsdqnModel
from core.buffers import ExperienceReplayBuffer
from core.qlearning import do_online_qlearning
from core.testing import do_testing
from core.utils import evaluate_random

from envs.envs import Continuous_UsdqnOneDoFEnv

NUM_RUNS = 1

ENVS = {
    '1dof': {
        'env_name': 'Continuous 1 DoF',
        'env_cls': Continuous_UsdqnOneDoFEnv,
        'learning_rate': 0.0001,
        'gpu_device': '/gpu:0'
    },
}

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--env', nargs='?', type=str,
                      help='Select environment to train')
    parser.add_argument('--train', nargs='?', const=True, type=bool,
                      default=False,
                      help='If true, train model with fixed learning rate.')
    parser.add_argument('--test', nargs='?', const=True, type=bool,
                      default=False,
                      help='If true, train model with fixed learning rate.')
    parser.add_argument('--render', nargs='?', const=True, type=bool,
                      default=False,
                      help='If true, train model with fixed learning rate.')
    parser.add_argument('--notraining', nargs='?', const=True, type=bool,
                      default=False,
                      help='If true, train model with fixed learning rate.')
    parser.add_argument('--random', nargs='?', const=True, type=bool,
                      default=False,
                      help='If true, train model with fixed learning rate.')
    parser.add_argument('--episodes', nargs='?', type=int,
                      help='Number of episodes for testing')

    FLAGS, _ = parser.parse_known_args()

    try:
        tenv = ENVS[FLAGS.env]
    except KeyError:
        print('Env does not exist.')
        raise

    main_dumps_dir = os.path.join(
        os.path.dirname(__file__), 'dumps', FLAGS.env)      
    if not os.path.exists(main_dumps_dir):
        os.mkdir(main_dumps_dir)

    env = tenv['env_cls'](is_training=True)
    test_env = tenv['env_cls'](is_training=False)
    epsilon_s = { 'start': 0.5, 'end': 0.005, 'decay': 2000 }

    if FLAGS.train:
        for i in range(NUM_RUNS):
            print('Running %s run %d/%d ...' % (tenv['env_name'], (i+1), NUM_RUNS))

            dumps_dir = os.path.join(main_dumps_dir, '%01d' % (i+1))      
            if not os.path.exists(dumps_dir):
                os.mkdir(dumps_dir)

            losses_file = os.path.join(
                dumps_dir, 'losses.csv')

            results_file = os.path.join(
                    dumps_dir, 'results.csv')

            loss, means = do_online_qlearning(env, test_env,
                                model=UsdqnModel(env.action_space.n), 
                                learning_rate=tenv['learning_rate'],
                                epsilon_s=epsilon_s, 
                                gpu_device=tenv['gpu_device'],
                                target_model=UsdqnModel(env.action_space.n, varscope='target'),
                                replay_buffer=ExperienceReplayBuffer(5000, 64),
                                dpaths=os.path.join(dumps_dir, FLAGS.env))

            np.savetxt(losses_file, loss, delimiter=',')
            np.savetxt(results_file, means, delimiter=',')

    elif FLAGS.test:
        dpaths = [os.path.join(main_dumps_dir, '1'), 
            os.path.join(main_dumps_dir, '1', '%s.meta' % FLAGS.env)]

        if not FLAGS.episodes:
            num_episodes = 100
        else:
            num_episodes = FLAGS.episodes

        do_testing(env,
                    model=UsdqnModel(env.action_space.n), 
                    target_model=UsdqnModel(env.action_space.n, varscope='target'),
                    dpaths=dpaths, 
                    render=FLAGS.render, 
                    num_episodes=num_episodes)

    elif FLAGS.notraining:
        do_online_qlearning(env, test_env,
                            model=UsdqnModel(env.action_space.n), 
                            learning_rate=tenv['learning_rate'],
                            epsilon_s=epsilon_s, 
                            gpu_device=tenv['gpu_device'],
                            target_model=UsdqnModel(env.action_space.n, varscope='target'),
                            replay_buffer=ExperienceReplayBuffer(500, 64),
                            dpaths=None,
                            training=False)

    elif FLAGS.random:
        evaluate_random(env,
                num_episodes=100, 
                gamma=0.99, 
                silent=False,
                render=FLAGS.render)



