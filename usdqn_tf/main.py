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
from core.dqlearning import do_online_double_qlearning
from core.testing import do_testing
from core.utils import evaluate_random

from envs.envs import (
    #Continuous_UsdqnOneDoFEnv,
    # UsdqnOneDoFSimulatorDiscreteActions,
    # UsdqnOneDoFSimulatorTwoActions,
    # UsdqnOneDoFSimulatorTwoActionsSl,
    # UsdqnOneDoFSimulatorTwoActionsSlStay,
    # UsdqnOneDoFSimulatorSixActions,
    # UsdqnOneDoFSimulatorDiscreteActionsSemi
    UsdqnOneDoFEnv,
    OneDoFSim_BinaryActions_Supervised,
    OneDoFSim_DiscActions_Unsupervised,
    OneDoFSim_DiscActions_Supervised,
)


NUM_RUNS = 1

ENVS = {
    '1dof_ba_s_0': {
        'env_name': '1 DoF Binary Actions Supervised',
        'train_env': UsdqnOneDoFEnv(OneDoFSim_BinaryActions_Supervised(0.02, is_training=True)),
        'test_env': UsdqnOneDoFEnv(OneDoFSim_BinaryActions_Supervised(0.02, is_training=False)),
        'learning_rate': 0.001,
        'gpu_device': '/gpu:1',
        'exp_replay': ExperienceReplayBuffer(200000, 32),
        'epsilon': { 'start': 1, 'end': 0.1, 'decay': 500000 },
        'params': {
            'TRAINING_STEPS':800000,
            'LOG_STEPS':2000,
            'LOSS_STEPS':1000,
            'EVAL_STEPS':6000,
            'SAVE_STEPS':10000,
            'TARGET_UPDATE':5000,
            'EVAL_EPISODES':20,
        }
    },
    '1dof_ba_s_1': {
        'env_name': '1 DoF Binary Actions Supervised',
        'train_env': UsdqnOneDoFEnv(OneDoFSim_BinaryActions_Supervised(0.02, is_training=True)),
        'test_env': UsdqnOneDoFEnv(OneDoFSim_BinaryActions_Supervised(0.02, is_training=False)),
        'learning_rate': 0.0001,
        'gpu_device': '/gpu:1',
        'exp_replay': ExperienceReplayBuffer(200000, 32),
        'epsilon': { 'start': 1, 'end': 0.1, 'decay': 500000 },
        'params': {
            'TRAINING_STEPS':800000,
            'LOG_STEPS':2000,
            'LOSS_STEPS':1000,
            'EVAL_STEPS':6000,
            'SAVE_STEPS':10000,
            'TARGET_UPDATE':5000,
            'EVAL_EPISODES':20,
        }
    },
    '1dof_ba_s_2': {
        'env_name': '1 DoF Binary Actions Supervised',
        'train_env': UsdqnOneDoFEnv(OneDoFSim_BinaryActions_Supervised(0.02, is_training=True)),
        'test_env': UsdqnOneDoFEnv(OneDoFSim_BinaryActions_Supervised(0.02, is_training=False)),
        'learning_rate': 0.01,
        'gpu_device': '/gpu:1',
        'exp_replay': ExperienceReplayBuffer(200000, 32),
        'epsilon': { 'start': 1, 'end': 0.1, 'decay': 500000 },
        'params': {
            'TRAINING_STEPS':800000,
            'LOG_STEPS':2000,
            'LOSS_STEPS':1000,
            'EVAL_STEPS':6000,
            'SAVE_STEPS':10000,
            'TARGET_UPDATE':5000,
            'EVAL_EPISODES':20,
        }
    },
    '1dof_da_u_0': {
        'env_name': '1 DoF Discrete Actions Unupervised',
        'train_env': UsdqnOneDoFEnv(OneDoFSim_DiscActions_Unsupervised(0.02, is_training=True)),
        'test_env': UsdqnOneDoFEnv(OneDoFSim_DiscActions_Unsupervised(0.02, is_training=False)),
        'learning_rate': 0.001,
        'gpu_device': '/gpu:1',
        'exp_replay': ExperienceReplayBuffer(200000, 32),
        'epsilon': { 'start': 1, 'end': 0.1, 'decay': 500000 },
        'params': {
            'TRAINING_STEPS':800000,
            'LOG_STEPS':2000,
            'LOSS_STEPS':1000,
            'EVAL_STEPS':6000,
            'SAVE_STEPS':10000,
            'TARGET_UPDATE':5000,
            'EVAL_EPISODES':20,
        }
    },
    '1dof_da_u_1': {
        'env_name': '1 DoF Discrete Actions Unupervised',
        'train_env': UsdqnOneDoFEnv(OneDoFSim_DiscActions_Unsupervised(0.02, is_training=True)),
        'test_env': UsdqnOneDoFEnv(OneDoFSim_DiscActions_Unsupervised(0.02, is_training=False)),
        'learning_rate': 0.0001,
        'gpu_device': '/gpu:1',
        'exp_replay': ExperienceReplayBuffer(200000, 32),
        'epsilon': { 'start': 1, 'end': 0.1, 'decay': 500000 },
        'params': {
            'TRAINING_STEPS':800000,
            'LOG_STEPS':2000,
            'LOSS_STEPS':1000,
            'EVAL_STEPS':6000,
            'SAVE_STEPS':10000,
            'TARGET_UPDATE':5000,
            'EVAL_EPISODES':20,
        }
    },
    '1dof_da_u_2': {
        'env_name': '1 DoF Discrete Actions Unupervised',
        'train_env': UsdqnOneDoFEnv(OneDoFSim_DiscActions_Unsupervised(0.04, is_training=True)),
        'test_env': UsdqnOneDoFEnv(OneDoFSim_DiscActions_Unsupervised(0.04, is_training=False)),
        'learning_rate': 0.0001,
        'gpu_device': '/gpu:1',
        'exp_replay': ExperienceReplayBuffer(200000, 32),
        'epsilon': { 'start': 1, 'end': 0.1, 'decay': 500000 },
        'params': {
            'TRAINING_STEPS':800000,
            'LOG_STEPS':2000,
            'LOSS_STEPS':1000,
            'EVAL_STEPS':6000,
            'SAVE_STEPS':10000,
            'TARGET_UPDATE':5000,
            'EVAL_EPISODES':20,
        }
    },
    '1dof_da_u_3': {
        'env_name': '1 DoF Discrete Actions Unupervised',
        'train_env': UsdqnOneDoFEnv(OneDoFSim_DiscActions_Unsupervised(0.08, is_training=True)),
        'test_env': UsdqnOneDoFEnv(OneDoFSim_DiscActions_Unsupervised(0.08, is_training=False)),
        'learning_rate': 0.0001,
        'gpu_device': '/gpu:1',
        'exp_replay': ExperienceReplayBuffer(200000, 32),
        'epsilon': { 'start': 1, 'end': 0.1, 'decay': 500000 },
        'params': {
            'TRAINING_STEPS':800000,
            'LOG_STEPS':2000,
            'LOSS_STEPS':1000,
            'EVAL_STEPS':6000,
            'SAVE_STEPS':10000,
            'TARGET_UPDATE':5000,
            'EVAL_EPISODES':20,
        }
    },
    '1dof_da_s_0': {
        'env_name': '1 DoF Discrete Actions Supervised',
        'train_env': UsdqnOneDoFEnv(OneDoFSim_DiscActions_Supervised(0.02, is_training=True)),
        'test_env': UsdqnOneDoFEnv(OneDoFSim_DiscActions_Supervised(0.02, is_training=False)),
        'learning_rate': 0.001,
        'gpu_device': '/gpu:1',
        'exp_replay': ExperienceReplayBuffer(200000, 32),
        'epsilon': { 'start': 1, 'end': 0.1, 'decay': 500000 },
        'params': {
            'TRAINING_STEPS':800000,
            'LOG_STEPS':2000,
            'LOSS_STEPS':1000,
            'EVAL_STEPS':6000,
            'SAVE_STEPS':10000,
            'TARGET_UPDATE':5000,
            'EVAL_EPISODES':20,
        }
    },
    '1dof_da_s_1': {
        'env_name': '1 DoF Discrete Actions Supervised',
        'train_env': UsdqnOneDoFEnv(OneDoFSim_DiscActions_Supervised(0.02, is_training=True)),
        'test_env': UsdqnOneDoFEnv(OneDoFSim_DiscActions_Supervised(0.02, is_training=False)),
        'learning_rate': 0.0001,
        'gpu_device': '/gpu:1',
        'exp_replay': ExperienceReplayBuffer(200000, 32),
        'epsilon': { 'start': 1, 'end': 0.1, 'decay': 500000 },
        'params': {
            'TRAINING_STEPS':800000,
            'LOG_STEPS':2000,
            'LOSS_STEPS':1000,
            'EVAL_STEPS':6000,
            'SAVE_STEPS':10000,
            'TARGET_UPDATE':5000,
            'EVAL_EPISODES':20,
        }
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

    env = tenv['train_env']
    test_env = tenv['test_env']

    if not FLAGS.episodes:
        num_episodes = 100
    else:
        num_episodes = FLAGS.episodes

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
                                params=tenv['params'],
                                learning_rate=tenv['learning_rate'],
                                epsilon_s=tenv['epsilon'], 
                                gpu_device=tenv['gpu_device'],
                                target_model=UsdqnModel(env.action_space.n, varscope='target'),
                                replay_buffer=tenv['exp_replay'], # 10000 -> 100 lr 0.01
                                dpaths=os.path.join(dumps_dir, FLAGS.env))

            # loss, means = do_online_double_qlearning(env, test_env,
            #                     model_1=UsdqnModel(env.action_space.n, varscope='1'), 
            #                     model_2=UsdqnModel(env.action_space.n, varscope='2'), 
            #                     params=tenv['params'],
            #                     learning_rate=tenv['learning_rate'],
            #                     epsilon_s=tenv['epsilon'], 
            #                     gpu_device=tenv['gpu_device'],
            #                     target_model=UsdqnModel(env.action_space.n, varscope='target'),
            #                     replay_buffer=tenv['exp_replay'], # 10000 -> 100 lr 0.01
            #                     dpaths=os.path.join(dumps_dir, FLAGS.env))

            np.savetxt(losses_file, loss, delimiter=',')
            np.savetxt(results_file, means, delimiter=',')

    elif FLAGS.test:
        dpaths = [os.path.join(main_dumps_dir, '1'), 
            os.path.join(main_dumps_dir, '1', '%s.meta' % FLAGS.env)]

        do_testing(test_env,
                    model=UsdqnModel(env.action_space.n), 
                    target_model=UsdqnModel(env.action_space.n, varscope='target'),
                    dpaths=dpaths, 
                    render=FLAGS.render, 
                    num_episodes=num_episodes)

    elif FLAGS.notraining:
        do_online_qlearning(env, test_env,
                            model=UsdqnModel(env.action_space.n), 
                            params=tenv['params'],
                            learning_rate=tenv['learning_rate'],
                            epsilon_s=tenv['epsilon'], 
                            gpu_device=tenv['gpu_device'],
                            target_model=UsdqnModel(env.action_space.n, varscope='target'),
                            replay_buffer=tenv['exp_replay'],
                            dpaths=None,
                            training=False)

    elif FLAGS.random:

        evaluate_random(env,
                num_episodes=num_episodes, 
                gamma=0.99, 
                silent=False,
                render=FLAGS.render)



