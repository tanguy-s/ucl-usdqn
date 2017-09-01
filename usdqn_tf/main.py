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
    OneDoFSim_BinaryActions_Unsupervised,
    OneDoFSim_DiscActions_Unsupervised,
    OneDoFSim_DiscActions_Supervised,
)


NUM_RUNS = 1

ENVS = {
    # '1dof_ba_s_0': {
    #     'env_name': '1 DoF Binary Actions Supervised',
    #     'train_env': UsdqnOneDoFEnv(OneDoFSim_BinaryActions_Supervised(0.04, is_training=True, sample=True)),
    #     'test_env': UsdqnOneDoFEnv(OneDoFSim_BinaryActions_Supervised(0.04, is_training=False, sample=True)),
    #     'learning_rate': 0.00025,
    #     'gpu_device': '/gpu:1',
    #     'exp_replay': ExperienceReplayBuffer(200000, 64),
    #     'epsilon': { 'start': 0.7, 'end': 0.1, 'decay': 500000 },
    #     'params': {
    #         'TRAINING_STEPS':700000,
    #         'LOG_STEPS':2000,
    #         'LOSS_STEPS':1000,
    #         'EVAL_STEPS':6000,
    #         'SAVE_STEPS':100000,
    #         'TARGET_UPDATE':5000,
    #         'EVAL_EPISODES':40,
    #     }
    # },
    # '1dof_ba_s_1': {
    #     'env_name': '1 DoF Binary Actions Supervised',
    #     'train_env': UsdqnOneDoFEnv(OneDoFSim_BinaryActions_Supervised(0.04, is_training=True)),
    #     'test_env': UsdqnOneDoFEnv(OneDoFSim_BinaryActions_Supervised(0.04, is_training=False)),
    #     'learning_rate': 0.0001,
    #     'gpu_device': '/gpu:1',
    #     'exp_replay': ExperienceReplayBuffer(200000, 64),
    #     'epsilon': { 'start': 0.6, 'end': 0.1, 'decay': 500000 },
    #     'params': {
    #         'TRAINING_STEPS':700000,
    #         'LOG_STEPS':2000,
    #         'LOSS_STEPS':1000,
    #         'EVAL_STEPS':6000,
    #         'SAVE_STEPS':100000,
    #         'TARGET_UPDATE':5000,
    #         'EVAL_EPISODES':40,
    #     }
    # },
    # '1dof_ba_s_2': {
    #     'env_name': '1 DoF Binary Actions Supervised',
    #     'train_env': UsdqnOneDoFEnv(OneDoFSim_BinaryActions_Supervised(0.04, is_training=True)),
    #     'test_env': UsdqnOneDoFEnv(OneDoFSim_BinaryActions_Supervised(0.04, is_training=False)),
    #     'learning_rate': 0.01,
    #     'gpu_device': '/gpu:1',
    #     'exp_replay': ExperienceReplayBuffer(200000, 64),
    #     'epsilon': { 'start': 0.6, 'end': 0.1, 'decay': 500000 },
    #     'params': {
    #         'TRAINING_STEPS':700000,
    #         'LOG_STEPS':2000,
    #         'LOSS_STEPS':1000,
    #         'EVAL_STEPS':6000,
    #         'SAVE_STEPS':100000,
    #         'TARGET_UPDATE':5000,
    #         'EVAL_EPISODES':40,
    #     }
    # },
    # '1dof_ba_s_3': {
    #     'env_name': '1 DoF Binary Actions Supervised',
    #     'train_env': UsdqnOneDoFEnv(OneDoFSim_BinaryActions_Supervised(0.1, is_training=True)),
    #     'test_env': UsdqnOneDoFEnv(OneDoFSim_BinaryActions_Supervised(0.1, is_training=False)),
    #     'learning_rate': 0.01,
    #     'gpu_device': '/gpu:1',
    #     'exp_replay': ExperienceReplayBuffer(200000, 64),
    #     'epsilon': { 'start': 0.6, 'end': 0.1, 'decay': 500000 },
    #     'params': {
    #         'TRAINING_STEPS':700000,
    #         'LOG_STEPS':2000,
    #         'LOSS_STEPS':1000,
    #         'EVAL_STEPS':6000,
    #         'SAVE_STEPS':100000,
    #         'TARGET_UPDATE':5000,
    #         'EVAL_EPISODES':40,
    #     }
    # },
    # '1dof_ba_u_0': {
    #     'env_name': '1 DoF Binary Actions Unsupervised',
    #     'train_env': UsdqnOneDoFEnv(OneDoFSim_BinaryActions_Unsupervised(0.04, is_training=True, sample=True)),
    #     'test_env': UsdqnOneDoFEnv(OneDoFSim_BinaryActions_Unsupervised(0.04, is_training=False, sample=True)),
    #     'learning_rate': 0.00025,
    #     'gpu_device': '/gpu:1',
    #     'exp_replay': ExperienceReplayBuffer(200000, 64),
    #     'epsilon': { 'start': 0.7, 'end': 0.1, 'decay': 500000 },
    #     'params': {
    #         'TRAINING_STEPS':700000,
    #         'LOG_STEPS':2000,
    #         'LOSS_STEPS':1000,
    #         'EVAL_STEPS':6000,
    #         'SAVE_STEPS':100000,
    #         'TARGET_UPDATE':5000,
    #         'EVAL_EPISODES':40,
    #     }
    # },
    # '1dof_ba_u_1': {
    #     'env_name': '1 DoF Binary Actions Unsupervised',
    #     'train_env': UsdqnOneDoFEnv(OneDoFSim_BinaryActions_Unsupervised(0.04, is_training=True, sample=False)),
    #     'test_env': UsdqnOneDoFEnv(OneDoFSim_BinaryActions_Unsupervised(0.04, is_training=False, sample=False)),
    #     'learning_rate': 0.0001,
    #     'gpu_device': '/gpu:1',
    #     'exp_replay': ExperienceReplayBuffer(200000, 64),
    #     'epsilon': { 'start': 0.6, 'end': 0.1, 'decay': 500000 },
    #     'params': {
    #         'TRAINING_STEPS':700000,
    #         'LOG_STEPS':2000,
    #         'LOSS_STEPS':1000,
    #         'EVAL_STEPS':6000,
    #         'SAVE_STEPS':100000,
    #         'TARGET_UPDATE':5000,
    #         'EVAL_EPISODES':40,
    #     }
    # },
    '1dof_da_u_0': {
        'env_name': '1 DoF Discrete Actions Unupervised',
        'train_env': UsdqnOneDoFEnv(OneDoFSim_DiscActions_Unsupervised(0.04, is_training=True, sample=True)),
        'test_env': UsdqnOneDoFEnv(OneDoFSim_DiscActions_Unsupervised(0.04, is_training=False, sample=True)),
        'learning_rate': 0.00025, #0.00025
        'gpu_device': '/gpu:1',
        'exp_replay': ExperienceReplayBuffer(200000, 64),
        'epsilon': { 'start': 0.7, 'end': 0.1, 'decay': 500000 },
        'params': {
            'TRAINING_STEPS':700000,
            'LOG_STEPS':2000,
            'LOSS_STEPS':1000,
            'EVAL_STEPS':2000,
            'SAVE_STEPS':100000,
            'TARGET_UPDATE':5000,
            'EVAL_EPISODES':100,
        }
    },
    '1dof_da_u_00': {
        'env_name': '1 DoF Discrete Actions Unupervised',
        'train_env': UsdqnOneDoFEnv(OneDoFSim_DiscActions_Unsupervised(0.04, is_training=True, sample=True)),
        'test_env': UsdqnOneDoFEnv(OneDoFSim_DiscActions_Unsupervised(0.04, is_training=False, sample=True)),
        'learning_rate': 0.00025, #0.00025
        'gpu_device': '/gpu:1',
        'exp_replay': ExperienceReplayBuffer(200000, 64),
        'epsilon': { 'start': 0.7, 'end': 0.1, 'decay': 500000 },
        'params': {
            'TRAINING_STEPS':700000,
            'LOG_STEPS':2000,
            'LOSS_STEPS':1000,
            'EVAL_STEPS':2000,
            'SAVE_STEPS':100000,
            'TARGET_UPDATE':5000,
            'EVAL_EPISODES':100,
        }
    },
    '1dof_da_u_1': {
        'env_name': '1 DoF Discrete Actions Unupervised',
        'train_env': UsdqnOneDoFEnv(OneDoFSim_DiscActions_Unsupervised(0.04, is_training=True, sample=True)),
        'test_env': UsdqnOneDoFEnv(OneDoFSim_DiscActions_Unsupervised(0.04, is_training=False, sample=True)),
        'learning_rate': 0.001, #0.00025
        'gpu_device': '/gpu:1',
        'exp_replay': ExperienceReplayBuffer(200000, 64),
        'epsilon': { 'start': 0.7, 'end': 0.1, 'decay': 500000 },
        'params': {
            'TRAINING_STEPS':700000,
            'LOG_STEPS':2000,
            'LOSS_STEPS':1000,
            'EVAL_STEPS':2000,
            'SAVE_STEPS':100000,
            'TARGET_UPDATE':5000,
            'EVAL_EPISODES':100,
        }
    },
    '1dof_da_u_2': {
        'env_name': '1 DoF Discrete Actions Unupervised',
        'train_env': UsdqnOneDoFEnv(OneDoFSim_DiscActions_Unsupervised(0.04, is_training=True, sample=True)),
        'test_env': UsdqnOneDoFEnv(OneDoFSim_DiscActions_Unsupervised(0.04, is_training=False, sample=True)),
        'learning_rate': 0.0001, #0.00025
        'gpu_device': '/gpu:1',
        'exp_replay': ExperienceReplayBuffer(200000, 64),
        'epsilon': { 'start': 0.7, 'end': 0.1, 'decay': 500000 },
        'params': {
            'TRAINING_STEPS':700000,
            'LOG_STEPS':2000,
            'LOSS_STEPS':1000,
            'EVAL_STEPS':2000,
            'SAVE_STEPS':100000,
            'TARGET_UPDATE':5000,
            'EVAL_EPISODES':100,
        }
    },
    '1dof_da_u_3': {
        'env_name': '1 DoF Discrete Actions Unupervised',
        'train_env': UsdqnOneDoFEnv(OneDoFSim_DiscActions_Unsupervised(0.04, is_training=True, sample=True)),
        'test_env': UsdqnOneDoFEnv(OneDoFSim_DiscActions_Unsupervised(0.04, is_training=False, sample=True)),
        'learning_rate': 0.0005, #0.00025
        'gpu_device': '/gpu:1',
        'exp_replay': ExperienceReplayBuffer(200000, 64),
        'epsilon': { 'start': 0.7, 'end': 0.1, 'decay': 500000 },
        'params': {
            'TRAINING_STEPS':700000,
            'LOG_STEPS':2000,
            'LOSS_STEPS':1000,
            'EVAL_STEPS':2000,
            'SAVE_STEPS':100000,
            'TARGET_UPDATE':5000,
            'EVAL_EPISODES':100,
        }
    },
    '1dof_da_u_4': {
        'env_name': '1 DoF Discrete Actions Unupervised',
        'train_env': UsdqnOneDoFEnv(OneDoFSim_DiscActions_Unsupervised(0.04, is_training=True, sample=True)),
        'test_env': UsdqnOneDoFEnv(OneDoFSim_DiscActions_Unsupervised(0.04, is_training=False, sample=True)),
        'learning_rate': 0.0008, #0.00025
        'gpu_device': '/gpu:1',
        'exp_replay': ExperienceReplayBuffer(200000, 64),
        'epsilon': { 'start': 0.7, 'end': 0.1, 'decay': 500000 },
        'params': {
            'TRAINING_STEPS':700000,
            'LOG_STEPS':2000,
            'LOSS_STEPS':1000,
            'EVAL_STEPS':2000,
            'SAVE_STEPS':100000,
            'TARGET_UPDATE':5000,
            'EVAL_EPISODES':100,
        }
    },
    '1dof_da_u_5': {
        'env_name': '1 DoF Discrete Actions Unupervised',
        'train_env': UsdqnOneDoFEnv(OneDoFSim_DiscActions_Unsupervised(0.04, is_training=True, sample=True)),
        'test_env': UsdqnOneDoFEnv(OneDoFSim_DiscActions_Unsupervised(0.04, is_training=False, sample=True)),
        'learning_rate': 0.00008, #0.00025
        'gpu_device': '/gpu:1',
        'exp_replay': ExperienceReplayBuffer(200000, 64),
        'epsilon': { 'start': 0.7, 'end': 0.1, 'decay': 500000 },
        'params': {
            'TRAINING_STEPS':700000,
            'LOG_STEPS':2000,
            'LOSS_STEPS':1000,
            'EVAL_STEPS':2000,
            'SAVE_STEPS':100000,
            'TARGET_UPDATE':5000,
            'EVAL_EPISODES':100,
        }
    },
    '1dof_da_u_6': {
        'env_name': '1 DoF Discrete Actions Unupervised',
        'train_env': UsdqnOneDoFEnv(OneDoFSim_DiscActions_Unsupervised(0.04, is_training=True, sample=True)),
        'test_env': UsdqnOneDoFEnv(OneDoFSim_DiscActions_Unsupervised(0.04, is_training=False, sample=True)),
        'learning_rate': 0.00005, #0.00025
        'gpu_device': '/gpu:1',
        'exp_replay': ExperienceReplayBuffer(200000, 64),
        'epsilon': { 'start': 0.7, 'end': 0.1, 'decay': 500000 },
        'params': {
            'TRAINING_STEPS':700000,
            'LOG_STEPS':2000,
            'LOSS_STEPS':1000,
            'EVAL_STEPS':2000,
            'SAVE_STEPS':100000,
            'TARGET_UPDATE':5000,
            'EVAL_EPISODES':100,
        }
    },

    '1dof_da_u_4_1': {
        'env_name': '1 DoF Discrete Actions Unupervised',
        'train_env': UsdqnOneDoFEnv(OneDoFSim_DiscActions_Unsupervised(0.015, is_training=True, sample=True)),
        'test_env': UsdqnOneDoFEnv(OneDoFSim_DiscActions_Unsupervised(0.015, is_training=False, sample=True)),
        'learning_rate': 0.0008, #0.00025
        'gpu_device': '/gpu:1',
        'exp_replay': ExperienceReplayBuffer(200000, 64),
        'epsilon': { 'start': 0.7, 'end': 0.1, 'decay': 500000 },
        'params': {
            'TRAINING_STEPS':700000,
            'LOG_STEPS':2000,
            'LOSS_STEPS':1000,
            'EVAL_STEPS':2000,
            'SAVE_STEPS':100000,
            'TARGET_UPDATE':5000,
            'EVAL_EPISODES':100,
        }
    },
    '1dof_da_u_4_2': {
        'env_name': '1 DoF Discrete Actions Unupervised',
        'train_env': UsdqnOneDoFEnv(OneDoFSim_DiscActions_Unsupervised(0.02, is_training=True, sample=True)),
        'test_env': UsdqnOneDoFEnv(OneDoFSim_DiscActions_Unsupervised(0.02, is_training=False, sample=True)),
        'learning_rate': 0.0008, #0.00025
        'gpu_device': '/gpu:1',
        'exp_replay': ExperienceReplayBuffer(200000, 64),
        'epsilon': { 'start': 0.7, 'end': 0.1, 'decay': 500000 },
        'params': {
            'TRAINING_STEPS':700000,
            'LOG_STEPS':2000,
            'LOSS_STEPS':1000,
            'EVAL_STEPS':2000,
            'SAVE_STEPS':100000,
            'TARGET_UPDATE':5000,
            'EVAL_EPISODES':100,
        }
    },
    '1dof_da_u_4_3': {
        'env_name': '1 DoF Discrete Actions Unupervised',
        'train_env': UsdqnOneDoFEnv(OneDoFSim_DiscActions_Unsupervised(0.08, is_training=True, sample=True)),
        'test_env': UsdqnOneDoFEnv(OneDoFSim_DiscActions_Unsupervised(0.08, is_training=False, sample=True)),
        'learning_rate': 0.0008, #0.00025
        'gpu_device': '/gpu:1',
        'exp_replay': ExperienceReplayBuffer(200000, 64),
        'epsilon': { 'start': 0.7, 'end': 0.1, 'decay': 500000 },
        'params': {
            'TRAINING_STEPS':700000,
            'LOG_STEPS':2000,
            'LOSS_STEPS':1000,
            'EVAL_STEPS':2000,
            'SAVE_STEPS':100000,
            'TARGET_UPDATE':5000,
            'EVAL_EPISODES':100,
        }
    },
    '1dof_da_u_4_m': {
        'env_name': '1 DoF Discrete Actions Unupervised',
        'train_env': UsdqnOneDoFEnv(OneDoFSim_DiscActions_Unsupervised(0.04, is_training=True, sample=True)),
        'test_env': UsdqnOneDoFEnv(OneDoFSim_DiscActions_Unsupervised(0.04, is_training=False, sample=True)),
        'learning_rate': 0.0008, #0.00025
        'gpu_device': '/gpu:1',
        'exp_replay': ExperienceReplayBuffer(200000, 64),
        'epsilon': { 'start': 0.7, 'end': 0.1, 'decay': 500000 },
        'params': {
            'NUM_RUNS': 10,
            #'TRAINING_STEPS':400000,
            'TRAINING_STEPS':10000,
            'LOG_STEPS':2000,
            'LOSS_STEPS':1000,
            'EVAL_STEPS':2000,
            'EVAL_STEPS_START':100,
            'EVAL_STEPS_STOP':10000,
            'SAVE_STEPS':100000,
            'TARGET_UPDATE':5000,
            'EVAL_EPISODES':100,
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
    parser.add_argument('--gpu', nargs='?', type=int,
                      help='GPU id')

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

    if FLAGS.gpu is not None:
        tenv['gpu_device'] = '/gpu:%s' % FLAGS.gpu

    if FLAGS.train:

        if 'NUM_RUNS' in tenv['params']:
            NUM_RUNS = tenv['params']['NUM_RUNS']

        m_loss, m_means, m_stds = None, None, None
        for i in range(NUM_RUNS):
            print('Running %s run %d/%d ...' % (tenv['env_name'], (i+1), NUM_RUNS))

            dumps_dir = os.path.join(main_dumps_dir, '%01d' % (i+1))      
            if not os.path.exists(dumps_dir):
                os.mkdir(dumps_dir)

            losses_file = os.path.join(
                dumps_dir, 'losses_%s.csv' % (i+1))

            results_file = os.path.join(
                    dumps_dir, 'results_%s.csv' % (i+1))

            stds_file = os.path.join(
                    dumps_dir, 'stds_%s.csv' % (i+1))

            loss, means, stds = do_online_qlearning(env, test_env,
                                model=UsdqnModel(env.action_space.n), 
                                params=tenv['params'],
                                learning_rate=tenv['learning_rate'],
                                epsilon_s=tenv['epsilon'], 
                                gpu_device=tenv['gpu_device'],
                                target_model=UsdqnModel(env.action_space.n, varscope='target'),
                                replay_buffer=tenv['exp_replay'], # 10000 -> 100 lr 0.01
                                dpaths=os.path.join(dumps_dir, FLAGS.env))


            np.savetxt(losses_file, loss, delimiter=',')
            np.savetxt(results_file, means, delimiter=',')
            np.savetxt(stds_file, stds, delimiter=',')

            if m_loss is None:
                m_loss = loss
                m_means = means
                m_stds = stds
            else:
                m_loss = np.concatenate([m_loss, loss], axis=1)
                m_means = np.concatenate([m_means, means], axis=1)
                m_stds = np.concatenate([m_stds, stds], axis=1)

        np.savetxt(os.path.join(
                    dumps_dir, 'losses.csv'), m_loss, delimiter=',')
        np.savetxt(os.path.join(
                    dumps_dir, 'results.csv'), m_means, delimiter=',')
        np.savetxt(os.path.join(
                    dumps_dir, 'stds.csv'), m_stds, delimiter=',')

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



