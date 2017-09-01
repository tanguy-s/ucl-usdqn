from main import *
from subprocess import call

envs = [
    #'1dof_da_s_2',
    #'1dof_da_s_0',
    #'1dof_da_s_1',
    # '1dof_ba_s_0', 
    # '1dof_ba_s_1', 
    # '1dof_ba_s_2', 
    #'1dof_da_u_0', 
    #'1dof_da_u_0', 
    #'1dof_da_u_0', 
    #'1dof_da_s_0'
    #'1dof_da_u_2', 
    #'1dof_da_u_3']

    # '1dof_da_u_3', 
    # '1dof_da_u_4', 
    # '1dof_da_u_5', 
    # '1dof_da_u_6', 
    '1dof_da_u_4_2', 
]

for env in envs:
    print("Starting :", env)
    call(['python', 'main.py', '-e', env, '--train', '--gpu', '1'])