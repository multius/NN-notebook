import lib
import torch
import os

dqn_model = lib.DQN()
path = './model/'
model_name = 'dqn_model'
full_path = ''.join([path, model_name])

if __name__ == '__main__':
    if os.path.exists(path) == False:
        os.mkdir(path)

    dqn_model.save(full_path)