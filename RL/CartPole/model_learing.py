import lib
from lib import DQN
import model_make

dqn = DQN.load(model_make.full_path)
env = lib.Environment()

for i in range(5):
    dqn.run_in(env, times = 100, need_history = False)

    dqn.save(model_make.full_path)