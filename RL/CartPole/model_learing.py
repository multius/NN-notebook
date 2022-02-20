import lib
from lib import DQN
import model_make

dqn = DQN.load(model_make.full_path)
dqn.is_learning = True
env = lib.Environment()

for i in range(10):
    dqn.run_in(env, times = 100, need_history = False, need_window = False)

    dqn.save(model_make.full_path)