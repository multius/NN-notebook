import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym

# 环境超参数
N_STATES = 4                # 状态数量
N_ACTIONS = 2               # 动作数量

# DQN超参数 
EPSILON = 0.9               # 贪心比例
LR = 0.01                   # 学习率
GAMMA = 0.9                 # 价值折扣系数
TARGET_REPLACE_ITER = 100   # target网络更新率
 
# Memory超参数
MEMORY_CAPACITY = 2000      # 记忆容量
BATCH_SIZE = 32             # 批次大小


# Q-Net类
class QNet(nn.Module):
    def __init__(self, n_states, n_actions):
        super(QNet, self).__init__()

        self.fc1 = nn.Linear(n_states, 50)      # 输入层
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(50, n_actions)     # 输出层
        self.fc2.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        output = self.fc2(x)

        return output
        
# 记忆模块
class Memory(object):
    def __init__(self, capacity, n_states):
        self.capacity = capacity                                        # 容量
        self.n_states = n_states
        self.data_cnt = 0                                               # 数据添加次数
        self.data = np.zeros((self.capacity, self.n_states * 2 + 2))    # 储存数据的空间
        self.is_full = False                                            # 是否装满的标记

    #记忆储存函数
    def push(self, state, action, reward, next_state):
        t = np.hstack((state, action, reward, next_state))
        index = self.data_cnt % self.capacity                           # 循环利用记忆空间（最新的数据覆盖最旧的数据）
        self.data[index, :] = t

        if self.data_cnt >= self.capacity:                              # 记忆容量装满
            self.is_full = True
        self.data_cnt += 1

    # 记忆抽样函数
    def sample(self, batch_size):
        sample_index = np.random.choice(self.capacity, batch_size)      # 随机抽取batch_size份记忆样本
        b_memory = self.data[sample_index, :]

        b_state = torch.FloatTensor(b_memory[:, :self.n_states])        # 记忆样本拆分
        b_action = torch.LongTensor(b_memory[:, self.n_states : self.n_states+1].astype(int))
        b_reward = torch.FloatTensor(b_memory[:, self.n_states+1 : self.n_states+2])
        b_next_state = torch.FloatTensor(b_memory[:, -self.n_states:])
        
        return b_state, b_action, b_reward, b_next_state

# DQN类
class DQN(object):
    def __init__(self):
        self.eval_net = QNet(N_STATES, N_ACTIONS)                       # 需要训练的主网络
        self.target_net = QNet(N_STATES, N_ACTIONS)                     # 辅助主网络训练的target net
        self.learn_step_cnt = 0                                         # 主网络学习次数
        self.memory = Memory(MEMORY_CAPACITY, N_STATES)                 # 记忆模块
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr = LR)
        self.lossf = nn.MSELoss()
        self.is_learning = True                                         # 学习状态标记

    # 动作选择函数
    def choose_action(self, x):                                         
        x = torch.unsqueeze(torch.FloatTensor(x), 0)                    # 将x转换成32-bit floating point形式，并在dim=0增加维数为1的维度
        if self.is_learning and np.random.uniform() < EPSILON:          # 生成一个在[0, 1)内的随机数，如果小于EPSILON，选择最优动作
            actions_value = self.eval_net.forward(x)                    # 使用主网络获取各个动作的价值
            action = torch.max(actions_value, 1)[1].item()              # 输出最大值的索引
        else:
            action = np.random.randint(0, N_ACTIONS)                    # 随机选择动作
        return action

    # 网络学习函数
    def learn(self):
        if self.learn_step_cnt % TARGET_REPLACE_ITER == 0:              # 根据设定周期更新target网络中的数据
            self.target_net.load_state_dict(self.eval_net.state_dict()) 
        self.learn_step_cnt += 1

        b_s, b_a, b_r, b_next_s = self.memory.sample(BATCH_SIZE)        # 数据抽样

        q_eval = self.eval_net(b_s).gather(1, b_a)                      # 获取主网络对应动作价值
        q_next = self.target_net(b_next_s).detach()                     # 获取target网络下一步的最优动作价值
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # 贝尔曼方程

        loss = self.lossf(q_eval, q_target)                             # 计算loss
        self.optimizer.zero_grad()                                      # 清空上一步的残余更新参数值
        loss.backward()                                                 # 误差反向传播, 计算参数更新值
        self.optimizer.step()                                           # 更新评估网络的所有参数

    # 使DQN系统在环境中运行
    def run_in(self, env, times = 500, need_history = True, need_window = True):
        for episode in range(times):
            state = env.reset()                                         # 重置环境
            episode_reward_sum = 0                                      # 初始化该循环的总奖励

            while True:                                                 # 开始一个episode (每一个循环代表一步)
                if need_window:
                    env.render()                                        # 显示实验动画
                action = self.choose_action(state)                      # 输入该步对应的状态，选择动作
                next_state, reward, done, n_steps = env.step(action)    # 执行动作，获得反馈

                self.memory.push(state, action, reward, next_state)     # 存储样本
                episode_reward_sum += reward                            # 逐步累加该循环的总奖励

                print("\r[ episode:{:>4} | steps:{:>5} | reward:{:>7.1f} ]".format(episode + 1, n_steps, episode_reward_sum), end = '')

                state = next_state                                      # 更新环境信息

                if self.is_learning and self.memory.is_full:            # 当DQN处于学习状态且记忆模块装满时开始学习
                    self.learn()

                if done:
                    if need_history:
                        print()
                    break
        print('')

    # 保存模型
    def save(self, path):
        torch.save(self, path)
        print("DQN-model has been saved to", path)

    # 加载模型
    def load(path):
        model = torch.load(path)
        return model

# 环境装饰器
class Environment():
    def __init__(self):
        self.core = gym.make('CartPole-v0').unwrapped                   # 获取环境运行内核
        self.x_threshold = self.core.x_threshold                        # 允许小车水平移动的最大范围
        self.theta_threshold_radians = self.core.theta_threshold_radians# 允许长杆倾斜的最大倾角
        self.n_steps = 0                                                # 环境运行步数
    
    def render(self):
        self.core.render()

    # 重置环境
    def reset(self):
        state = self.core.reset()                                       # 重置环境内核
        self.n_steps = 0                                                # 重置环境运行步数
        return state                                                    # 返回环境初始信息
    
    # 根据输入动作运行环境至下一步
    def step(self, action):
        state, _reward, done, _info = self.core.step(action)            # 从内核中获取信息
        self.n_steps += 1

        x, x_dot, theta, theta_dot = state                              # 拆分环境信息
        r1 = (self.x_threshold - abs(x)) / self.x_threshold - 0.8       # 离屏幕中心越远奖励越低
        r2 = (self.theta_threshold_radians - abs(theta)) / self.theta_threshold_radians - 0.5# 倾斜角度越大奖励越低
        new_reward = r1 + r2

        return state, new_reward, done, self.n_steps                    # 返回经过处理的数据


if __name__ == '__main__':
    dqn = DQN()
    env = Environment()
    dqn.run_in(env)