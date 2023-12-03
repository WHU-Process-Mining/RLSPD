import os
import pickle
import random
from itertools import product
import time

import numpy as np
import torch
from heuristic_agent import HeuristicAgent
from process_discovery_environment import ProcessDiscoveryEnvironment
from td3_agent import TD3Agent, gru_type
from utils import Order, Algo, drift_visualization, generate_csv, save_results, generate_summary

seed = 42


# fix random seed for reproducibility
def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] =str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def agent_test(env, agent):
    env.multi_reward = False
    env.multi_log = False
    state, info = env.reset()
    episode_reward = []
    # one episode
    while True:
        with torch.no_grad():
            action = agent.make_action(state)
        if type(agent).__name__ == 'HeuristicAgent':
            state, reward, done, info = env.step(action)
        else:
            state, reward, done, info = env.step(np.squeeze(agent.scale_action(np.expand_dims(action,0),np.array(env.action_space)),0))
        episode_reward.append(reward) 
        if done:
            # print(sum(episode_reward)/len(episode_reward))
            print("test log: ", env.log_name,np.array(env.evaluations[1:])[:,2].mean(), len(env.drift_moments)-1)
            drift_visualization(env=env, agent_name=type(agent).__name__)
            save_results(env=env, agent_name=type(agent).__name__)
            generate_summary(agent_name=type(agent).__name__, log_name=log_name)
            break
    env.multi_reward = True
    env.multi_log = True

logs = [
    ('BPIC2013Incidents', 800),
    # ('BPIC2020DomesticDeclarations', 1000),
    # ('BPIC2020InternationalDeclarations', 600),
    # ('BPIC2020PermitLog', 700),
    # ('BPIC2020PrepaidTravelCost', 200),
    # ('BPIC2020RequestForPayment', 700),
]
memory_size_settings = [
                        # 10,
                        # 50, 
                        # 100, 
                        200, # default
                        # 300,
                        # 400,
                        # 500,
                        # 600,
                        # 700,
                        # 800,
                        # 900,
                        # 1000
                        ]  # 遗忘窗口长度
sampling_rate_settings = [
                        # 0.1,
                        # 0.2, 
                        # 0.3,
                        # 0.4, 
                        # 0.5,
                        # 0.6, 
                        # 0.7,
                        0.8, # default
                        # 0.9,
                        # 1.0
                        ]  # 频率采样的阈值
orders = [
    Order.FRQ,
    # Order.MAX,
    # Order.MIN
]  # 获得最具代表性轨迹的采样方式
algos = [
    Algo.IND,
    # Algo.ILP
]  # 静态流程发现方法
top_settings = [None]  # 选择几个最具代表性的轨迹
filtering_settings = [
    True,
    # False
]  # 流程发现方法是否采样预处理过滤
frequency_settings = [
    True,
    # False
]  # 输入流程发现的日志是否有带频数的重复trace
update_settings = [
    True,
    # False
]  # 是否动态重启流程发现
update_param_settings = [
    True,
    # False
]  # 是否动态更新参数

# env 可调超参数
max_memory_size = 500 # default 500
min_memory_size = 10
max_sampling_rate = 1.0
min_sampling_rate = 0.1
history_window_settings = [
    # 1,
    # 5,
    10, # default
    # 20, 
    # 40
]  # 指标变化检测窗口
observation_window_settings = [
    # 1,
    # 5,
    10, # default
    # 20,
    # 50,
    # 100
] # 每次step要算未来n个指标的均值
drift_punish = 1  # 对变动参数导致发生漂移的惩罚
memory_size_punish = 0.001 # 对遗忘窗口大小的惩罚，即考虑内存
reward_value = 'absolute' # reward是采用f值的绝对值还是相对值 'relative' / 'absolute' 

# agent 可调超参数
learning_rate=1e-4  # 1e-3的reward曲线波动比较大
replay_buffer_size = 1000000
batch_size = 1024
total_episodes = 200
gru_type = gru_type # state的处理方式是直接输出gru还是分成f值序列和动作序列分别输入gru 'full' / 'fusion'


fix_seed(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using {device} device')


for (log_name, cut), order, algo, top, filtering, frequency, update, update_param, memory_size, sampling_rate, history_window, observation_window in \
        product(logs, orders, algos, top_settings, filtering_settings, frequency_settings, update_settings, update_param_settings, 
                memory_size_settings, sampling_rate_settings, history_window_settings, observation_window_settings):
    # generate_csv(log_name)
    env = ProcessDiscoveryEnvironment(log_name, cut, algo, order, top, filtering, frequency, update, update_param, memory_size, sampling_rate, 
                                    max_memory_size, min_memory_size, max_sampling_rate, min_sampling_rate, history_window, observation_window,
                                    drift_punish, memory_size_punish, reward_value)

    print(
        f'log:{log_name},cut:{cut},algo:{algo.name},order:{order.name},top:{top},filtering:{filtering},frequency:{frequency},update:{update},update_param:{update_param},memory_size:{memory_size},sampling_rate:{sampling_rate},history_window:{history_window},observation_window:{observation_window}')
    start_time=time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())) 
    with open('experiment_record.txt','a+') as file:
        file.write(f'start_time:{start_time},log:{log_name},cut:{cut},algo:{algo.name},order:{order.name},top:{top},filtering:{filtering},frequency:{frequency},update:{update},update_param:{update_param}' +'\n' +
                    f'init_memory_size:{memory_size},init_sampling_rate:{sampling_rate},max_memory_size:{max_memory_size},min_memory_size:{min_memory_size},max_sampling_rate:{max_sampling_rate},min_sampling_rate:{min_sampling_rate}' +'\n' +
                    f'history_window:{history_window},observation_window:{observation_window},drift_punish:{drift_punish},memory_size_punish:{memory_size_punish},reward_value:{reward_value},multi_log_count:{len(env.log_list)}'+'\n' +
                    f'learning_rate:{learning_rate},replay_buffer_size:{replay_buffer_size},batch_size:{batch_size},total_episodes:{total_episodes},gru_type:{gru_type}'+'\n\n'
                    )

    heuristic_agent = HeuristicAgent(env=env, device=device, state_dim=env.state_dim, action_dim=env.action_dim)
    agent_test(env, heuristic_agent)

    # td3_agent = TD3Agent(env=env, device=device, state_dim=env.state_dim, action_dim=env.action_dim, learning_rate=learning_rate, start_time=start_time,
    #                         replay_buffer_size=replay_buffer_size, batch_size=batch_size, total_episodes=total_episodes)
    # td3_agent.train()
    # # td3_agent.load('model_dir/'+start_time+'/td3_agent_best_model.pth')
    # agent_test(env, td3_agent)

    # results={'evaluation':env.evaluations,'memory_size_list':env.memory_size_list,'sampling_rate_list':env.sampling_rate_list,'drift_flag':env.drift_flag}
    # with open("fixed_hyperparameters_results.pickle", "wb") as file:
    #     pickle.dump(results, file)

    # 命令行中用如下命令执行进行耗时分析
    # pyinstrument --outfile=time_profile.html -r html main.py