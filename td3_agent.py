import copy
import random
from collections import namedtuple, deque
from tqdm import tqdm
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn, optim
import os
from base_agent import BaseAgent

gru_type = 'fusion' # state的处理方式是直接输出gru还是分成f值序列和动作序列分别输入gru 'full' / 'fusion'

# 强化学习的四元组
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


# DQN的经验存储区
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, state, action, next_state, reward):
        """Save a transition"""
        self.memory.append(Transition(state, action, next_state, reward))

    # 从经验池中随机采样
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# actor的行动网络，根据state返回采取各个action的值
class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, action_range):
        super(ActorNetwork, self).__init__()
        self.action_range = action_range
        self.hidden_size = 128

        self.gru=nn.GRU(input_size=5, hidden_size=self.hidden_size, num_layers=1, batch_first=True)
        # 拆成动作序列和f值序列
        self.gru_fmeasure = nn.GRU(input_size=3, hidden_size=self.hidden_size, num_layers=1, batch_first=True)
        self.gru_action = nn.GRU(input_size=2, hidden_size=self.hidden_size, num_layers=1, batch_first=True)
        self.gru_linear_fusion = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LeakyReLU() # 加这层激活，学习曲线更平滑些
        )

        self.linear_active_stack = nn.Sequential(
            # nn.Linear(state_dim, self.hidden_size),
            # nn.LeakyReLU(),
            # nn.Linear(self.hidden_size, self.hidden_size),
            # nn.LeakyReLU(),
            nn.Linear(self.hidden_size, action_dim),
            nn.Tanh()  # 放缩到(-1,1)
        )
        # 使用He初始化对参数进行初始化
        # for module in self.linear_active_stack.modules():
        #     if isinstance(module, nn.Linear):
        #         nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='leaky_relu')
        #         nn.init.zeros_(module.bias)
    
    def forward(self, state):
        if gru_type == 'full':
            shared_features = self.gru(state.reshape(len(state),-1,5))[1][-1,:,:]
        elif gru_type == 'fusion':
            shared_features = self.gru_linear_fusion(
                torch.cat(
                    [
                    # self.gru(state.reshape(len(state),-1,5))[1][-1,:,:],
                    self.gru_fmeasure(state.reshape(len(state),-1,5)[:,:,0:3])[1][-1,:,:],
                    self.gru_action(state.reshape(len(state),-1,5)[:,:,3:5])[1][-1,:,:]]
                    ,1))
        else:
            shared_features = state
        action = self.linear_active_stack(shared_features)
        return action

# critic的Q值估计网络，根据state返回采取action的Q值
class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticNetwork, self).__init__()
        self.hidden_size = 128

        self.gru=nn.GRU(input_size=5, hidden_size=self.hidden_size, num_layers=1, batch_first=True)
        # 拆成动作序列和f值序列
        self.gru_fmeasure = nn.GRU(input_size=3, hidden_size=self.hidden_size, num_layers=1, batch_first=True)
        self.gru_action = nn.GRU(input_size=2, hidden_size=self.hidden_size, num_layers=1, batch_first=True)
        self.gru_linear_fusion = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LeakyReLU() # 加这层激活，学习曲线更平滑些
        )

        self.linear_active_stack = nn.Sequential(
            # nn.Linear(state_dim + action_dim, self.hidden_size),
            # nn.LeakyReLU(),
            # nn.Linear(self.hidden_size, self.hidden_size),
            nn.Linear(self.hidden_size + action_dim, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, 1)
        )
        # 使用He初始化对参数进行初始化
        # for module in self.linear_active_stack.modules():
        #     if isinstance(module, nn.Linear):
        #         nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='leaky_relu')
        #         nn.init.zeros_(module.bias)

    def forward(self, state, action):
        if gru_type == 'full':
            shared_features = self.gru(state.reshape(len(state),-1,5))[1][-1,:,:]
        elif gru_type == 'fusion':
            shared_features = self.gru_linear_fusion(
                torch.cat(
                    [
                    # self.gru(state.reshape(len(state),-1,5))[1][-1,:,:],
                    self.gru_fmeasure(state.reshape(len(state),-1,5)[:,:,0:3])[1][-1,:,:],
                    self.gru_action(state.reshape(len(state),-1,5)[:,:,3:5])[1][-1,:,:]]
                    ,1))
        else:
            shared_features = state

        q_value = self.linear_active_stack(torch.cat([shared_features,action],1))
        return q_value


class TD3Agent(BaseAgent):
    def __init__(self, state_dim, action_dim, learning_rate, env, device, start_time,
                replay_buffer_size, batch_size, total_episodes):
        super(TD3Agent, self).__init__(env, device, state_dim, action_dim)

        # self.action_range = torch.tensor(np.column_stack((self.env.action_space.low, self.env.action_space.high)),
        #                                 device=self.device, requires_grad=False)
        self.action_range = torch.tensor([[-1,1],[-1,1]], device=self.device, requires_grad=False)
        self.env_action_range = torch.tensor(self.env.action_space, device=self.device, requires_grad=False)

        self.actor_network = ActorNetwork(self.state_dim, self.action_dim, self.action_range).to(self.device)
        self.target_actor_network = ActorNetwork(self.state_dim, self.action_dim, self.action_range).to(self.device)
        self.target_actor_network.load_state_dict(self.actor_network.state_dict())
        self.actor_optimizer = optim.AdamW(self.actor_network.parameters(), lr=learning_rate, amsgrad=True)

        self.critic_network_1 = CriticNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_critic_network_1 = CriticNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_critic_network_1.load_state_dict(self.critic_network_1.state_dict())
        self.critic_optimizer_1 = optim.AdamW(self.critic_network_1.parameters(), lr=learning_rate, amsgrad=True)

        self.critic_network_2 = CriticNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_critic_network_2 = CriticNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_critic_network_2.load_state_dict(self.critic_network_2.state_dict())
        self.critic_optimizer_2 = optim.AdamW(self.critic_network_2.parameters(), lr=learning_rate, amsgrad=True)

        self.replay_memory = ReplayMemory(replay_buffer_size) # 太大的话会学到旧的policy产生的数据
        self.batch_size = batch_size  # 每次从ReplayMemory中采样的transition数量
        self.gamma = 0.99  # 累计回报的折扣系数
        self.tau = 0.005  # target_network的更新率

        self.episode_durations = []  # 每个episode的持续步数
        self.total_episodes = total_episodes  # 总共玩的游戏场数
        self.max_step = 10000  # 每个episode最多玩n步就结束，并执行梯度更新

        self.start_time = start_time

        self.exploration_noise_std = 0.1
        self.policy_noise_std = 0.2 # TODO:衰减
        self.noise_clip = 0.4
        self.optimize_count = 0
        self.delay_update_frequency = 5

        self.static_episode = 0
        self.random_episode = 0

        self.best_model = [0,1000,0,None] # [f_measure,drift_count,episode,model]用于保存最好模型

    # 使用actor_network选择动作
    def make_action(self, state):
        action = self.actor_network(torch.tensor(state, dtype = torch.float32, device=self.device).unsqueeze(0))
        return action.squeeze(0).cpu().numpy()

    # 使用actor_network选择动作，并加上噪声进行探索
    def sample_action(self, state):
        raw_action = self.actor_network(state)
        noise = torch.randn(self.action_dim, device=self.device)
        exploration_noise = noise * self.exploration_noise_std * ((self.action_range[:,1]-self.action_range[:,0]) * 0.5)
        action = (raw_action + exploration_noise).clamp(self.action_range[:,0], self.action_range[:,1])
        return action
    
    # action按照取值范围放缩
    def scale_action(self, action, action_range):
        action_scaled = action_range[:, 0] + (action + 1) * 0.5 * (action_range[:, 1] - action_range[:, 0])
        return action_scaled

    
    def train(self):
        self.actor_network.train()  # 训练前，先确保 network 处在 training 模式
        self.critic_network_1.train()  # 训练前，先确保 network 处在 training 模式
        self.critic_network_2.train()  # 训练前，先确保 network 处在 training 模式
        total_rewards,average_rewards, final_rewards = [], [], []  # 每个episode的总reward、平均reward、结束时的reward
        episode_actor_loss,episode_critic_loss=[],[]
        episode_f_measure, episode_drift_count = [], []

        prg_bar = tqdm(range(self.total_episodes))
        for i_episode in prg_bar:
            total_reward = 0
            state, info = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(
                0)  # .unsqueeze(0)作用是升维，将此数据视为一个整体而给予的一个索引，方便后续对数据的批处理。
            self.current_episode_actor_loss = []
            self.current_episode_critic_loss = []
            action_1_list, action_2_list = [], []
            memory_size_list, sampling_rate_list = [], []
            for t in tqdm(range(self.max_step)):
                with torch.no_grad():  # 收集的数据并不马上用于更新网络，故不保留梯度
                    action = self.sample_action(state)
                observation, reward, done, info = self.env.step(self.scale_action(action, self.env_action_range).squeeze(0).cpu().numpy())
                total_reward += reward
                reward = torch.tensor([reward], dtype=torch.float32, device=self.device)
                actor_output = action.squeeze(0).cpu().numpy()
                action_1_list.append(actor_output[0])
                action_2_list.append(actor_output[1])
                memory_size_list.append(self.env.memory_size)
                sampling_rate_list.append(self.env.sampling_rate)
                if done:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

                # 将transition存入经验区
                self.replay_memory.push(state, action, next_state, reward)

                # 移动到下一个state
                state = next_state

                # 进行一次模型优化
                self.optimize_model()

                # if terminated or truncated:
                if done or t==self.max_step-1:
                    self.episode_durations.append(t + 1)
                    final_rewards.append(reward.item())
                    total_rewards.append(total_reward)
                    average_rewards.append(total_reward/(t + 1))
                    episode_f_measure.append(np.array(self.env.evaluations[1:])[:,2].mean())
                    episode_drift_count.append(len(self.env.drift_moments)-1)
                    episode_actor_loss.append(sum(self.current_episode_actor_loss)/len(self.current_episode_actor_loss) if len(self.current_episode_actor_loss)>0 else 0)
                    episode_critic_loss.append(sum(self.current_episode_critic_loss)/len(self.current_episode_critic_loss) if len(self.current_episode_critic_loss)>0 else 0)
                    break
            
            print(f"TD3, Episode: {i_episode}, actor_loss: {episode_actor_loss[-1]: 4.3f}, critic_loss: {episode_critic_loss[-1]: 4.3f}, Total_Reward: {total_rewards[-1]: 4.3f}, Average_Reward: {average_rewards[-1]: 4.3f}, Final_Reward: {final_rewards[-1]: 4.3f}, episode_duration: {self.episode_durations[-1]}, optimize_count: {self.optimize_count}")

            plt.figure()
            plt.subplot(2, 2, 1)
            plt.plot(action_1_list)
            plt.title("action_1_list")
            plt.subplot(2, 2, 2)
            plt.plot(action_2_list)
            plt.title("action_2_list")
            plt.subplot(2, 2, 3)
            plt.plot(memory_size_list)
            plt.title("memory_size_list")
            plt.subplot(2, 2, 4)
            plt.plot(sampling_rate_list)
            plt.title("sample_rate_list")
            plt.tight_layout() # 解决标题重叠问题
            os.makedirs("agent_action/" + self.start_time, exist_ok=True)
            plt.savefig("agent_action/" + self.start_time + "/td3_episode_" + str(len(self.episode_durations)-1) + "_train_" + '%.3f'%(np.array(self.env.evaluations[1:])[:,0].mean()) + "_" + '%.3f'%(np.array(self.env.evaluations[1:])[:,1].mean()) + "_" + '%.3f'%(np.array(self.env.evaluations[1:])[:,2].mean()) + ".png")
            plt.close()

            test_f_measure, test_drift_count = self.agent_test()
            if test_f_measure > self.best_model[0]:
                self.best_model[0] = test_f_measure
                self.best_model[1] = test_drift_count
                self.best_model[2] = i_episode+1
                self.best_model[3] = {
                                        "actor_network": copy.deepcopy(self.actor_network.state_dict()),
                                        "actor_optimizer": copy.deepcopy(self.actor_optimizer.state_dict()),
                                        "critic_network_1": copy.deepcopy(self.critic_network_1.state_dict()),
                                        "critic_optimizer_1": copy.deepcopy(self.critic_optimizer_1.state_dict()),
                                        "critic_network_2": copy.deepcopy(self.critic_network_2.state_dict()),
                                        "critic_optimizer_2": copy.deepcopy(self.critic_optimizer_2.state_dict())
                                    }

            if (i_episode+1)%50 == 0:
                os.makedirs("model_dir/" + self.start_time, exist_ok=True)
                self.save('model_dir/' + self.start_time + '/td3_agent_model_episode_'+ str(i_episode+1) +'.pth')
                plt.figure()
                plt.subplot(2, 3, 1)
                plt.plot(total_rewards)
                plt.title("Total Rewards")
                plt.subplot(2, 3, 2)
                plt.plot(average_rewards)
                plt.title("Average Rewards")
                plt.subplot(2, 3, 3)
                plt.plot(episode_f_measure)
                plt.title("f_measure")
                plt.subplot(2, 3, 4)
                plt.plot(episode_actor_loss)
                plt.title("actor loss")
                plt.subplot(2, 3, 5)
                plt.plot(episode_critic_loss)
                plt.title("critic loss")
                plt.subplot(2, 3, 6)
                plt.plot(episode_drift_count)
                plt.title("drift_count")
                plt.tight_layout() # 解决标题重叠问题
                plt.savefig('model_dir/' + self.start_time + '/td3_performance_episode_'+ str(i_episode+1) +'.png')
                plt.close()
        
        torch.save(self.best_model[3], 'model_dir/' + self.start_time + '/td3_agent_best_model.pth')
        print('save best model_episode_'+ str(self.best_model[2]),'f_measure:',self.best_model[0],'drift_count:',self.best_model[1])

    def optimize_model(self):
        if len(self.replay_memory) < self.batch_size:
            return
        self.optimize_count += 1
        transitions = self.replay_memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for detailed explanation).
        # This converts batch-array of Transitions to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # 获取非结束状态
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)

        # critic_network计算当前状态st采取行动at的Q值即Q(s_t, a_t)
        #TODO：网络给出的action和实际采样占比有差距，导致action略微改变不会使reward改变，从而无法学习
        state_action_values_1 = self.critic_network_1(state_batch, action_batch)
        state_action_values_2 = self.critic_network_2(state_batch, action_batch)
        
        # target_network计算状态st+1采取行动at+1的Q值即Q(s_t+1, a_t+1),终止状态的值为0
        next_state_values_1 = torch.zeros_like(state_action_values_1, device=self.device)
        next_state_values_2 = torch.zeros_like(state_action_values_2, device=self.device)
        with torch.no_grad():
            # 在target action上添加噪声
            next_action_batch = self.target_actor_network(non_final_next_states)
            policy_noise = (torch.randn_like(next_action_batch) * self.policy_noise_std).clamp(-self.noise_clip, self.noise_clip) # randn_like为标准正态分布N(0,1)
            smoothed_next_action_batch = (next_action_batch + policy_noise * ((self.action_range[:,1]-self.action_range[:,0]) * 0.5)).clamp(self.action_range[:,0], self.action_range[:,1])
            next_state_values_1[non_final_mask] = self.target_critic_network_1(non_final_next_states, smoothed_next_action_batch)
            next_state_values_2[non_final_mask] = self.target_critic_network_2(non_final_next_states, smoothed_next_action_batch)

            # 计算状态st期待的Q值即r_t+γ*Q(s_t+1, a_t+1)
            target_value = torch.min(next_state_values_1,next_state_values_2)
            expected_state_action_values = target_value * self.gamma + reward_batch.unsqueeze(1)

        # Compute critic loss
        critic_loss_1 = nn.functional.mse_loss(state_action_values_1, expected_state_action_values)
        critic_loss_2 = nn.functional.mse_loss(state_action_values_2, expected_state_action_values)
        self.current_episode_critic_loss.append(critic_loss_1.item()+critic_loss_2.item())
        # Optimize the critic
        self.critic_optimizer_1.zero_grad()
        critic_loss_1.backward()
        self.critic_optimizer_1.step()
        self.critic_optimizer_2.zero_grad()
        critic_loss_2.backward()
        self.critic_optimizer_2.step()

        if(self.optimize_count % self.delay_update_frequency == 0):
            # compute actor loss
            actor_loss = - self.critic_network_1(state_batch,self.actor_network(state_batch)).mean()
            self.current_episode_actor_loss.append(actor_loss.item())
            # optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # 平滑更新target network：θ′ ← τ θ + (1 −τ )θ′
            for param, target_param in zip(self.actor_network.parameters(), self.target_actor_network.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic_network_1.parameters(), self.target_critic_network_1.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic_network_2.parameters(), self.target_critic_network_2.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    # 在环境中测试agent的表现
    def test(self, total_episodes=5):
        self.actor_network.eval()
        self.critic_network_1.eval()
        self.critic_network_2.eval()
        with torch.no_grad():
            super().test()
    
    def agent_test(self):
        self.actor_network.eval()
        self.critic_network_1.eval()
        self.critic_network_2.eval()
        self.env.multi_reward = False
        self.env.multi_log = False
        state, info = self.env.reset()
        episode_reward = []
        action_1_list, action_2_list = [], []
        memory_size_list, sampling_rate_list = [], []
        # 进行一场游戏
        while True:
            with torch.no_grad():
                action = self.make_action(state)
            # state, reward, done, info = self.env.step(action)
            state, reward, done, info = self.env.step(np.squeeze(self.scale_action(np.expand_dims(action,0),np.array(self.env.action_space)),0))
            action_1_list.append(action[0])
            action_2_list.append(action[1])
            memory_size_list.append(self.env.memory_size)
            sampling_rate_list.append(self.env.sampling_rate)
            episode_reward.append(reward) 
            if done:
                print("test log: ", self.env.log_name,np.array(self.env.evaluations[1:])[:,2].mean(), len(self.env.drift_moments)-1)
                break
        self.env.multi_reward = True
        self.env.multi_log = True
        plt.figure()
        plt.subplot(2, 2, 1)
        plt.plot(action_1_list)
        plt.title("action_1_list")
        plt.subplot(2, 2, 2)
        plt.plot(action_2_list)
        plt.title("action_2_list")
        plt.subplot(2, 2, 3)
        plt.plot(memory_size_list)
        plt.title("memory_size_list")
        plt.subplot(2, 2, 4)
        plt.plot(sampling_rate_list)
        plt.title("sample_rate_list")
        plt.tight_layout() # 解决标题重叠问题
        os.makedirs("agent_action/" + self.start_time, exist_ok=True)
        plt.savefig("agent_action/" + self.start_time + "/td3_episode_" + str(len(self.episode_durations)-1) + "_test_" + '%.3f'%(np.array(self.env.evaluations[1:])[:,0].mean()) + "_" + '%.3f'%(np.array(self.env.evaluations[1:])[:,1].mean()) + "_" + '%.3f'%(np.array(self.env.evaluations[1:])[:,2].mean()) + ".png")
        plt.close()

        plt.figure()
        plt.subplot(5, 1, 1)
        plt.plot(np.array(self.env.evaluations[1:])[:,0])
        plt.vlines(x= np.where(np.array(self.env.drift_flag)==1), ymin=0.5, ymax=1, linestyles='dashed', colors='red')
        plt.title("Fitness")
        plt.subplot(5, 1, 2)
        plt.plot(np.array(self.env.evaluations[1:])[:,1])
        plt.vlines(x= np.where(np.array(self.env.drift_flag)==1), ymin=0.5, ymax=1, linestyles='dashed', colors='red')
        plt.title("Precision")
        plt.subplot(5, 1, 3)
        plt.plot(np.array(self.env.evaluations[1:])[:,2])
        plt.vlines(x= np.where(np.array(self.env.drift_flag)==1), ymin=0.5, ymax=1, linestyles='dashed', colors='red')
        plt.title("F_measure")
        plt.subplot(5, 1, 4)
        plt.plot(self.env.memory_size_list)
        plt.vlines(x= np.where(np.array(self.env.drift_flag)==1), ymin=250, ymax=500, linestyles='dashed', colors='red')
        plt.title("Memory_size")
        plt.subplot(5, 1, 5)
        plt.plot(self.env.sampling_rate_list)
        plt.vlines(x= np.where(np.array(self.env.drift_flag)==1), ymin=0.5, ymax=1, linestyles='dashed', colors='red')
        plt.title("Sampling_rate")
        plt.tight_layout() # 解决标题重叠问题
        os.makedirs("agent_action/" + self.start_time, exist_ok=True)
        plt.savefig("agent_action/" + self.start_time + "/td3_episode_" + str(len(self.episode_durations)-1) + "_test_drift_visualization" + ".png")
        plt.close()
        
        # 避免后续还需训练
        self.actor_network.train()  # 训练前，先确保 network 处在 training 模式
        self.critic_network_1.train()  # 训练前，先确保 network 处在 training 模式
        self.critic_network_2.train()  # 训练前，先确保 network 处在 training 模式

        return np.array(self.env.evaluations[1:])[:,2].mean(), len(self.env.drift_moments)-1

    def save(self, model_path):
        agent_dict = {
            "actor_network": self.actor_network.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_network_1": self.critic_network_1.state_dict(),
            "critic_optimizer_1": self.critic_optimizer_1.state_dict(),
            "critic_network_2": self.critic_network_2.state_dict(),
            "critic_optimizer_2": self.critic_optimizer_2.state_dict()
        }
        torch.save(agent_dict, model_path)
        print('saving model: ', model_path)

    def load(self, model_path):
        checkpoint = torch.load(model_path)
        self.actor_network.load_state_dict(checkpoint["actor_network"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_network_1.load_state_dict(checkpoint["critic_network_1"])
        self.critic_optimizer_1.load_state_dict(checkpoint["critic_optimizer_1"])
        self.critic_network_2.load_state_dict(checkpoint["critic_network_2"])
        self.critic_optimizer_2.load_state_dict(checkpoint["critic_optimizer_2"])
        print('loading model: ', model_path)
