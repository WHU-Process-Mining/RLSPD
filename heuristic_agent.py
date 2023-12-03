import numpy as np

from base_agent import BaseAgent


class HeuristicAgent(BaseAgent):
    def __init__(self, env, device, state_dim, action_dim):
        super(HeuristicAgent, self).__init__(env, device, state_dim, action_dim)
        self.update_rate = 0.1
        self.min_memory_size = env.min_memory_size
        self.max_memory_size = env.max_memory_size
        self.min_sampling_rate = env.min_sampling_rate
        self.max_sampling_rate = env.max_sampling_rate
        self.last_memory_size = env.init_memory_size
        self.last_sampling_rate = env.init_sampling_rate

    def train(self):
        pass

    def make_action(self, state):
        """
        自动改参数的机制
        查看memory窗口内的评估指标，f值在上升则不变
        f值下降时，如因fitness下降则提高memory size，提高sample rate；如因precision下降则降低memory size，降低sample rate。
        # 都下降 将调整回退
        """
        recent_evaluation = state.reshape(-1, 5)[:,0:3]  # 时序数据
        memory_size = state[-2] * (self.max_memory_size - self.min_memory_size) + self.min_memory_size
        sampling_rate = state[-1] * (self.max_sampling_rate - self.min_sampling_rate) + self.min_sampling_rate
        
        # 指标太少时不更新参数
        if len(recent_evaluation) < 5:
            return int(memory_size), sampling_rate

        index = [i for i in range(len(recent_evaluation))]
        recent_f_measure = [evaluation[2] for evaluation in recent_evaluation]

        # f值在上升则不更新参数
        if np.polyfit(x=index, y=recent_f_measure, deg=1)[0] > 0:  # line[0]是斜率k，line[1]是截距b
            return int(memory_size), sampling_rate

        recent_fitness = [evaluation[0] for evaluation in recent_evaluation]
        recent_precision = [evaluation[1] for evaluation in recent_evaluation]
        fitness_line = np.polyfit(x=index, y=recent_fitness, deg=1)
        precision_line = np.polyfit(x=index, y=recent_precision, deg=1)

        update_memory_size = int(memory_size * self.update_rate) if int(
            memory_size * self.update_rate) > 20 else 20
        update_sample_rate = sampling_rate * self.update_rate if sampling_rate * self.update_rate > 0.05 else 0.05
        # precision下降则降低memory size，降低sample rate
        if precision_line[0] < 0 <= fitness_line[0]:
            self.last_memory_size = int(memory_size)
            self.last_sampling_rate = sampling_rate
            memory_size -= update_memory_size
            sampling_rate -= update_sample_rate
        # fitness下降则提高memory size，提高sample rate
        elif fitness_line[0] < 0 <= precision_line[0]:
            self.last_memory_size = int(memory_size)
            self.last_sampling_rate = sampling_rate
            memory_size += update_memory_size
            sampling_rate += update_sample_rate
        # 都下降 将调整回退
        else:
            self.last_memory_size, memory_size = int(memory_size), self.last_memory_size
            self.last_sampling_rate, sampling_rate = sampling_rate, self.last_sampling_rate

        memory_size = np.clip(a=memory_size, a_min=self.min_memory_size, a_max=self.max_memory_size)
        sampling_rate = np.clip(a=sampling_rate, a_min=self.min_sampling_rate, a_max=self.max_sampling_rate)
        return int(memory_size), sampling_rate
