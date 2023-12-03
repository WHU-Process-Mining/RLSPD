from collections import Counter, deque
from os import path
from time import process_time
import random
import numpy as np
from anytree import Node
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.discovery.ilp import algorithm as ilp_miner
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.algo.evaluation.replay_fitness import algorithm as fitness_evaluator
from pm4py.objects.conversion.process_tree import converter as pt_converter
from pm4py.objects.log.obj import EventLog, Trace
from pm4py.streaming.importer.csv import importer as csv_stream_importer

from utils import ACTIVITY_KEY, Order, Algo, CASE_ID_KEY, FINAL_ACTIVITY, get_trace, pruned_tree, check_exist_child


class ProcessDiscoveryEnvironment:
    def __init__(self, log_name, cut, algo, order, top, filtering, frequency, update, update_param, memory_size, sampling_rate,
                 max_memory_size, min_memory_size, max_sampling_rate, min_sampling_rate, history_window, observation_window,
                 drift_punish, memory_size_punish, reward_value):
        self.log_name = log_name  # 包含事件流的CSV文件的名称
        self.cut = cut  # 进行初始化流程模型的轨迹数，一般为日志长度的10%
        self.algo = algo  # 静态流程发现算法
        self.order = order  # 轨迹变体排序标准
        self.top = top  # 模型构建中要使用的轨迹变体数（Pareto分布为None）
        self.filtering = filtering  # 指示流程发现算法是否使用filtering预处理的布尔值
        self.frequency = frequency  # 指示输入流程发现的日志是否按频率包含相同trace的布尔值
        self.update = update  # 是否进行动态流程发现
        self.update_param = update_param  # 是否动态更新参数

        self.history_window = history_window  # 指标变化检测窗口

        self.memory_size = memory_size  # 遗忘窗口长度
        self.init_memory_size = memory_size
        self.max_memory_size = max_memory_size
        self.min_memory_size = min_memory_size

        self.sampling_rate = sampling_rate  # 频率采样的阈值
        self.init_sampling_rate = sampling_rate
        self.max_sampling_rate = max_sampling_rate
        self.min_sampling_rate = min_sampling_rate

        self.event_stream = None  # 日志对应的事件流
        self.start_time = None  # 记录处理一条trace的时间
        self.root = None  # root树保存了所有未完成case的活动路径，用树使节点可以共用节省内存
        self.i = 0  # root树的节点编号
        self.hashtable = {}  # 保存了未完成的case的尾节点

        self.processed_traces = 0  # 日志流中已处理的trace数量
        # 最近窗口内的trace ,+1避免当窗口为最大时取未来trace会遗忘掉第一条
        self.trace_queue = deque(maxlen=self.max_memory_size + 1)
        self.recent_variants = Counter()  # 最近窗口内的轨迹变体

        self.best_variants = None  # 当前最具代表性的轨迹
        self.drift_moments = []  # 发生概念漂移的轨迹下标位置
        self.drift_variants = []  # 每次概念漂移的轨迹变体集
        self.models = []  # 每次概念漂移重新发现的模型
        self.evaluations = []  # 评估指标

        self.step_count = 0  # 动态更新参数的步数统计

        self.action_dim = 2  # 要动态更新的参数个数
        self.action_space = [[self.min_memory_size, self.max_memory_size],
                             [self.min_sampling_rate, self.max_sampling_rate]
                             ]
        self.used_evaluation_type = 3  # 状态中使用的评估指标种类 有fitness、precision、F_value
        self.state_dim = self.history_window * \
            (self.used_evaluation_type + self.action_dim)  # 窗口内每个时间步的指标和参数

        self.multi_reward = True
        self.observation_window = observation_window  # 每次step要算未来n个指标的均值
        self.upcoming_traces = deque(
            maxlen=self.observation_window)  # 未来用于评估的trace
        self.step_observation_window = True  # 每次step走一个窗口的traces

        self.reward_baseline = 0  # reward减去一个基准值
        self.reward_value = reward_value  # reward是采用f值的绝对值还是相对值 'relative' / 'absolute'
        self.reward_list = []  # 每步step的reward

        self.sampling_info = []  # 实际采样占比和轨迹变体数
        self.memory_size_list = []
        self.sampling_rate_list = []
        self.drift_flag = []
        self.drift_punish = drift_punish  # 对变动参数导致发生漂移的惩罚
        self.memory_size_punish = memory_size_punish  # 对遗忘窗口大小的惩罚，即考虑内存
        self.log_list = [
            ('BPIC2013Incidents', 800),
            # ('BPIC2020DomesticDeclarations', 1000),
            # ('BPIC2020InternationalDeclarations', 600),
            # ('BPIC2020PermitLog', 700),
            # ('BPIC2020PrepaidTravelCost', 200),
            # ('BPIC2020RequestForPayment', 700),
        ]

        self.multi_log = True

        self.init_with_memory_window = True

    def reset(self):
        """
        :return: init_state
        """
        # TODO:reset尝试每个日志只用前半部分训练
        self.event_stream = csv_stream_importer.apply(
            path.join('eventlog', 'CSV', self.log_name + '.csv'))
        self.init_cut = self.cut

        if self.multi_log:  # 换日志
            random_log = random.choice(self.log_list)
            self.event_stream = csv_stream_importer.apply(
                path.join('eventlog', 'CSV', random_log[0] + '.csv'))
            self.init_cut = random_log[1]
            print("train log: ", random_log[0])

        # root树保存了所有未完成case的活动路径，用树使节点可以共用节省内存
        self.root = Node(id='root', name='root', parent=None, case_id=[])
        self.i = 0  # root树的节点编号
        self.hashtable = {}  # 保存了未完成的case的尾节点

        self.processed_traces = 0  # 日志流中已处理的trace数量
        # 最近窗口内的trace ,+1避免当窗口为最大时取未来trace会遗忘掉第一条
        self.trace_queue = deque(maxlen=self.max_memory_size + 1)
        self.recent_variants = Counter()  # 最近窗口内的轨迹变体

        self.best_variants = None  # 当前最具代表性的轨迹
        self.drift_moments = []  # 发生概念漂移的轨迹下标位置
        self.drift_variants = []  # 每次概念漂移的轨迹变体集
        self.models = []  # 每次概念漂移重新发现的模型
        self.evaluations = []  # 评估指标
        self.upcoming_traces = deque(
            maxlen=self.observation_window)  # 未来用于评估的trace

        self.memory_size = self.init_memory_size
        self.sampling_rate = self.init_sampling_rate
        self.step_count = 0

        self.reward_list = []  # 每步step的reward

        self.sampling_info = []  # 实际采样占比和轨迹变体数
        self.memory_size_list = []
        self.sampling_rate_list = []
        self.drift_flag = []

        self.start_time = process_time()
        print('Processing event stream...')

        init_trace_num = self.init_memory_size if self.init_with_memory_window else self.init_cut
        for _ in range(init_trace_num):  # 准备若干条trace初始化流程模型
            for event in self.event_stream:
                self.process_event_with_tree(event)
                if event[ACTIVITY_KEY] == FINAL_ACTIVITY:
                    break
        self.select_best_variants()  # 基于频率采样得到高频轨迹变体集
        self.learn_model()  # 使用self.best_variants作为日志调用静态流程发现方法
        end = process_time()
        self.evaluations.append([None, None, None, end - self.start_time])
        self.start_time = end

        for _ in range(self.history_window):  # 避免state出现空的0值
            if self.update:
                self.select_best_variants()
                if self.best_variants.keys() != self.drift_variants[-1].keys():
                    self.learn_model()
                    self.drift_flag.append(1)
                else:
                    self.drift_flag.append(0)

            # 每次step走一个窗口的traces
            for _ in range(self.observation_window):
                for event in self.event_stream:
                    self.process_event_with_tree(event)
                    if event[ACTIVITY_KEY] == FINAL_ACTIVITY:
                        break
            self.evaluations.append(self.evaluate_model_with_traces(
                *self.models[-1], list(self.trace_queue)[-self.observation_window:]))

            self.memory_size_list.append(self.memory_size)
            self.sampling_rate_list.append(self.sampling_rate)
            end = process_time()
            self.evaluations[-1].append(end - self.start_time)
            self.start_time = end

        for _ in range(self.observation_window):  # 提前准备未来若干条trace做评估
            for event in self.event_stream:
                self.process_event_with_tree(event)
                if event[ACTIVITY_KEY] == FINAL_ACTIVITY:
                    self.upcoming_traces.append(self.trace_queue.pop())
                    break

        state = self.get_state()
        info = self.get_info()
        return state, info

    # update_param -> dynamic_discovery -> new trace -> eval ，然后reward为eval的指标
    def step(self, action):
        """
        :return: state,reword,done,info
        """
        if self.update_param:
            self.update_parameters(action)
        if self.update:
            self.select_best_variants()
            if self.best_variants.keys() != self.drift_variants[-1].keys():
                self.learn_model()
                self.drift_flag.append(1)
            else:
                self.drift_flag.append(0)

        self.memory_size_list.append(self.memory_size)
        self.sampling_rate_list.append(self.sampling_rate)

        self.evaluations.append(self.evaluate_model_with_traces(
            *self.models[-1], self.upcoming_traces))
        self.trace_queue.extend(self.upcoming_traces)
        self.upcoming_traces = deque(maxlen=self.observation_window)

        # 每次step走一个窗口的traces
        for _ in range(self.observation_window):
            for event in self.event_stream:
                self.process_event_with_tree(event)
                if event[ACTIVITY_KEY] == FINAL_ACTIVITY:
                    self.upcoming_traces.append(self.trace_queue.pop())
                    break

        end = process_time()
        self.evaluations[-1].append(end - self.start_time)
        self.start_time = end

        self.step_count += 1
        state = self.get_state()
        reward = self.get_reward()
        done = not self.event_stream.reading_log
        info = self.get_info()
        return state, reward, done, info

    def process_event_with_tree(self, event):
        case = event[CASE_ID_KEY]
        activity = event[ACTIVITY_KEY]
        check_case_id = self.hashtable.get(case)

        if activity == FINAL_ACTIVITY:
            # 在树中从check_case_id向上取出完整trace
            new_trace = tuple(get_trace(check_case_id))
            self.hashtable.pop(str(case))  # 将case从hashtable中取出
            pruned_tree(check_case_id, self.hashtable)  # 在树中删除仅被该case使用的节点

            self.processed_traces += 1

            # 用有序定长队列记录轨迹
            self.trace_queue.append(new_trace)

        else:
            if check_case_id is None:  # 当前event为所属case的第一个活动
                child = check_exist_child(self.root.children, activity)
                if child is None:  # 当前activity没在树的第一层出现过
                    nodo = Node(name=self.i, id=activity, parent=self.root)
                    self.i = self.i + 1
                else:
                    nodo = child
            else:
                x = self.hashtable.get(case)
                child = check_exist_child(x.children, activity)
                if child is not None:  # 当前event的所属case目前的活动轨迹在树中存在
                    nodo = child
                else:
                    father = x.id
                    grandfather = x.parent.id
                    if father == activity and grandfather != activity:  # 当前activity与父节点相同，第一次自循环时新建第二个节点
                        nodo = Node(name=self.i, id=activity, parent=x)
                        self.i = self.i + 1
                    elif father == activity and grandfather == activity:  # 当前activity与父节点相同，自循环时用两个相同节点表示
                        nodo = x
                    else:  # 当前activity与父节点不同
                        nodo = Node(name=self.i, id=activity, parent=x)
                        self.i = self.i + 1
            self.hashtable[case] = nodo

    def select_best_variants(self):
        """
        根据所选顺序标准确定当前时刻最显著的轨迹变体
        """
        # 获取遗忘窗口内的轨迹变体
        self.recent_variants = Counter(list(self.trace_queue)[-self.memory_size:]) if len(
            self.trace_queue) > self.memory_size else Counter(self.trace_queue)

        top_variants = self.top  # 要选几个最有代表性的轨迹变体，self.top=None说明要根据高频采样率来选
        if top_variants is None:
            counter = 0
            top_variants = 0
            total = sum(self.recent_variants.values())  # 当前轨迹变体集的总基数
            variant_frequency_list = sorted(
                self.recent_variants.values(), reverse=True)
            while counter / total < self.sampling_rate:  # 按频数降序，依次挑选直到挑出的轨迹变体占比大于采样率
                counter += variant_frequency_list[top_variants]
                top_variants += 1
            self.sampling_info.append(
                (counter / total, top_variants, variant_frequency_list))

        if self.order == Order.FRQ:
            self.best_variants = {
                item[0]: item[1] for item in self.recent_variants.most_common(top_variants)}
        else:
            candidate_variants = list(
                item[0] for item in self.recent_variants.most_common())
            candidate_variants.sort(key=lambda v: len(
                v), reverse=self.order == Order.MAX)
            self.best_variants = {
                var: self.recent_variants[var] for var in candidate_variants[:top_variants]}

    def learn_model(self):
        """
        使用当前时刻最重要的变量生成流程模型
        """
        log = EventLog()
        for variant, occurrence in self.best_variants.items():  # 将高频轨迹变体集转为日志作为流程发现算法的输入，按self.frequency区分是否加入了频数
            for i in range(occurrence if self.frequency else 1):
                log.append(Trace({ACTIVITY_KEY: activity}
                           for activity in variant))
        if self.algo == Algo.IND:
            variant = inductive_miner.Variants.IMf if self.filtering else inductive_miner.Variants.IM
            process_tree = inductive_miner.apply(log, variant=variant)
            model = pt_converter.apply(process_tree)
        elif self.algo == Algo.ILP:
            variant = ilp_miner.Variants.CLASSIC
            model = ilp_miner.apply(log, variant=variant, parameters={
                                    variant.value.Parameters.SHOW_PROGRESS_BAR: False})
        self.models.append(model)
        # TODO: 减去future_window中的trace
        self.drift_moments.append(self.processed_traces)
        self.drift_variants.append(self.best_variants)  # 保存每次概念漂移时的高频活动轨迹变体集

    def evaluate_model_with_traces(self, petri_net, initial_marking, final_marking, traces):
        """
        根据输入中提供的traces评估流程模型
        :param traces: 要在评估中使用的流程实例traces
        """
        log = EventLog([Trace({ACTIVITY_KEY: activity}
                       for activity in trace) for trace in traces])

        # 计算precision和fitness，基于TOKEN的方法会比基于ALIGNMENT的方法快很多
        variant = fitness_evaluator.Variants.TOKEN_BASED
        parameters = {variant.value.Parameters.SHOW_PROGRESS_BAR: False}
        fitness = fitness_evaluator.apply(log, petri_net, initial_marking, final_marking,
                                          variant=variant, parameters=parameters)['average_trace_fitness']
        variant = precision_evaluator.Variants.ETCONFORMANCE_TOKEN
        parameters = {variant.value.Parameters.SHOW_PROGRESS_BAR: False}
        precision = precision_evaluator.apply(
            log, petri_net, initial_marking, final_marking, variant=variant, parameters=parameters)
        f_measure = 2 * fitness * precision / \
            (fitness + precision) if fitness != 0 else 0

        return [fitness, precision, f_measure]

    def get_state(self):
        evaluation_state = np.array(self.evaluations[-self.history_window:])[
            :, 0:self.used_evaluation_type]  # reset中确保了recent_evaluation不为空
        memory_size_state = (np.array(self.memory_size_list[-self.history_window:]).reshape(-1, 1) - self.min_memory_size)/(
            self.max_memory_size - self.min_memory_size)  # 放缩到(0,1)，reshape(-1, 1)扩充一维好以便拼接
        sampling_rate_state = (np.array(self.sampling_rate_list[-self.history_window:]).reshape(
            -1, 1) - self.min_sampling_rate)/(self.max_sampling_rate - self.min_sampling_rate)

        state = np.concatenate((evaluation_state,
                                memory_size_state,
                                sampling_rate_state
                                ), axis=1).flatten()  # 按时间步水平拼接再平铺

        return state

    def get_info(self):
        return {'memory_size': self.memory_size,
                'sampling_rate': self.sampling_rate,
                'sampling_percentage': self.sampling_info[-1][0],
                'top_variant': self.sampling_info[-1][1],
                'variant_count': self.sampling_info[-1][2],
                }

    # TODO：设计更合理的reward
    def get_reward(self):
        # 每次step一个窗口，reward为这个窗口traces的F值
        self.reward_list.append(self.evaluations[-1][2]
                                # 引入对更新次数的惩罚（辅助学习memory_size）
                                - (self.drift_punish * self.drift_flag[-1] if len(self.drift_flag) > 0 else 0)
                                - self.memory_size_punish * \
                                self.memory_size_list[-1]  # 对遗忘窗口大小的惩罚，即考虑内存
                                - self.reward_baseline
                                )
        if self.reward_value == 'absolute':
            return self.reward_list[-1]  # reward为评估值的绝对值
        elif self.reward_value == 'relative':
            # reward为评估值的差值
            return self.reward_list[-1] - self.reward_list[-2] if (len(self.reward_list) > 1) else 0

    def update_parameters(self, action):
        self.memory_size = round(
            np.clip(a=action[0], a_min=self.min_memory_size, a_max=self.max_memory_size))
        self.sampling_rate = float(np.clip(
            a=action[1], a_min=self.min_sampling_rate, a_max=self.max_sampling_rate))
