import pickle
from datetime import timedelta
from enum import Enum
from os import path, makedirs, listdir
from matplotlib import pyplot as plt
import numpy as np

from pandas import DataFrame, read_csv
from pm4py import PetriNet
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.conversion.log import converter as log_converter


class Algo(Enum):
    IND = 1
    ILP = 2


class Order(Enum):
    FRQ = 1
    MIN = 2
    MAX = 3


CASE_ID_KEY = 'case:concept:name'
ACTIVITY_KEY = 'concept:name'
TIMESTAMP_KEY = 'time:timestamp'
FINAL_ACTIVITY = '_END_'


def generate_csv(log_name, case_id=CASE_ID_KEY, activity=ACTIVITY_KEY, timestamp=TIMESTAMP_KEY):
    """
    将输入的XES文件转换为CSV格式按时间顺序排序的事件流，每个trace通过定义术语的最终事件进行扩展
    :param log_name: 包含事件日志的XES文件（可能已压缩）的名称
    :param case_id: 流程实例标识符属性（前缀为“case:”）
    :param activity: 标识已执行活动的属性
    :param timestamp: 指示事件执行时刻的属性
    :return:
    """
    csv_path = path.join('eventlog', 'CSV', log_name + '.csv')
    if not path.isfile(csv_path):
        print('Generating CSV file from XES log...')
        xes_path = path.join('eventlog', 'XES', log_name)
        xes_path += '.xes.gz' if path.isfile(xes_path + '.xes.gz') else '.xes'
        log = xes_importer.apply(xes_path, variant=xes_importer.Variants.LINE_BY_LINE)
        for trace in log:
            trace.append({activity: FINAL_ACTIVITY, timestamp: trace[-1][timestamp] + timedelta(seconds=1)})
        dataframe = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)
        dataframe = dataframe.filter(items=[activity, timestamp, case_id]).sort_values(timestamp, kind='mergesort')
        dataframe = dataframe.rename(columns={activity: ACTIVITY_KEY, case_id: CASE_ID_KEY})
        makedirs(path.dirname(csv_path), exist_ok=True)
        dataframe.to_csv(csv_path, index=False)


def compute_model_complexity(net: PetriNet):
    """
    计算流程模型的复杂性
    :param net: 流程模型的petri net
    :return：库所数量、转换数量、有向弧数量和模型的“Extended Cardoso”度量
    """
    ext_card = 0
    for place in net.places:
        successor_places = set()
        for place_arc in place.out_arcs:
            successors = frozenset(transition_arc.target for transition_arc in place_arc.target.out_arcs)
            successor_places.add(successors)
        ext_card += len(successor_places)
    return len(net.places), len(net.transitions), len(net.arcs), ext_card


def get_save_file_name(env):
    filtering = 'UFL' if env.filtering else 'NFL'
    frequency = 'UFR' if env.frequency else 'NFR'
    update = 'D' if env.update else 'S'
    top_variants = 'P' if env.top is None else env.top
    update_param = 'DP' if env.update_param else 'SP'
    file = f'{env.order.name}_{env.algo.name}_{env.cut}_{top_variants}_{filtering}_{frequency}_{update}_{update_param}_{env.init_memory_size}_{env.init_sampling_rate}_{env.history_window}_{env.observation_window}_'
    return file


def save_results(env, agent_name):
    """
        导出报告、模型、评估指标
    """
    print('Exporting results...')
    file = get_save_file_name(env)
    folder = path.join('results', agent_name, env.log_name, 'report')
    makedirs(folder, exist_ok=True)
    top_variants = max(len(variants.keys()) for variants in env.drift_variants)
    # 'trace'是发生漂移时的已处理trace数量，'places', 'transitions', 'arcs', 'ext_cardoso'是petri网的复杂度表征，trace_{i}是当前的轨迹变体集
    columns = ['trace', 'places', 'transitions', 'arcs', 'ext_cardoso',
               *[f'trace_{i}' for i in range(1, top_variants + 1)]]
    report = DataFrame(columns=columns)
    report.index.name = 'n°_training'
    for index, current_variants in enumerate(env.drift_variants):
        traces = [f'[{v}]{k}' if env.order == Order.FRQ else f'[{len(k)}:{v}]{k}' for k, v in
                  current_variants.items()] + [None] * (top_variants - len(current_variants))
        report.loc[len(report)] = [env.drift_moments[index], *compute_model_complexity(env.models[index][0]), *traces]
    report.to_csv(path.join(folder, file + '.csv'))
    folder = path.join('results', agent_name, env.log_name, 'evaluation')
    makedirs(folder, exist_ok=True)
    columns = ['fitness', 'precision', 'f-measure', 'time']
    evaluation = DataFrame(env.evaluations, columns=columns)
    evaluation.index.name = 'n°_evaluation'
    total_time = evaluation['time'].sum()
    evaluation.loc['AVG'] = evaluation.mean()
    evaluation.loc['TOT'] = [None, None, None, total_time]
    evaluation.to_csv(path.join(folder, file + '.csv'))
    # 保存流程模型的petri网，包括pnml和png
    # folder = path.join('results', agent_name, self.log_name, 'petri')
    # makedirs(folder, exist_ok=True)
    # for index, model in enumerate(self.models):
    #     model_info = f'-{index}' if self.update else ''
    #     pnml_exporter.apply(model[0], model[1], path.join(folder, file + model_info + '.pnml'), model[2])
    #     pn_visualizer.save(pn_visualizer.apply(*model), path.join(folder, file + model_info + '.png'))

def drift_visualization(env, agent_name):
    """
        评估指标和概念漂移的可视化
    """
    file = get_save_file_name(env)
    folder = path.join('results', agent_name, env.log_name, 'drift')
    makedirs(folder, exist_ok=True)
    plt.figure()
    plt.subplot(5, 1, 1)
    plt.plot(np.array(env.evaluations[1:])[:,0])
    plt.vlines(x= np.where(np.array(env.drift_flag)==1), ymin=0.5, ymax=1, linestyles='dashed', colors='red')
    plt.title("Fitness")
    plt.subplot(5, 1, 2)
    plt.plot(np.array(env.evaluations[1:])[:,1])
    plt.vlines(x= np.where(np.array(env.drift_flag)==1), ymin=0.5, ymax=1, linestyles='dashed', colors='red')
    plt.title("Precision")
    plt.subplot(5, 1, 3)
    plt.plot(np.array(env.evaluations[1:])[:,2])
    plt.vlines(x= np.where(np.array(env.drift_flag)==1), ymin=0.5, ymax=1, linestyles='dashed', colors='red')
    plt.title("F_measure")
    plt.subplot(5, 1, 4)
    plt.plot(env.memory_size_list)
    plt.vlines(x= np.where(np.array(env.drift_flag)==1), ymin=250, ymax=500, linestyles='dashed', colors='red')
    plt.title("Memory_size")
    plt.subplot(5, 1, 5)
    plt.plot(env.sampling_rate_list)
    plt.vlines(x= np.where(np.array(env.drift_flag)==1), ymin=0.5, ymax=1, linestyles='dashed', colors='red')
    plt.title("Sampling_rate")
    plt.tight_layout() # 解决标题重叠问题
    plt.savefig(path.join(folder, file + '.png'))
    plt.close()


def generate_summary(agent_name, log_name):
    """
    生成所获得结果的摘要视图
    :param log_name: 要为其生成结果摘要的日志的名称
    """
    folder = path.join('results', agent_name, log_name)
    if not path.isdir(folder):
        print('No results found')
        return
    print('Generating summary...\n')
    error_file = path.join(folder, 'errors.csv')
    errors = read_csv(error_file) if path.isfile(error_file) else DataFrame(columns=['algo'])
    columns = ['order', 'top-variants', 'set-up', 'memory_size', 'sampling_rate', 'history_window', 'observation_window', 'fitness', 'precision', 'f-measure', 'time', 'drift_count', 'ext_cardoso']
    for algo in Algo:
        summary = DataFrame(columns=columns)
        for error in errors.loc[errors['algo'] == algo.name].itertuples():
            row = [error.order, error.top, f'{error.filtering} {error.frequency} {error.update}'] + ['-'] * 5
            summary.loc[len(summary)] = row
        for file in listdir(path.join(folder, 'evaluation')):
            if algo.name in file:
                parameters = file.split(sep='_')
                row = [parameters[0], parameters[3], f'{parameters[4]} {parameters[5]} {parameters[6]} {parameters[7]}', parameters[8], parameters[9], parameters[10], parameters[11]]
                evaluation = read_csv(path.join(folder, 'evaluation', file), dtype={'n°_evaluation': 'str'})
                row.extend(evaluation[val][len(evaluation) - 2] for val in ('fitness', 'precision', 'f-measure'))
                row.append(evaluation['time'][len(evaluation) - 1])
                report = read_csv(path.join(folder, 'report', file))
                row.append(len(report)-1)
                row.append(str(report['ext_cardoso'].tolist()))
                summary.loc[len(summary)] = row
        summary['top-variants'] = summary['top-variants'].replace('P', -1).astype(int)
        summary = summary.sort_values(['order', 'top-variants', 'set-up'], ignore_index=True)
        summary['top-variants'] = summary['top-variants'].replace(-1, 'P')
        summary.index.name = 'experiment'
        summary.index += 1
        summary.to_csv(path.join(folder, algo.name + '.csv'))


def save_models(env, agent_name):
    print('\nExporting models...')
    file = get_save_file_name(env)
    folder = path.join('results', agent_name, env.log_name, 'model')
    makedirs(folder, exist_ok=True)

    model_path = path.join(folder, file + '.txt')
    with open(model_path, mode='wb+') as model_file:
        pickle.dump(env, model_file)
    return model_path


def load_models(model_path):
    with open(model_path, mode='rb') as model_file:
        model = pickle.load(model_file)
    return model


def pruned_tree(check_case_id, hashtable):
    if check_case_id.is_root:
        return
    result = dict((new_val, new_k) for new_k, new_val in hashtable.items()).get(check_case_id)
    if check_case_id.is_leaf and result is None:
        # 当前活动在树中是叶节点且没有被其他未完成case使用，则将当前节点删除，并尝试修剪他的父节点
        pruned_tree_recursive(check_case_id.parent, hashtable)
        check_case_id.parent = None


def pruned_tree_recursive(check_case_id, hashtable):
    if check_case_id.is_root:
        return
    result = dict((new_val, new_k) for new_k, new_val in hashtable.items()).get(check_case_id)
    if len(check_case_id.children) == 1 and result is None:
        # 当前活动在树中的子节点只有一个（即要被上一轮递归删除的那个）且没有被其他未完成case使用，则将当前节点删除，并尝试修剪他的父节点
        pruned_tree_recursive(check_case_id.parent, hashtable)
        check_case_id.parent = None


def check_exist_child(node, name):
    for n in node:
        if name == n.id:
            return n


def get_trace(node):
    list_act = []
    if node.is_root:
        return list_act
    else:
        list_act = get_trace(node.parent)
        list_act.append(node.id)
        return list_act
