import pickle
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# results={'evaluation':env.evaluations,'memory_size_list':env.memory_size_list,'sampling_rate_list':env.sampling_rate_list,'drift_flag':env.drift_flag}

with open("fixed_hyperparameters_results.pickle", "rb") as file:
    fixed_results = pickle.load(file)

with open("rl_dynamic_hyperparameters_results.pickle", "rb") as file:
    dynamic_results = pickle.load(file)


font = {'family':'STIXGeneral', # 这个长得最像Times New Roman
    'style':'normal',
    'weight':'medium',
    'color':'black',
    'size': 18}
fig = plt.figure(figsize=(10,6))

axs=plt.subplot(2, 2, 1)
axs.plot(np.array(fixed_results['evaluation'][1:])[:400,0], color='cornflowerblue', alpha=1, label='fitness')
axs.plot(np.array(fixed_results['evaluation'][1:])[:400,1], color='green', alpha=0.5, label='precision')
axs.plot(np.array(fixed_results['evaluation'][1:])[:400,2], color='red', alpha=0.5, label='F measure')
axs.legend(fontsize=10, loc='lower left') # 图例
# axs.set_ylim(0, 1.1) # y 轴的范围
axs.set_yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1]) # y 轴的刻度
axs.tick_params(labelsize=10) # 刻度字体的大小
axs.set_ylabel('appropriateness', fontdict=font) # y 轴标题
axs.set_title('(a) Fixed parameters')

axs=plt.subplot(2, 2, 2)
axs.plot(np.array(dynamic_results['evaluation'][1:])[:400,0], color='cornflowerblue', alpha=1, label='fitness')
axs.plot(np.array(dynamic_results['evaluation'][1:])[:400,1], color='green', alpha=0.5, label='precision')
axs.plot(np.array(dynamic_results['evaluation'][1:])[:400,2], color='red', alpha=0.5, label='F measure')
# axs.legend(fontsize=8, loc='lower left') # 图例
# axs.set_ylim(0, 1.1) # y 轴的范围
axs.set_yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1]) # y 轴的刻度
axs.tick_params(labelsize=10) # 刻度字体的大小
axs.set_ylabel('appropriateness', fontdict=font) # y 轴标题
axs.set_title('(b) Dynamic parameters')

axs=plt.subplot(2, 2, 3)
axs.plot(fixed_results['memory_size_list'][:400], label='memory Size')
plt.vlines(x= np.where(np.array(fixed_results['drift_flag'][:400])==1), ymin=220, ymax=250, linestyles='dashed', colors='red')
axs.set_yticks([0,50,100,150,200,250]) # y 轴的刻度
axs.tick_params(labelsize=10) # 刻度字体的大小
legend_handles = [Line2D([0], [0], lw=2, label='memory Size'),
                Line2D([0], [0], color='red', linestyle='dashed', lw=2, label='detected concept drift')]
axs.legend(handles=legend_handles, fontsize=10, loc='lower left')
axs.set_ylabel('memory size', fontdict=font) # y 轴标题
axs.set_xlabel('time step', fontdict=font) # x 轴标题


axs=plt.subplot(2, 2, 4)
axs.plot(dynamic_results['memory_size_list'][:400], label='memory Size')
plt.vlines(x= np.where(np.array(dynamic_results['drift_flag'][:400])==1), ymin=220, ymax=250, linestyles='dashed', colors='red')
axs.set_yticks([0,50,100,150,200,250]) # y 轴的刻度
axs.tick_params(labelsize=10) # 刻度字体的大小
axs.set_ylabel('memory size', fontdict=font) # y 轴标题
axs.set_xlabel('time step', fontdict=font) # x 轴标题

plt.tight_layout() # 减小间距，这个会改变标题等地方的字体大小
plt.savefig('dynamic_hyperparameters_analysis.png') # 保存为png格式
plt.savefig('dynamic_hyperparameters_analysis.pdf') # 保存为pdf格式




fig = plt.figure(figsize=(5,6))
axs=plt.subplot(2, 1, 1)
axs.plot(np.array(fixed_results['evaluation'][1:])[:400,0], color='cornflowerblue', alpha=1, label='fitness')
axs.plot(np.array(fixed_results['evaluation'][1:])[:400,1], color='green', alpha=0.5, label='precision')
axs.plot(np.array(fixed_results['evaluation'][1:])[:400,2], color='red', alpha=0.5, label='F measure')
axs.legend(fontsize=10, loc='lower left') # 图例
# axs.set_ylim(0, 1.1) # y 轴的范围
axs.set_yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1]) # y 轴的刻度
axs.tick_params(labelsize=10) # 刻度字体的大小
axs.set_ylabel('appropriateness', fontdict=font) # y 轴标题
# axs.set_title('(a) Fixed parameters')

axs=plt.subplot(2, 1, 2)
axs.plot(fixed_results['memory_size_list'][:400], label='memory Size')
plt.vlines(x= np.where(np.array(fixed_results['drift_flag'][:400])==1), ymin=220, ymax=250, linestyles='dashed', colors='red')
axs.set_yticks([0,50,100,150,200,250]) # y 轴的刻度
axs.tick_params(labelsize=10) # 刻度字体的大小
legend_handles = [Line2D([0], [0], lw=2, label='memory size'),
                Line2D([0], [0], color='red', linestyle='dashed', lw=2, label='detected concept drift')]
axs.legend(handles=legend_handles, fontsize=10, loc='lower left')
axs.set_ylabel('memory size', fontdict=font) # y 轴标题
axs.set_xlabel('time step', fontdict=font) # x 轴标题

plt.tight_layout() # 减小间距，这个会改变标题等地方的字体大小
plt.savefig('fixed_parameters.png') # 保存为png格式
plt.savefig('fixed_parameters.pdf') # 保存为pdf格式


fig = plt.figure(figsize=(5,6))

axs=plt.subplot(2, 1, 1)
axs.plot(np.array(dynamic_results['evaluation'][1:])[:400,0], color='cornflowerblue', alpha=1, label='fitness')
axs.plot(np.array(dynamic_results['evaluation'][1:])[:400,1], color='green', alpha=0.5, label='precision')
axs.plot(np.array(dynamic_results['evaluation'][1:])[:400,2], color='red', alpha=0.5, label='F measure')
# axs.legend(fontsize=8, loc='lower left') # 图例
# axs.set_ylim(0, 1.1) # y 轴的范围
axs.set_yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1]) # y 轴的刻度
axs.tick_params(labelsize=10) # 刻度字体的大小
axs.set_ylabel('appropriateness', fontdict=font) # y 轴标题
# axs.set_title('(b) Dynamic parameters')

axs=plt.subplot(2, 1, 2)
axs.plot(dynamic_results['memory_size_list'][:400], label='memory Size')
plt.vlines(x= np.where(np.array(dynamic_results['drift_flag'][:400])==1), ymin=220, ymax=250, linestyles='dashed', colors='red')
axs.set_yticks([0,50,100,150,200,250]) # y 轴的刻度
axs.tick_params(labelsize=10) # 刻度字体的大小
axs.set_ylabel('memory size', fontdict=font) # y 轴标题
axs.set_xlabel('time step', fontdict=font) # x 轴标题

plt.tight_layout() # 减小间距，这个会改变标题等地方的字体大小
plt.savefig('dynamic_parameters.png') # 保存为png格式
plt.savefig('dynamic_parameters.pdf') # 保存为pdf格式