import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors

# 设置全局样式和字体
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
sns.set_style("whitegrid")
plt.rcParams['axes.unicode_minus'] = False

# 您提供的原始数据
data = {
    'alpha': [0.001, 0.01, 0.1, 0.001, 0.01, 0.1, 0.001, 0.01, 0.1],
    'lambda': [0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.001, 0.001, 0.001],
    'hr': [35.38, 36.49, 37.01, 37.31, 37.95, 36.09, 34.87, 39.54, 39.77],
    'ndcg': [17.89, 18.67, 18.70, 19.80, 20.14, 18.43, 18.15, 21.22, 21.21]
}

# 创建DataFrame
df = pd.DataFrame(data)

# 为了创建热力图，我们需要将数据透视成矩阵形式
hr_matrix = df.pivot(index='alpha', columns='lambda', values='hr')
ndcg_matrix = df.pivot(index='alpha', columns='lambda', values='ndcg')

custom_cmap = colors.LinearSegmentedColormap.from_list(
    'soft_pink', ['#fff0f5', '#ffd6e7', '#ffc2d6', '#ffadc6', '#ff99b5', '#ff85a5', '#ff7096', '#ff5c87'], N=256
)

# 创建图形和轴
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
#fig.suptitle('超参数调优实验结果热力图', fontsize=18, fontweight='bold', y=0.98)

# 绘制HR热力图
hr_plot = sns.heatmap(hr_matrix,
                      annot=True,
                      fmt=".2f",
                      cmap=custom_cmap,
                      cbar_kws={'label': 'HR (%)'},
                      ax=ax1,
                      linewidths=0.5,
                      linecolor='lightgray',
                      annot_kws={"size": 20, "color": "black"})
ax1.set_title('HR@10', fontsize=20, pad=12)
ax1.set_xlabel('λ (lambda)', fontsize=20)
ax1.set_ylabel('α (alpha)', fontsize=20)
ax1.tick_params(labelsize=18)

# 绘制NDCG热力图
ndcg_plot = sns.heatmap(ndcg_matrix,
                        annot=True,
                        fmt=".2f",
                        cmap=custom_cmap,
                        cbar_kws={'label': 'NDCG (%)'},
                        ax=ax2,
                        linewidths=0.5,
                        linecolor='lightgray',
                        annot_kws={"size": 20, "color": "black"})
ax2.set_title('NDCG@10', fontsize=20, pad=12)
ax2.set_xlabel('λ (lambda)', fontsize=20)
ax2.set_ylabel('')
ax2.tick_params(labelsize=18)

# 调整颜色条标签字体大小
if hr_plot.collections[0].colorbar:
    hr_plot.collections[0].colorbar.ax.set_ylabel('HR (%)', fontsize=12)
    hr_plot.collections[0].colorbar.ax.tick_params(labelsize=10)

if ndcg_plot.collections[0].colorbar:
    ndcg_plot.collections[0].colorbar.ax.set_ylabel('NDCG (%)', fontsize=12)
    ndcg_plot.collections[0].colorbar.ax.tick_params(labelsize=10)

# 调整布局
plt.tight_layout()

# 突出显示最佳性能值
max_hr_value = hr_matrix.max().max()
for i in range(len(hr_matrix.index)):
    for j in range(len(hr_matrix.columns)):
        if hr_matrix.iloc[i, j] == max_hr_value:
            ax1.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black', lw=4))

max_ndcg_value = ndcg_matrix.max().max()
for i in range(len(ndcg_matrix.index)):
    for j in range(len(ndcg_matrix.columns)):
        if ndcg_matrix.iloc[i, j] == max_ndcg_value:
            ax2.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black', lw=4))

# 添加文本说明
"""plt.figtext(0.5, 0.01, 
            f'最佳HR值: {max_hr_value} (η={hr_matrix.stack().idxmax()[0]}, λ={hr_matrix.stack().idxmax()[1]}) | '
            f'最佳NDCG值: {max_ndcg_value} (η={ndcg_matrix.stack().idxmax()[0]}, λ={ndcg_matrix.stack().idxmax()[1]})', 
            ha='center', fontsize=12, style='italic')"""

plt.show()

# 保存高分辨率图像（可选）
# fig.savefig('hyperparameter_heatmap.png', dpi=300, bbox_inches='tight')
# print("图像已保存为 'hyperparameter_heatmap.png'")