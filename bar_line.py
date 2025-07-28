import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

font = {'family': 'Times New Roman',
        'weight': '400',
        'size': 15,
        }

def overlapped_bar(
    true_o, pre_o, true_l, pre_l, time_l, label_list,
    width=1, alpha=.6,
    title='', xlabel='', ylabel='', figsize=(7, 6),
    hide_xsticks=True, show=False, save_name="",
    **plot_kwargs
):
    """
    绘制重叠条形图

    :param high_list: 待处理数据, 数值较大的列表
    :param low_list: 待处理数据, 数值较小的列表
    :param label_list: 标签列表，默认2个字符串, 例如 ['A', 'B']
    :param width: 每个 bin 的宽度
    :param alpha: 透明度
    :param title: 图表标题
    :param xlabel: X 轴标签
    :param ylabel: Y 轴标签
    :param figsize: 图像尺寸
    :param hide_xsticks: 是否隐藏 X 轴的索引，建议数据较多时，隐藏
    :param show:  是否显示
    :param save_name:  是否存储图像
    :param plot_kwargs:  其余参数
    :return:
    """
    assert len(true_o) == len(pre_o) and len(label_list) == 4
    df = pd.DataFrame(np.matrix([true_l, pre_l,true_o, pre_o]).T, columns=label_list)

    plt.figure(figsize=figsize)  # 设置plt的尺寸
    xlabel = xlabel or df.index.name  # 标签
    N = len(df)   # 类别数
    M = len(df.columns)   # 列数
    indices = np.arange(N)
    colors = ['darkgray', 'r', 'darkgrey', 'darkgreen'] # 颜色
    for i, label, color in zip(range(M), df.columns, colors):
        kwargs = plot_kwargs
        kwargs.update({'color': color, 'label': label})
        plt.bar(indices, df[label], width=width, alpha=alpha if i else 1, **kwargs)
        # if not hide_xsticks:  # 如果水平坐标太多，隐藏水平坐标
        #     plt.xticks(indices + .5 * width, ['{}'.format(idx) for idx in df.index.values])

    plt.plot(time_l, true_o, linewidth=1.0, color='darkgray')
    plt.plot(time_l, pre_o,  linewidth=1.0, color='forestgreen')
    plt.plot(time_l, true_l, linewidth=1.0, color='grey')
    plt.plot(time_l, pre_l, linewidth=1.0, color='brown')

    plt.legend()
    plt.title(title, font)
    plt.xlabel(xlabel, font)
    plt.ylabel(ylabel, font)
    a = np.arange(0, 580)
    plt.xticks(a[::100])
    if show:
        plt.savefig('./bar_line.png')
        plt.show()
    return plt.gcf()


data_2 = pd.read_csv("../test/data_2.csv")
true_l = data_2["真实值日产液"].tolist()
pre_l = data_2["预测值日产液"].tolist()
true_o = data_2["真实值日产油"].tolist()
pre_o = data_2["预测值日产油"].tolist()
time_l = data_2["time"].tolist()
overlapped_bar(true_o, pre_o, true_l, pre_l,time_l, label_list=["Real values for oil", "Prediction for oil", "Real values for fluid", "Prediction for fluid"], xlabel="time", ylabel="production", show=True, title="distribution of forecasted parameter")


