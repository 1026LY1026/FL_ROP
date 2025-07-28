# 导入必要的模块
# 绘图与数据分析模块
import matplotlib.pyplot as plt
from pylab import mpl

# 设置matplotlib的配置
# mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# 忽略警告提示
import warnings

warnings.filterwarnings('ignore')

def acc_loss_plot_(train_data, type_, path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_data, label='train_data', color='blue', linewidth=1)
    plt.xlabel('epoch', fontsize=18)
    plt.title(f'train_test_{type_}')
    path_ = f'{path}'

    plt.grid()
    plt.savefig(path_)

    plt.legend()
   # plt.show()

def acc_loss_plot(train_data, test_data, type_, path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_data, label='train_data', color='blue', linewidth=1)
    plt.plot(test_data, label='test_data', color='red', linewidth=1)
    plt.xlabel('epoch', fontsize=18)
    plt.title(f'train_test_{type_}')
    path_ = f'{path}'

    plt.grid()
    plt.savefig(path_)

    plt.legend()
  #  plt.show()


def true_test_plot(depth, true_data, predicted_data, type_, path):
    plt.figure(figsize=(20, 6))
    plt.plot(depth, true_data, label='true_data', color='blue', linewidth=1)
    plt.plot(depth, predicted_data, label='test_data', color='green', linewidth=1)
    plt.ylabel("GRA", fontsize=18)
    plt.xlabel('depth', fontsize=18)
    path_ = f'{path}'
    plt.grid()
    plt.savefig(path_)
#    plt.show()