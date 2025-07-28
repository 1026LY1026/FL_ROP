# from encodings import gb18030

import pandas as pd
import numpy as np

# from federated.fed_oil import *
# 加载数据集
import torch
import argparse
volve_4="./data/S_F4.csv"
volve_5="./data/S_F5.csv"
volve_10="./data/S_F10.csv"
volve_12="./data/S_F12.csv"
volve_14="./data/S_F14.csv"
well_2="./data/well_2.csv"
well_3="./data/well_3.csv"
bz_6="./data/BZ19_6_6.csv"
bz_10="./data/BZ19_6_10.csv"

# 做标准化归一化
def noramlization(data):
    minVals = data.min(0)
    maxVals = data.max(0)
    ranges = maxVals - minVals
    normData = (data - minVals) / ranges
    return normData

# 数据加载函数，负责加载数据返回特征和标签  X Y
def data_load(path, is_normalize):
    Data_ = pd.read_csv(path)
    # 是否做归一化
    for col_index, col_value in Data_.items():
        minVals = col_value.min(0)
        maxVals = col_value.max(0)
        if maxVals - minVals > 1000:
            col_value = noramlization(col_value)
            Data_.loc[:, col_index] = np.array(col_value)

    # 取数据
    X_data = Data_.values.astype(np.float32)

    # 取目标值（或说取y 取标签，这里是rop）
    y_data = Data_.iloc[:, -1].values

    return X_data, y_data

# 制作成时间序列数据集(分为训练集和测试集)  返回data （[[数据]，[标签]]）
def make_time_series_data(seq_length, pre_length, y_data, x_data,train_ratio):
    y_data = y_data[seq_length:]
    y_data = y_data.astype(np.float32)
    # an.log_train_rido = train_ratio

    # total data in torch format: input followed by output, non-uniform
    # 整理成torch  要求的格式 （[[数据]，[标签]]）
    data = []
    data_num = x_data.shape[0]
    for i in range(data_num - seq_length - pre_length):
        # input data
        input_temp = x_data[i:i + seq_length, :]
        # collect input and output
        data_temp = [input_temp, y_data[i:i + pre_length]]
        # append output column
        data.append(data_temp)

    # It is divided into training set and test set
    # 分成测试集和训练集
    # train_data_num = int(len(data) * train_ratio)
    # train_data = data[0:train_data_num]
    # test_data = data[train_data_num:]

    return data
# 计算准确率和loss
def compute_acc(iter_, pre_length, net, device, isTest=False):
    acc_ = np.zeros((1, pre_length))
    y_ = np.zeros((1, pre_length))
    for X, y in iter(iter_):
        with torch.no_grad():
            var_x = X.to(device)
            y_pre = net(var_x)
            # 按垂直方向（行顺序）堆叠数组构成一个新的数组
            acc_ = np.vstack((acc_, y_pre.cpu().detach().numpy()))
            y_ = np.vstack((y_, y.detach().numpy()))

    mse_loss = ((acc_ - y_) ** 2).mean()
    acc_ = acc_[:, 0]
    y_ = y_[:, 0]

    # 去掉为0的项
    index_ = np.where(y_ == 0)
    y_ = np.delete(y_, index_)
    acc_ = np.delete(acc_, index_)

    return 1 - (np.abs(acc_ - y_) / y_).mean(), mse_loss



