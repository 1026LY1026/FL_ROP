# -*- coding: UTF-8 -*-
from torch import nn
from nets.models import lstm_uni_attention
import warnings
warnings.filterwarnings("ignore", category=Warning)
from federated import data_preprocess
from federated.data_preprocess import *

from torch.utils.data import DataLoader

pre_length=10

hidden_size=128
num_layers=2
seq_length = 100
train_ratio = 0.8
batch_size = 64
features_num = 13

device = torch.device('cpu')
server_model = lstm_uni_attention(input_size=features_num, hidden_size=hidden_size, num_layers=num_layers,
                                             pre_length=pre_length, seq_length=seq_length).to(device)

loss_fun=nn.MSELoss()

# x_data_4, y_data_4 = data_preprocess.data_load(data_path_4, False, elements=elements)
# train_data1,test_data1 = data_preprocess.make_time_series_data(seq_length, pre_length, y_data_4, x_data_4,train_ratio)
# test_data1_loader = torch.utils.data.DataLoader(test_data1,batch_size=batch_size,
#                                          shuffle=False, num_workers=0)
# # print(len(test_data1))
# # 3031
# x_data_14, y_data_14 = data_preprocess.data_load(data_path_14, False, elements=elements)
# train_data3,test_data3 = data_preprocess.make_time_series_data(seq_length, pre_length, y_data_14, x_data_14,train_ratio)
# test_data3_loader = torch.utils.data.DataLoader(test_data3,batch_size=batch_size,
#                                          shuffle=False, num_workers=0)
#
# x_data_12, y_data_12 = data_preprocess.data_load(data_path_12, False, elements=elements)
# train_data2,test_data2 = data_preprocess.make_time_series_data(seq_length, pre_length, y_data_12, x_data_12,train_ratio)
# test_data2_loader = torch.utils.data.DataLoader(test_data2,batch_size=batch_size,
#                                          shuffle=False, num_workers=0)
#
# x_data_9, y_data_9 = data_preprocess.data_load(data_path_9A, False, elements=elements)
# train_data9,test_data9 = data_preprocess.make_time_series_data(seq_length, pre_length, y_data_9, x_data_9,train_ratio)
# test_data9_loader = torch.utils.data.DataLoader(test_data9,batch_size=batch_size,
#                                          shuffle=False, num_workers=0)
x_data_10, y_data_10 = data_preprocess.data_load(data_path_10A, False)
test_data10 = data_preprocess.make_time_series_data(seq_length, pre_length, y_data_10, x_data_10, train_ratio)
test_data10_loader = torch.utils.data.DataLoader(test_data10,batch_size=batch_size,
                                         shuffle=False, num_workers=0)
# from sklearn.preprocessing import StandardScaler
# depth = x_data_10[seq_length:-pre_length, 0]
#
# print(len(depth))
# print(depth)

checkpoint = torch.load(r'G:\FN-oil\modeldata\99_pre.pth',map_location=torch.device('cpu'))
server_model.load_state_dict(checkpoint)
#
#
# 开始测试
def testacc(a):
    acc_ = np.zeros((1, pre_length))
    y_ = np.zeros((1, pre_length))
    for X, y in iter(a):
        with torch.no_grad():
            var_x = X.to(device)
            y_pre = server_model(var_x)
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
    acc = 1 - (np.abs(acc_ - y_) / y_).mean()
    print(y_)
    print(acc_)
    return acc,y_,acc_

acc,y_,acc_=testacc(test_data10_loader)

# test=pd.DataFrame(columns=['1'], data=acc_)
test=pd.DataFrame(columns=['2'], data=y_)
test.to_csv('save2.csv', index=False, sep=',')
print("acc:{}".format(acc))


#  画图
# def plot(x_data,test_data):
#     depth = x_data[seq_length:-pre_length, 0]
#     print(len(depth))
# #
# #     num = int(len(depth) * 0.8)
# #     testdep = depth[num:]
# #     print(len(testdep))
# #
#     acc_ = np.zeros((1, pre_length))
#     y_ = np.zeros((1, pre_length))
#     for X, y in iter(test_data):
#         with torch.no_grad():
#             var_x = X.to(device)
#             y_pre = server_model(var_x)
#             acc_ = np.vstack((acc_, y_pre.cpu().detach().numpy()))
#             y_ = np.vstack((y_, y.detach().numpy()))
#             col_one_real = y_[:, 0].copy()
#             col_one_pre = acc_[:, 0].copy()
#     real = col_one_real[1:]
#     pre = col_one_pre[1:]
#     print(real)
#     print(pre)
#     print(len(real))
#     print(len(pre))
#     font = {'family': 'Times New Roman',
#             'weight': '400',
#             'size': 25,
#             }
#     unit = ["ROP, m/h", "Depth, m"]
#     plt.figure(figsize=(24, 8))
#     plt.ylabel(unit[0], font)
#     plt.xlabel(unit[1], font)
#     # plt.xticks(range(0,2000,200))
# # #
#     plt.plot(depth, pre)
#     plt.plot(depth, real)
# # #     plt.plot(testdep, pre)
# # #     plt.plot(testdep, real)
#     plt.tick_params(labelsize=20)
#     plt.legend(['pre', 'real'], prop=font)
#     plt.savefig('./png/lstm_ROP_D2.png')
#     plt.close()
# #
# plot(x_data_10,test_data10_loader)