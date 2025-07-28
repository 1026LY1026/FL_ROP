import os
from torch import nn, optim
import time
import copy
from nets.models import lstm_uni_attention
import warnings
warnings.filterwarnings("ignore", category=Warning)
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import torch
import argparse

seq_length = 100
pre_length = 50
train_ratio = 0.8
batch_size = 64
features_num = 12

# 加载数据集
volve_4="./data/S_F4.csv"
volve_5="./data/S_F5.csv"
volve_10="./data/S_F10.csv"
volve_12="./data/S_F12.csv"
volve_14="./data/S_F14.csv"
well_2="./data/well_2.csv"
well_3="./data/well_3.csv"
bz_6="./data/BZ19_6_6.csv"
bz_10="./data/BZ19_6_10.csv"

# 数据加载函数，负责加载数据返回特征和标签  X Y
def data_load(path, is_normalize):
    Data_ = pd.read_csv(path)

    # 取数据
    X_data = Data_.values.astype(np.float32)

    # 取目标值（或说取y 取标签，这里是rop）
    y_data = Data_.iloc[:, -1].values

    return X_data, y_data

# 制作成时间序列数据集(分为训练集和测试集)  返回data （[[数据]，[标签]]）
def make_time_series_data(seq_length, pre_length, y_data, x_data,train_ratio):
    y_data = y_data[seq_length:]
    y_data = y_data.astype(np.float32)
    # 整理成torch  要求的格式 （[[数据]，[标签]]）
    data = []
    data_num = x_data.shape[0]
    for i in range(0,data_num - seq_length - pre_length+1):
        input_temp = x_data[i:i + seq_length, :]
        # collect input and output
        data_temp = [input_temp, y_data[i:i + pre_length]]
        # append output column
        data.append(data_temp)

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

# 划分数据集X Y
x_data_4, y_data_4 = data_load(volve_4, False)
x_data_12, y_data_12 = data_load(volve_12, False)
x_data_14, y_data_14 = data_load(volve_14, False)
x_data_10, y_data_10 = data_load(volve_10, False)
# 做成时序数据集
train_data1 = make_time_series_data(seq_length, pre_length, y_data_4, x_data_4, train_ratio)
train_data2 = make_time_series_data(seq_length, pre_length, y_data_12, x_data_12, train_ratio)
train_data3= make_time_series_data(seq_length, pre_length, y_data_14, x_data_14, train_ratio)
test_data = make_time_series_data(seq_length, pre_length, y_data_10, x_data_10, train_ratio)

train_data1_loader = torch.utils.data.DataLoader(train_data1, batch_size=batch_size,
                                         shuffle=False, num_workers=0)
train_data2_loader = torch.utils.data.DataLoader(train_data2, batch_size=batch_size,
                                         shuffle=False, num_workers=0)
train_data3_loader = torch.utils.data.DataLoader(train_data3, batch_size=batch_size,
                                         shuffle=False, num_workers=0)
test_data1_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                         shuffle=False, num_workers=0)

train_loaders = [train_data1_loader, train_data2_loader, train_data3_loader]
test_loaders = [test_data1_loader]

# 定义训练方法
def train(model, train_loader, optimizer, loss_fun, client_num, device,lr= 0.00015):
    criterion = loss_fun.to(device)
    model.train()
    # 存储模型预测的输出 y_pre 和真实标签 y，pre_length 是预测序列的长度
    acc_=np.zeros((1,pre_length))
    y_=np.zeros((1,pre_length))
    train_iter = iter(train_loader) # 将训练数据加载器 train_loader 转换为可迭代对象，方便按批次获取数据
    for step in range(len(train_iter)):
        optimizer.zero_grad()
        x, y = next(train_iter) # 获取训练数据和对应的标签
        # num_data += y.size(0)
        x = x.to(device).float()
        y = y.to(device).float()
        y_pre = model(x)
        loss = criterion(y_pre, y)

        loss.backward()
        optimizer.step()

        acc_=np.vstack((acc_,y_pre.cpu().detach().numpy()))
        y_ = np.vstack((y_, y.detach().cpu().numpy()))
    mse_loss = ((acc_ - y_) ** 2).mean()
    acc_ = acc_[:, 0]
    y_ = y_[:, 0]

    index_ = np.where(y_ == 0)
    y_ = np.delete(y_, index_)
    acc_ = np.delete(acc_, index_)

    return 1 - (np.abs(acc_ - y_) / y_).mean(), mse_loss

# 定义train—prox训练方法
def train_prox(model, train_loader, optimizer, loss_fun, client_num, device,lr= 0.00015):
    criterion = loss_fun.to(device)
    # optimizer and parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)  # weight_decay=0.001 范数，提高泛化能力
    model.train()
    acc_=np.zeros((1,pre_length))
    y_=np.zeros((1,pre_length))
    train_iter = iter(train_loader)
    for step in range(len(train_iter)):
        optimizer.zero_grad()
        x, y = next(train_iter)
        # num_data += y.size(0)
        x = x.to(device).float()
        y = y.to(device).long()
        y_pre = model(x)
        loss = criterion(y_pre, y)
        # 添加loss修正项
        if step>0:
            w_diff = torch.tensor(0., device=device)
            for w, w_t in zip(server_model.parameters(), model.parameters()):
                w_diff += torch.pow(torch.norm(w - w_t), 2)
            loss += args.mu / 2. * w_diff
        loss.backward()
        optimizer.step()

        acc_=np.vstack((acc_,y_pre.cpu().detach().numpy()))
        y_=np.vstack((y_,y.detach.numpy()))
    mse_loss = ((acc_ - y_) ** 2).mean()
    acc_ = acc_[:, 0]
    y_ = y_[:, 0]

    index_ = np.where(y_ == 0)
    y_ = np.delete(y_, index_)
    acc_ = np.delete(acc_, index_)

    return 1 - (np.abs(acc_ - y_) / y_).mean(), mse_loss

# 测试方法：
def test(model, test_loader, loss_fun, device):
    model.eval()
    acc_ = np.zeros((1, pre_length))
    y_ = np.zeros((1, pre_length))
    for X, y in iter(test_loader):
        with torch.no_grad():
            var_x = X.to(device)
            y_pre = model(var_x)
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


# 主要功能communication
# ‘bn.weight’,
# ‘bn.bias’,
# ‘bn.running_mean’,
# ‘bn.running_var’,
# ‘bn.num_batches_tracked’,
def communication(args, server_model, models, client_weights):
    '''
    args：传入的参数对象，包含训练的配置项。
    server_model：服务器端的模型，通常是汇聚后的模型。
    models：每个客户端的模型列表，在联邦学习中，每个客户端有一个独立的模型副本。
    client_weights：每个客户端的权重，用于加权聚合模型参数
    '''
    with torch.no_grad():
        # aggregate params
        if args.mode.lower() == 'fedbn':
            for key in server_model.state_dict().keys():
                if 'bn' not in key: # 跳过包含批量归一化（Batch Normalization，bn）的参数，因为批量归一化的参数在联邦学习中通常不需要像其他参数一样进行权重聚合
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32) # 创建一个与当前参数相同形状和数据类型的零张量，用于存储加权聚合后的参数
                    for client_idx in range(client_num):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key] # 使用每个客户端的权重和对应模型的参数进行加权累加
                    server_model.state_dict()[key].data.copy_(temp) # 将聚合后的参数更新到服务器模型的对应位置
                    for client_idx in range(client_num):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key]) # 将聚合后的参数同步回每个客户端的模型
        else:
            for key in server_model.state_dict().keys():
                # num_batches_tracked is a non trainable LongTensor and
                # num_batches_tracked are the same for all clients for the given datasets
                if 'num_batches_tracked' in key: # 查参数名称中是否包含 num_batches_tracked，该参数是 Batch Normalization 层中的一个计数器，不需要进行加权聚合
                    server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])  # 将第一个客户端的 num_batches_tracked 参数同步到服务器端
                else: # 对于其他模型参数，进行加权聚合
                    temp = torch.zeros_like(server_model.state_dict()[key])
                    for client_idx in range(len(client_weights)):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(len(client_weights)):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
    return server_model, models # 返回值：返回更新后的 server_model（服务器模型）和 models（客户端模型）

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed= 1
    np.random.seed(seed) # 设置NumPy的随机种子
    torch.manual_seed(seed) # 设置PyTorch CPU的随机种子
    torch.cuda.manual_seed_all(seed) # 设置PyTorch GPU的随机种子（如果使用GPU）

    print('Device:', device)
    parser = argparse.ArgumentParser()  # 创建一个ArgumentParser对象，用于解析命令行传入的参数
    parser.add_argument('--log', default=True, help='whether to make a log') # 是否记录日志（默认值为True）
    parser.add_argument('--test', action='store_true', help='test the pretrained model') # 如果添加该参数，则表示测试预训练的模型。此参数是一个布尔值，若指定则为True
    # parser.add_argument('--percent', type = float, default= 0.1, help ='percentage of dataset to train')
    parser.add_argument('--lr', type=float, default=0.00015, help='learning rate')
    parser.add_argument('--batch', type=int, default=64, help='batch size')
    parser.add_argument('--iters', type=int, default=50, help='iterations for communication') # 全局迭代次数
    parser.add_argument('--wk_iters', type=int, default=50,help='optimization iters in local worker between communication') # 每次本地工作者优化的迭代次数
    parser.add_argument('--mode', type=str, default='fedbn', help='fedavg | fedprox | fedbn') # 择联邦学习的模式，默认值为'fedbn'，可以是fedavg、fedprox或fedbn
    parser.add_argument('--mu', type=float, default=1e-2, help='The hyper parameter for fedprox') # fedprox模式下的超参数（默认为1e-2）
    parser.add_argument('--save_path', type=str, default='../checkpoint/oil', help='path to save the checkpoint')
    parser.add_argument('--resume', action='store_true', help='resume training from the save path checkpoint') # 是否从保存的检查点恢复训练。若指定，则为True
    args = parser.parse_args()

exp_folder = 'federated_oil'

args.save_path = os.path.join(args.save_path, exp_folder) # 将保存路径 args.save_path 和实验文件夹 exp_folder 拼接起来，形成一个新的路径，用于存储模型检查点或实验结果

log = args.log
if log:
    log_path = os.path.join('../logs/oil/', exp_folder)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logfile = open(os.path.join(log_path, '{}.log'.format(args.mode)), 'a')
    logfile.write('==={}===\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    logfile.write('===Setting===\n')
    logfile.write('    lr: {}\n'.format(args.lr))
    logfile.write('    batch: {}\n'.format(args.batch))
    logfile.write('    iters: {}\n'.format(args.iters))
    logfile.write('    wk_iters: {}\n'.format(args.wk_iters))

'''
args.save_path = "../checkpoint/oil/experiment_1"
args.mode = "fedbn" 执行后：
SAVE_PATH = "../checkpoint/oil/experiment_1/fedbn"'''
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
SAVE_PATH = os.path.join(args.save_path, '{}'.format(args.mode))

hidden_size=128
num_layers=2

# lstm_uni_attention_test = lstm_uni_attention(input_size=features_num, hidden_size=hidden_size, num_layers=num_layers,
#                                              pre_length=pre_length, seq_length=seq_length)
server_model = lstm_uni_attention(input_size=features_num, hidden_size=hidden_size, num_layers=num_layers,
                                             pre_length=pre_length, seq_length=seq_length).to(device)

loss_fun=nn.MSELoss()
# name of each client dataset
datasets = ['x_data_4', 'x_data_12', 'x_data_14'] # 不同客户端数据集的名称 模拟联邦学习环境中的多个客户端，每个客户端都有自己的数据集

# federated setting
client_num = len(datasets) # 客户端的数量
client_weights = [1 / client_num for i in range(client_num)] # 为每个客户端分配权重，均等分配（1 / 客户端数量）
# 对服务器端模型进行深拷贝，确保每个客户端都有独立的模型副本
models = [copy.deepcopy(server_model).to(device) for idx in range(client_num)]

# 测试预训练模型
if args.test:
    print('Loading snapshots...')
    checkpoint = torch.load('../snapshots/digits/{}'.format(args.mode.lower()))
    server_model.load_state_dict(checkpoint['server_model']) # 加载预训练模型的权重 （根据 args.mode 区分模式，如 fedavg、fedbn）
    if args.mode.lower() == 'fedbn': # 如果使用 fedbn 模式，需要加载每个客户端独立的模型参数
        for client_idx in range(client_num):
            models[client_idx].load_state_dict(checkpoint['model_{}'.format(client_idx)])
        for test_idx, test_loader in enumerate(test_loaders): # 遍历测试集 test_loaders 并计算每个客户端的测试精度
            _, test_acc = test(models[test_idx], test_loader, loss_fun, device)
            print(' {:<11s}| Test  Acc: {:.4f}'.format(datasets[test_idx], test_acc))
    else:
        for test_idx, test_loader in enumerate(test_loaders): # 如果不是 fedbn 模式，则使用全局模型 server_model 直接进行测试
            _, test_acc = test(server_model, test_loader, loss_fun, device)
            print(' {:<11s}| Test  Acc: {:.4f}'.format(datasets[test_idx], test_acc))
    exit(0)
args.resume=False
if args.resume: # 判断是否启用训练恢复功能
    checkpoint = torch.load(SAVE_PATH) # 加载之前保存的训练模型权重
    server_model.load_state_dict(checkpoint['server_model'])
    if args.mode.lower() == 'fedbn': # 在 fedbn 模式下，分别加载每个客户端的模型权重
        for client_idx in range(client_num):
            models[client_idx].load_state_dict(checkpoint['model_{}'.format(client_idx)])
    else:
        for client_idx in range(client_num): # 在非 fedbn 模式下，所有客户端模型都从服务器端模型 server_model 恢复权重
            models[client_idx].load_state_dict(checkpoint['server_model'])
    resume_iter = int(checkpoint['a_iter']) + 1 # 从保存的检查点 a_iter 获取上次训练的轮次（epoch）
    print('Resume training from epoch {}'.format(resume_iter))
else:
    resume_iter = 0

# 开始训练
for a_iter in range(resume_iter, args.iters):
    optimizers = [optim.Adam(params=models[idx].parameters(), lr=args.lr,weight_decay=0.001) for idx in range(client_num)]
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)
    for wi in range(args.wk_iters):
        print("============ Train epoch {} ============".format(wi + a_iter * args.wk_iters))
        if args.log:
            logfile.write("============ Train epoch {} ============\n".format(wi + a_iter * args.wk_iters))

        for client_idx in range(client_num):
            model, train_loader, optimizer = models[client_idx], train_loaders[client_idx], optimizers[client_idx]
            if args.mode.lower() == 'fedprox':
                if a_iter > 0: # 采用 train_prox() 进行 FedProx 模式的训练
                    train_prox(args, model, train_loader, optimizer, loss_fun, client_num, device)
                else: # 在第一次迭代（a_iter == 0）时，使用常规的 train() 训练
                    train(model, train_loader, optimizer, loss_fun, client_num, device)
            else:
                train(model, train_loader, optimizer, loss_fun, client_num, device)

    # aggregation
    server_model, models = communication(args, server_model, models, client_weights)


    for client_idx in range(client_num):
        model, train_loader, optimizer = models[client_idx], train_loaders[client_idx], optimizers[client_idx]
        train_acc, train_loss = test(model, train_loader, loss_fun, device)
        print(
            ' {:<11s}| Train Acc: {:.4f} | Train Loss: {:.4f}'.format(datasets[client_idx], train_acc, train_loss))
        if args.log:
            logfile.write(
                ' {:<11s}| Train Acc: {:.4f} | Train Loss: {:.4f}\n'.format(datasets[client_idx], train_acc, train_loss))

    # start testing
    for test_idx, test_loader in enumerate(test_loaders):
        test_acc, test_loss = test(models[test_idx], test_loader, loss_fun, device)
        print(' {:<11s}| Test  Acc: {:.4f} | Test  Loss: {:.4f}'.format(datasets[test_idx], test_acc, test_loss))
        if args.log:
            logfile.write(' {:<11s}| Test  Acc: {:.4f} | Test  Loss: {:.4f}\n'.format(datasets[test_idx], test_acc,
                                                                                      test_loss))

# Save checkpoint
print(' Saving checkpoints to {}...'.format(SAVE_PATH))
if args.mode.lower() == 'fedbn':
    torch.save({
        'model_0': models[0].state_dict(),
        'model_1': models[1].state_dict(),
        'model_2': models[2].state_dict(),
        'server_model': server_model.state_dict(),
    }, SAVE_PATH)
else:
    torch.save({
        'server_model': server_model.state_dict(),
    }, SAVE_PATH)

if log:
    logfile.flush()
    logfile.close()