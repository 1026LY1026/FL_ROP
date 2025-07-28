# coding:utf-8
import os
from torch import nn, optim
import time
import copy
from nets.models import lstm_uni_attention
import warnings
warnings.filterwarnings("ignore", category=Warning)
from federated import data_preprocess
from federated.data_preprocess import *
from torch.utils.data import DataLoader

"""
 Parameter Description:                                            参数说明：
Path: the path where the dataset is located                             path：数据集所在的路径
seq_ Length: the length of the time slice                               seq_length：时间片的长度
pre_ Length: the predicted length of the model                          pre_length：模型预测长度
train_ Ratio: proportion of training set                                train_ratio：训练集占比
batch_ Size: batch size                                                 batch_size：批量大小
features_ Num: number of features                                       features_num：特征个数
"""

seq_length = 100
pre_length = 10
train_ratio = 0.8
batch_size = 64
features_num = 12
# 划分数据集X Y
x_data_4, y_data_4 = data_preprocess.data_load(data_path_4, False, elements=elements)
x_data_9, y_data_9 = data_preprocess.data_load(data_path_9, False, elements=elements)
x_data_15A, y_data_15A = data_preprocess.data_load(data_path_15A, False, elements=elements)
x_data_9A, y_data_9A = data_preprocess.data_load(data_path_9A, False, elements=elements)
# 做成时序数据集
data1 = data_preprocess.make_time_series_data(seq_length, pre_length, y_data_4, x_data_4, train_ratio)
data2 = data_preprocess.make_time_series_data(seq_length, pre_length, y_data_9, x_data_9, train_ratio)
data3 = data_preprocess.make_time_series_data(seq_length, pre_length, y_data_15A, x_data_15A, train_ratio)
test_data = data_preprocess.make_time_series_data(seq_length, pre_length, y_data_9A, x_data_9A, train_ratio)
# a=len(train_data1)
# a=len(test_data1)
# print("训练集1的长度:{}".format(a))
# print(train_data2)
# dataloader 加载数据集
train_data1_loader = torch.utils.data.DataLoader(data1, batch_size=batch_size,
                                         shuffle=True, num_workers=0)
train_data2_loader = torch.utils.data.DataLoader(data2, batch_size=batch_size,
                                         shuffle=True, num_workers=0)
train_data3_loader = torch.utils.data.DataLoader(data3, batch_size=batch_size,
                                         shuffle=True, num_workers=0)
test_data_loader = torch.utils.data.DataLoader(test_data,batch_size=batch_size,
                                         shuffle=True, num_workers=0)
# test_data2_loader = torch.utils.data.DataLoader(test_data2,batch_size=batch_size,
#                                          shuffle=True, num_workers=0)
# test_data3_loader = torch.utils.data.DataLoader(test_data3,batch_size=batch_size,
#                                          shuffle=True, num_workers=0)

train_loaders = [train_data1_loader, train_data2_loader, train_data3_loader]
test_loaders = [test_data_loader]
# 测试 DataLoader
# for x,y in train_data1_loader:
    # print(y.size(0))
    # print(x.shape)
       # torch.Size([64, 100, 12])
    # print(y.shape)
    # torch.Size([64, 50])

# 定义训练方法
def train(model, train_loader, optimizer, loss_fun, client_num, device,lr= 0.00015):
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
        y = y.to(device).float()
        y_pre = model(x)
        loss = criterion(y_pre, y)

        loss.backward()
        optimizer.step()

        acc_=np.vstack((acc_,y_pre.cpu().detach().numpy()))
        y_=np.vstack((y_,y.detach().numpy()))
    mse_loss = ((acc_ - y_) ** 2).mean()
    acc_ = acc_[:, 0]
    y_ = y_[:, 0]

    index_ = np.where(y_ == 0)
    y_ = np.delete(y_, index_)
    acc_ = np.delete(acc_, index_)
    acc=1 - (np.abs(acc_ - y_) / y_).mean()
    return acc

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
    with torch.no_grad():
        # aggregate params
        if args.mode.lower() == 'fedbn':
            for key in server_model.state_dict().keys():
                if 'bn' not in key:
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                    for client_idx in range(client_num):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(client_num):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
        else:
            for key in server_model.state_dict().keys():
                # num_batches_tracked is a non trainable LongTensor and
                # num_batches_tracked are the same for all clients for the given datasets
                if 'num_batches_tracked' in key:
                    server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
                else:
                    temp = torch.zeros_like(server_model.state_dict()[key])
                    for client_idx in range(len(client_weights)):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(len(client_weights)):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
    return server_model, models

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed= 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print('Device:', device)
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', default=True, help='whether to make a log')
    parser.add_argument('--test', action='store_true', help='test the pretrained model')
    # parser.add_argument('--percent', type = float, default= 0.1, help ='percentage of dataset to train')
    parser.add_argument('--lr', type=float, default=0.00015, help='learning rate')
    parser.add_argument('--batch', type=int, default=64, help='batch size')
    parser.add_argument('--iters', type=int, default=1, help='iterations for communication')
    parser.add_argument('--wk_iters', type=int, default=1,help='optimization iters in local worker between communication')
    parser.add_argument('--mode', type=str, default='fedbn', help='fedavg | fedprox | fedbn')
    parser.add_argument('--mu', type=float, default=1e-2, help='The hyper parameter for fedprox')
    parser.add_argument('--save_path', type=str, default='../checkpoint/oil', help='path to save the checkpoint')
    parser.add_argument('--resume', action='store_true', help='resume training from the save path checkpoint')
    args = parser.parse_args()

exp_folder = 'federated_oil'

args.save_path = os.path.join(args.save_path, exp_folder)

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
datasets = ['x_data_4', 'x_data_9', 'x_data_15A']

# federated setting
client_num = len(datasets)
client_weights = [1 / client_num for i in range(client_num)]
models = [copy.deepcopy(server_model).to(device) for idx in range(client_num)]

if args.test:
    print('Loading snapshots...')
    checkpoint = torch.load('../snapshots/digits/{}'.format(args.mode.lower()))
    server_model.load_state_dict(checkpoint['server_model'])
    if args.mode.lower() == 'fedbn':
        for client_idx in range(client_num):
            models[client_idx].load_state_dict(checkpoint['model_{}'.format(client_idx)])
        for test_idx, test_loader in enumerate(test_loaders):
            _, test_acc = test(models[test_idx], test_loader, loss_fun, device)
            print(' {:<11s}| Test  Acc: {:.4f}'.format(datasets[test_idx], test_acc))
    else:
        for test_idx, test_loader in enumerate(test_loaders):
            _, test_acc = test(server_model, test_loader, loss_fun, device)
            print(' {:<11s}| Test  Acc: {:.4f}'.format(datasets[test_idx], test_acc))
    exit(0)
# args.resume=True
if args.resume:
    checkpoint = torch.load(SAVE_PATH)
    server_model.load_state_dict(checkpoint['server_model'])
    if args.mode.lower() == 'fedbn':
        for client_idx in range(client_num):
            models[client_idx].load_state_dict(checkpoint['model_{}'.format(client_idx)])
    else:
        for client_idx in range(client_num):
            models[client_idx].load_state_dict(checkpoint['server_model'])
    resume_iter = int(checkpoint['a_iter']) + 1
    print('Resume training from epoch {}'.format(resume_iter))
else:
    resume_iter = 0

# 开始训练
    maxacc=[0,0,0]
    for a_iter in range(resume_iter, args.iters):
        optimizers = [optim.Adam(params=models[idx].parameters(), lr=args.lr,weight_decay=0.001) for idx in range(client_num)]
        # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)
        for wi in range(args.wk_iters):
            print("============ Train epoch {} ============".format(wi + a_iter * args.wk_iters))
            if args.log: logfile.write("============ Train epoch {} ============\n".format(wi + a_iter * args.wk_iters))

            for client_idx in range(client_num):
                model, train_loader, optimizer = models[client_idx], train_loaders[client_idx], optimizers[client_idx]

                if args.mode.lower() == 'fedprox':
                    if a_iter > 0:
                        train_prox(args, model, train_loader, optimizer, loss_fun, client_num, device)
                    else:
                        train(model, train_loader, optimizer, loss_fun, client_num, device)
                else:
                    acc=train(model, train_loader, optimizer, loss_fun, client_num, device)
                    if(acc>maxacc[client_idx]):
                        maxacc[client_idx]=acc
                    else:
                        model=models[client_idx]
                    #     server_model, models = communication(args, server_model, models, client_weights)
        # aggregation
        server_model, models = communication(args, server_model, models, client_weights)

        # report after aggregation
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