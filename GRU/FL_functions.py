import math
import warnings
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.optim import lr_scheduler
from sklearn.metrics import r2_score,mean_squared_error, mean_absolute_error
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from pylab import mpl
from utility import *
warnings.filterwarnings('ignore')

# 设置matplotlib的配置
# mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

class GRUModel(nn.Module):
    def __init__(self, feature_size=model_feature_size, hidden_size=model_d_model, num_layers=model_num_layers, dropout=model_dropout, device='cuda'):
        super(GRUModel, self).__init__()

        self.embedding = nn.Linear(feature_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, src):
        # src: [batch_size, seq_len, feature_size]
        # 输入embedding
        src = self.embedding(src)  # [batch_size, seq_len, hidden_size]

        # GRU层
        gru_out, _ = self.gru(src)  # gru_out: [batch_size, seq_len, hidden_size]

        # 输出层
        output = self.linear(gru_out)  # [batch_size, seq_len, out_size]
        output_squeeze = output.squeeze()
        return output_squeeze

def data_load(path):

    data = pd.read_csv(path)
    # data = data[['MD', 'TVD', 'RPMA', 'WOBA', 'ROPA']]
   # data = data.iloc[::interval, :]

    # data = data.clip(lower=0)  # 设置小于0的数都赋0
    # data = data.apply(lambda x: x.mask((x < x.quantile(0.25) - 1.5 * (x.quantile(0.75) - x.quantile(0.25))) |
    #                                      (x > x.quantile(0.75) + 1.5 * (
    #                                                  x.quantile(0.75) - x.quantile(0.25)))).ffill().bfill())
    #
    # data = data.sort_values(by='MD')
    # data=data.reset_index(drop=True)
    data = data.astype('float32')
    data.dropna(inplace=True)
    data = data.values

    data_ =torch.tensor(data[:len(data)])
    maxc, _ = data_.max(dim=0)
    minc, _ = data_.min(dim=0)
    y_max = maxc[-1]
    y_min = minc[-1]
    de_max = maxc[0]
    de_min = minc[0]
    data_ = (data_ - minc) / (maxc - minc)

    data_last_index = data_.shape[0] - model_seq_len

    data_X = []
    data_Y = []
    for i in range(0, data_last_index - model_pre_len+1):
        data_x = np.expand_dims(data_[i:i + model_seq_len], 0)  # [1,seq,feature_size]
        data_y = np.expand_dims(data_[i + model_seq_len:i + model_seq_len + model_pre_len], 0)  # [1,seq,out_size]
        data_X.append(data_x)
        data_Y.append(data_y)

    data_X=np.concatenate(data_X, axis=0)
    data_Y = np.concatenate(data_Y, axis=0)

    process_data = torch.from_numpy(data_X).type(torch.float32)
    process_label = torch.from_numpy(data_Y).type(torch.float32)

    data_feature_size = process_data.shape[-1]

    dataset_train = TensorDataset(process_data, process_label)

    data_dataloader = DataLoader(dataset_train, batch_size=model_batch, shuffle=False)
    return data_dataloader,y_max,y_min, de_max,de_min

def train(TModel, loader,optimizer):
    epoch_loss = 0
    criterion = nn.MSELoss()  # 占位符 索引为0.9
    for X, y in loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        # 使用LSTM模型进行前向传播
        output = TModel(X)  # output 512 300 1
        # 计算损失
        pre_out=output[:,-model_pre_len:]
        pre_y=y[:, -model_pre_len:, -1]
        loss = criterion(pre_out, pre_y )
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(TModel.parameters(), 0.10)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss

def test(TModel, tf_loader, y_max, y_min, de_max, de_min):
    epoch_loss = 0
    y_pre = []
    y_true = []
    y_depth = []
    criterion = nn.MSELoss()  # 占位符 索引为0.9
    for x, y in tf_loader:
        with torch.no_grad():
            label = y[:, -model_pre_len:, -1].detach().reshape(1, len(y[:, -model_pre_len:, -1]) * model_pre_len).squeeze()
            label = label * (y_max - y_min) + y_min
            label = label.numpy().tolist()
            y_true += label

            de = y[:, -model_pre_len:, 0].detach().reshape(1, len(y[:, -model_pre_len:, 0]) * model_pre_len).squeeze()
            de = de * (de_max - de_min) + de_min
            de = de.numpy().tolist()
            y_depth += de

            x, y = x.to(device), y.to(device)

            output = TModel(x)

            pre_out = output[:, -model_pre_len:]
            loss = criterion(pre_out, y[:, -model_pre_len:, -1])

            epoch_loss += loss.item()

            hat = pre_out.cpu().detach().reshape(1, len(y[:, -model_pre_len:, -1]) * model_pre_len).squeeze()
            hat = hat * (y_max - y_min) + y_min
            hat = hat.numpy().tolist()
            y_pre += hat


    label = np.array(y_true)
    predict = np.array(y_pre)
    dep = np.array(y_depth)

    seq_label = label.reshape(int(len(label) / model_pre_len), model_pre_len)
    seq_predict = predict.reshape(int(len(predict) / model_pre_len), model_pre_len)
    seq_depth = dep.reshape(int(len(dep) / model_pre_len), model_pre_len)

    true = np.concatenate((seq_label[:-1, 0], seq_label[-1, :]), axis=0)
    depth = np.concatenate((seq_depth[:-1, 0], seq_depth[-1, :]), axis=0)
    pre = averages(seq_predict)

    r2 = r2_score(true, pre)
    acc = 1 - (np.abs(pre - true) / (true + 1e-8)).mean()
    mse = mean_squared_error(true, pre)

    mae = mean_absolute_error(true, pre)

    return acc, r2, mse, mae, epoch_loss, true, pre, depth

def averages(matrix):  # 计算平均值
    matrix = np.array(matrix)
    row_count, col_count = matrix.shape
    max_diagonal = row_count + col_count - 1
    diagonals = np.zeros(max_diagonal)
    counts = np.zeros(max_diagonal, dtype=int)
    for i in range(row_count):
        for j in range(col_count):
            num = matrix[i, j]
            diagonal_index = i + j
            diagonals[diagonal_index] += num
            counts[diagonal_index] += 1
    averages = diagonals / counts
    return averages

def acc_loss_plot_one(train_data, type_, path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_data, label='train_data', color='blue', linewidth=1)
    plt.xlabel('epoch', fontsize=18)
    plt.title(f'train_test_{type_}')
    path_ = f'{path}'

    plt.grid()
    plt.savefig(path_)

    plt.legend()
   # plt.show()

def acc_loss_plot_two(train_data, test_data, type_, path):
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