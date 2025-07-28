import math
import warnings
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from torch.utils.data import TensorDataset,DataLoader
from pylab import mpl
import os
import torch
import pandas as pd
warnings.filterwarnings('ignore')
# 设置matplotlib的配置
# mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

#model_pre_len = 50
model_seq_len = 300
#model_tf_lr = 0.00015
model_batch = 128
model_feature_size=5
model_d_model=512
model_num_layers=1
model_dropout=0

USE_MULTI_GPU = True
# 设置默认的CUDA设备
torch.cuda.set_device(0)
# 初始化CUDA环境
torch.cuda.init()
if USE_MULTI_GPU and torch.cuda.device_count() > 1:
    MULTI_GPU = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"  # 设置所有六张显卡的编号
    device_ids = ['0','1','2','3','4','5',] # 设置所有六张显卡的编号
else:
    MULTI_GPU = False
    device_ids = ['0']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(MULTI_GPU)
deviceCount = torch.cuda.device_count()
torch.cuda.set_device(device)
print(deviceCount)
print(device)


volve_4 = './data/volve/volve_4.csv'
volve_5 = './data/volve/volve_5.csv'
volve_7 = './data/volve/volve_7.csv'
volve_9 = './data/volve/volve_9.csv'
volve_9A = './data/volve/volve_9A.csv'
volve_10 = './data/volve/volve_10.csv'
volve_12 = './data/volve/volve_12.csv'
volve_14 = './data/volve/volve_14.csv'
volve_15A = './data/volve/volve_15A.csv'
volve_4_5_7_9A_10 = './data/volve/volve_4_5_7_9A_10.csv'
volve_5_7_10_12 = './data/volve/volve_5_7_10_12.csv'
xj_3 =  './data/xj/well_3.csv'
xj_2 = './data/xj/well_2.csv'
xj_1 = './data/xj/well_1.csv'

bh_1 = './data/bh/bh_1.csv'
bh_2 = './data/bh/bh_2.csv'
bh_3 = './data/bh/bh_3.csv'
bh_4 = './data/bh/bh_4.csv'
bh_5 = './data/bh/bh_5.csv'
bh_6 = './data/bh/bh_6.csv'
bh_7 = './data/bh/bh_7.csv'
bh_8 = './data/bh/bh_8.csv'
bh_9 = './data/bh/bh_9.csv'
bh_10 = './data/bh/bh_10.csv'
bh_11 = './data/bh/bh_11.csv'
bh_12 = './data/bh/bh_12.csv'
bh_14 = './data/bh/bh_14.csv'
bh_15 = './data/bh/bh_15.csv'
bh_16 = './data/bh/bh_16csv'
bh_7_15 = './data/bh/bh_7_15.csv'

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)  # 64*512
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # 64*1
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # 256   model/2
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # pe.requires_grad = False
        self.register_buffer('pe', pe)  # 64*1*512

    def forward(self, x):  # [seq,batch,d_model]
        return x + self.pe[:x.size(0), :]  # 64*64*512

class TransAm(nn.Module):
    def __init__(self, feature_size=model_feature_size, d_model=model_d_model, num_layers=model_num_layers,
                 dropout=model_dropout):
        super(TransAm, self).__init__()
        self.feature_size = feature_size
        self.model_type = 'Transformer'
        self.embedding = nn.Linear(feature_size, d_model)
        self.dec_input_fc = nn.Linear(feature_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)  # 50*512
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=8, dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(d_model, 1)
        self.src_mask = None
        self.src_key_padding_mask = None
        # 添加 LSTM 层
       # self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=1, batch_first=True)

    def forward(self, src, tgt, tgt_mask):
        # if self.src_key_padding_mask is None:
        #     mask_key = src_padding  # [batch,seq]
        #     self.src_key_padding_mask = mask_key
        src_em = self.embedding(src)  # [seq,batch,d_model]
        src_em_pos = self.pos_encoder(src_em)  # [seq,batch,d_model]
        encoder_output = self.transformer_encoder(src_em_pos)
        # 在 Transformer 编码器输出后添加 LSTM
      #  lstm_out, _ = self.lstm(encoder_output)  # lstm_out: [batch, seq, d_model]

        tgt_em = self.embedding(tgt)
        tgt_em_pos = self.pos_encoder(tgt_em)

       # decoder_output = self.transformer_decoder(tgt_em_pos, lstm_out, tgt_mask=tgt_mask)
        decoder_output = self.transformer_decoder(tgt_em_pos, encoder_output, tgt_mask=tgt_mask)
        output = self.linear(decoder_output)
        output_squeeze = output.squeeze()

        self.tgt_mask = None
        return output_squeeze

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

# 定义保存路径和模型参数
model_pre_len_values = [1, 50, 100, 150, 200, 250, 300]
base_path = "./output0518/pre50_compare_dan/bh/test/300_{}"
model_path = "./output0518/pre50_compare_dan/bh/model/Model_bh.pkl"
# base_path = "./output0518/compare/volve2/test/300_{}"
# model_path = "./output0518/compare/volve2/model/Model_volve.pkl"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# 创建汇总结果的字典
summary_results = {
    "model_pre_len": [],
    "R2": [],
    "mse": [],
    "mae": [],
    "acc": [],
    "mre": []
}

# 定义测试函数
def initiate(model_pre_len):
    # 更新保存路径
    save_path = base_path.format(model_pre_len)
    os.makedirs(save_path, exist_ok=True)

    # 初始化测试指标
    test_acc_size = []
    test_r2_size = []
    test_mse_size = []
    test_mae_size = []
    test_loss_size = []
    test_mre_size = []

    start = pd.datetime.now()

    model.eval()

    test_acc, test_r2, test_mse, test_mae, test_mre, true_test, pre_test, test_depth = test(model, data_dataloader,
                                                                                            y_max,
                                                                                            y_min, de_max, de_min)
    test_mse_size.append(test_mse)
    test_mae_size.append(test_mae)
    test_acc_size.append(test_acc)
    test_r2_size.append(test_r2)
    test_mre_size.append(test_mre)
    print(' acc =', '{:.6f}'.format(test_acc), ' r2 =', '{:.6f}'.format(test_r2), ' mre =', '{:.6f}'.format(test_mre),
          ' mse =', '{:.6f}'.format(test_mse), ' mae =', '{:.6f}'.format(test_mae), 'time = ', start)

    acc_mse_mae_dict = {'test_acc': test_acc_size, 'test_r2': test_r2_size,
                        'test_mse': test_mse_size, 'test_mae': test_mae_size, 'test_mre': test_mre_size, }
    acc_mse_mae = pd.DataFrame(acc_mse_mae_dict)

    test_de = pd.DataFrame(test_depth, columns=['test_depth'])
    test_t = pd.DataFrame(true_test, columns=['test_true'])
    test_p = pd.DataFrame(pre_test, columns=['test_pre'])

    csv_test = pd.concat([test_de, test_t, test_p], axis=1)

    acc_mse_mae.to_csv(os.path.join(save_path, "acc_mse_mae_bh11.csv"), sep=",", index=True)
    csv_test.to_csv(os.path.join(save_path, "rel_pre_test_bh11.csv"), sep=",", index=True)

    # 绘制测试结果图
    true_test_plot(csv_test['test_depth'], csv_test['test_true'], csv_test['test_pre'], 'test',
                   os.path.join(save_path, "bh11.png"))

    # 将结果汇总到字典中
    summary_results["model_pre_len"].append(model_pre_len)
    summary_results["R2"].append(test_r2)
    summary_results["mse"].append(test_mse)
    summary_results["mae"].append(test_mae)
    summary_results["acc"].append(test_acc)
    summary_results["mre"].append(test_mre)

# 主循环：运行不同的 model_pre_len
for model_pre_len in model_pre_len_values:
    def test(TModel, tf_loader, y_max, y_min, de_max, de_min):
        y_pre = []
        y_true = []
        y_depth = []

        for x, y in tf_loader:
            with torch.no_grad():
                label = y[:, :, -1].detach().view(1, len(y[:, :, -1]) * model_pre_len).squeeze()
                label = label * (y_max - y_min) + y_min
                label = label.numpy().tolist()
                y_true += label

                de = y[:, :, 0].detach().view(1, len(y[:, :, 0]) * model_pre_len).squeeze()
                de = de * (de_max - de_min) + de_min
                de = de.numpy().tolist()
                y_depth += de

                x, y = x.to(device), y.to(device)

                mask = (torch.triu(torch.ones(y.size(1), y.size(1))) == 1).transpose(0, 1)
                tgt_mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to(
                    device)

                output = TModel(x, y, tgt_mask)

                hat = output.cpu().detach().view(1, len(y[:, :, -1]) * model_pre_len).squeeze()
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
        mape = mean_absolute_percentage_error(true, pre)
        mre = mape / 100
        return acc, r2, mse, mae, mre, true, pre, depth

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

        data_ = torch.tensor(data[:len(data)])
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
        for i in range(0, data_last_index - model_pre_len + 1):
            data_x = np.expand_dims(data_[i:i + model_seq_len], 0)  # [1,seq,feature_size]
            data_y = np.expand_dims(data_[i + model_seq_len:i + model_seq_len + model_pre_len], 0)  # [1,seq,out_size]
            data_X.append(data_x)
            data_Y.append(data_y)

        data_X = np.concatenate(data_X, axis=0)
        data_Y = np.concatenate(data_Y, axis=0)

        process_data = torch.from_numpy(data_X).type(torch.float32)
        process_label = torch.from_numpy(data_Y).type(torch.float32)

        data_feature_size = process_data.shape[-1]

        dataset_train = TensorDataset(process_data, process_label)

        data_dataloader = DataLoader(dataset_train, batch_size=model_batch, shuffle=False)
        return data_dataloader, y_max, y_min, de_max, de_min


    # 加载数据
    data_dataloader, y_max, y_min, de_max, de_min = data_load(bh_11)

    # 初始化模型
    model = TransAm().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    criterion = nn.MSELoss()

    #    plt.show()
    print(f"Running with model_pre_len = {model_pre_len}")
    initiate(model_pre_len)

# 将汇总结果保存到TXT文件
summary_df = pd.DataFrame(summary_results)
summary_df.set_index("model_pre_len", inplace=True)
summary_df.to_csv("./output0518/pre50_compare_dan/bh/test/summary_results_bh11.csv", sep="\t", index=True, float_format="%.6f")

print("All experiments completed and results saved.")