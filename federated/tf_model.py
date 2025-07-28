import math
import pandas as pd
import torch.nn as nn
from torch.optim import lr_scheduler
from sklearn.metrics import r2_score,mean_squared_error, mean_absolute_error
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from utility import *


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
    def __init__(self, feature_size=12, d_model=512, num_layers=1, dropout=0.1):
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

    def forward(self, src,tgt,tgt_mask):

        # if self.src_key_padding_mask is None:
        #     mask_key = src_padding  # [batch,seq]
        #     self.src_key_padding_mask = mask_key
        src_em = self.embedding(src)  # [seq,batch,d_model]
        src_em_pos = self.pos_encoder(src_em)  # [seq,batch,d_model]
        encoder_output = self.transformer_encoder(src_em_pos)

        tgt_em = self.embedding(tgt)
        tgt_em_pos = self.pos_encoder(tgt_em)

        decoder_output = self.transformer_decoder(tgt_em_pos, encoder_output, tgt_mask=tgt_mask)

        output = self.linear(decoder_output)
        output_squeeze = output.squeeze()

        self.tgt_mask = None
        return output_squeeze

def train(TModel, loader,optimizer):
    epoch_loss = 0
    criterion = nn.MSELoss()  # 占位符 索引为0.9

    # 定义学习率衰减策略
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.2)
    for X, y in loader:
        # X--[batch,seq,feature_size]  y--[batch,seq,feature_size]   64 300 13  64 50 13
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        mask = (torch.triu(torch.ones(y.size(1), y.size(1))) == 1).transpose(0, 1)
        tgt_mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to(device)

        output = TModel(X, y, tgt_mask)
        loss = criterion(output, y[:, :, -1])
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
            label = y[:, :, -1].detach().view(1, len(y[:, :, -1]) * pre_len).squeeze()
            label = label * (y_max - y_min) + y_min
            label = label.numpy().tolist()
            y_true += label

            de = y[:, :, 0].detach().view(1, len(y[:, :, 0]) * pre_len).squeeze()
            de = de * (de_max - de_min) + de_min
            de = de.numpy().tolist()
            y_depth += de

            x, y = x.to(device), y.to(device)

            mask = (torch.triu(torch.ones(y.size(1), y.size(1))) == 1).transpose(0, 1)
            tgt_mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to(device)

            output = TModel(x, y, tgt_mask)

            loss = criterion(output, y[:, :, -1])
            epoch_loss += loss.item()

            hat = output.cpu().detach().view(1, len(y[:, :, -1]) * pre_len).squeeze()
            hat = hat * (y_max - y_min) + y_min
            hat = hat.numpy().tolist()
            y_pre += hat

    label = np.array(y_true)
    predict = np.array(y_pre)
    dep = np.array(y_depth)

    seq_label = label.reshape(int(len(label) / pre_len), pre_len)
    seq_predict = predict.reshape(int(len(predict) / pre_len), pre_len)
    seq_depth = dep.reshape(int(len(dep) / pre_len), pre_len)

    true = np.concatenate((seq_label[:-1, 0], seq_label[-1, :]), axis=0)
    depth = np.concatenate((seq_depth[:-1, 0], seq_depth[-1, :]), axis=0)
    pre = averages(seq_predict)

    r2 = r2_score(true, pre)
    acc = 1 - (np.abs(pre - true) / (true + 1e-8)).mean()
    mse = mean_squared_error(true, pre)

    mae = mean_absolute_error(true, pre)

    return acc, r2, mse, mae, epoch_loss, true, pre, depth

def data_load(path):

    data = pd.read_csv(path)
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

    data_last_index = data_.shape[0] - seq_len

    data_X = []
    data_Y = []
    for i in range(0, data_last_index - pre_len+1):
        data_x = np.expand_dims(data_[i:i + seq_len], 0)  # [1,seq,feature_size]
        data_y = np.expand_dims(data_[i + seq_len:i + seq_len + pre_len], 0)  # [1,seq,out_size]
        data_X.append(data_x)
        data_Y.append(data_y)

    data_X=np.concatenate(data_X, axis=0)
    data_Y = np.concatenate(data_Y, axis=0)

    process_data = torch.from_numpy(data_X).type(torch.float32)
    process_label = torch.from_numpy(data_Y).type(torch.float32)

    data_feature_size = process_data.shape[-1]

    dataset_train = TensorDataset(process_data, process_label)

    data_dataloader = DataLoader(dataset_train, batch_size=args.batch, shuffle=False)
    return data_dataloader,y_max,y_min, de_max,de_min

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