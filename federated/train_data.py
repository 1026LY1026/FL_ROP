from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from utility import *
import pandas as pd
import numpy as np

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



