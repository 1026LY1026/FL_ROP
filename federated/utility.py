import os
import argparse
import numpy as np
import torch
import pandas as pd

volve_4 = pd.read_csv('./data/volve/volve_4.csv')
volve_5 = pd.read_csv('./data/volve/volve_5.csv')
volve_10 = pd.read_csv('./data/volve/volve_10.csv')
volve_12 = pd.read_csv('./data/volve/volve_12.csv')
volve_14 = pd.read_csv('./data/volve/volve_14.csv')
volve_4_5_10_12 = pd.read_csv('./data/volve/volve_4_5_10_14.csv')
xj_3 =  pd.read_csv('./data/xj/well_3.csv')
xj_2 = pd.read_csv('./data/xj/well_2.csv')
bh_10 = pd.read_csv('./data/bh/bh_10.csv')
bh_6 = pd.read_csv('./data/bh/bh_6.csv')

# 创建一个字典，其中包含所有客户端的训练和测试指标列表
loss_acc_r2_mse_mae_metrics = {
    'client_0': {
        'train': {
            'acc_size': [],'r2_size': [],'mse_size': [],'mae_size': [],'loss_size': []
        },
        'test': {
            'acc_size': [],'r2_size': [],'mse_size': [],'mae_size': [],'loss_size': []
        }
    },
    'client_1': {
        'train': {
            'acc_size': [],'r2_size': [],'mse_size': [],'mae_size': [],'loss_size': []
        },
        'test': {
            'acc_size': [],'r2_size': [],'mse_size': [],'mae_size': [],'loss_size': []
        }
    },
    'client_2': {
        'train': {
            'acc_size': [],'r2_size': [],'mse_size': [],'mae_size': [],'loss_size': []
        },
        'test': {
            'acc_size': [],'r2_size': [],'mse_size': [],'mae_size': [],'loss_size': []
        }
    },
    'server_model': {
        'test_volve': {
            'acc_size': [],'r2_size': [],'mse_size': [],'mae_size': [],'loss_size': []
        },
        'test_xj': {
            'acc_size': [],'r2_size': [],'mse_size': [],'mae_size': [],'loss_size': []
        },
        'test_bh': {
            'acc_size': [],'r2_size': [],'mse_size': [],'mae_size': [],'loss_size': []
        }
    }
}

de_t_p_metrics = {
    'client_0': {
        'train': {'depth': [],'true': [],'pre': []},
        'test': {'depth': [],'r2_size': [],'true': [],'pre': []}
    },
    'client_1': {
        'train': {'depth': [], 'true': [], 'pre': []},
        'test': {'depth': [], 'true': [], 'pre': []}
    },
    'client_2': {
        'train': {'depth': [], 'true': [], 'pre': []},
        'test': {'depth': [], 'true': [], 'pre': []}
    },
    'server_model': {
        'test_volve': {'depth': [], 'true': [], 'pre': []},
        'test_xj': {'depth': [], 'true': [], 'pre': []},
        'test_bh': {'depth': [], 'true': [], 'pre': []},
    },

}


pre_len = 50
seq_len = 300
interval = 5

seed = 1
np.random.seed(seed)  # 设置NumPy的随机种子
torch.manual_seed(seed)  # 设置PyTorch CPU的随机种子
torch.cuda.manual_seed_all(seed)  # 设置PyTorch GPU的随机种子（如果使用GPU）

parser = argparse.ArgumentParser()  # 创建一个ArgumentParser对象，用于解析命令行传入的参数
parser.add_argument('--test', action='store_true', help='test the pretrained model')  # 如果添加该参数，则表示测试预训练的模型。此参数是一个布尔值，若指定则为True
parser.add_argument('--tf_lr', type=float, default=0.00015, help='learning rate')
parser.add_argument('--batch', type=int, default=64, help='batch size')
parser.add_argument('--mode', type=str, default='fedbn',help='fedavg | fedprox | fedbn')  # 择联邦学习的模式，默认值为'fedbn'，可以是fedavg、fedprox或fedbn
parser.add_argument('--mu', type=float, default=1e-2, help='The hyper parameter for fedprox')  # fedprox模式下的超参数（默认为1e-2）
parser.add_argument('--resume', action='store_true',help='resume training from the save path checkpoint')  # 是否从保存的检查点恢复训练。若指定，则为True
args = parser.parse_args()


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
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
print(MULTI_GPU)
deviceCount = torch.cuda.device_count()
torch.cuda.set_device(device)
print(deviceCount)
print(device)

