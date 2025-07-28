import os
import torch
import pandas as pd

model_pre_len = 50
model_seq_len = 300
model_tf_lr = 0.00078
model_batch = 128
model_feature_size=5
model_d_model=512
model_num_layers=1
model_dropout=0

# USE_MULTI_GPU = True
# # 设置默认的CUDA设备
# torch.cuda.set_device(0)
# # 初始化CUDA环境
# torch.cuda.init()
# if USE_MULTI_GPU and torch.cuda.device_count() > 1:
#     MULTI_GPU = True
#     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"  # 设置所有六张显卡的编号
#     device_ids = ['0','1','2','3','4','5',] # 设置所有六张显卡的编号
# else:
#     MULTI_GPU = False
#     device_ids = ['0']
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(MULTI_GPU)
# deviceCount = torch.cuda.device_count()
# torch.cuda.set_device(device)
# print(deviceCount)
# print(device)

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


