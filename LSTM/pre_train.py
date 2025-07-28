import copy
from functions import *
import torch.optim as optim
from datetime import datetime
from utility import *

def communication(server_model, models, client_weights):
    with torch.no_grad():
        for key in server_model.state_dict().keys():
            if 'bn' not in key: # 跳过包含批量归一化（Batch Normalization，bn）的参数，因为批量归一化的参数在联邦学习中通常不需要像其他参数一样进行权重聚合
                temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32) # 创建一个与当前参数相同形状和数据类型的零张量，用于存储加权聚合后的参数
                for client_idx in range(client_num):
                    temp += client_weights[client_idx] * models[client_idx].state_dict()[key] # 使用每个客户端的权重和对应模型的参数进行加权累加
                server_model.state_dict()[key].data.copy_(temp) # 将聚合后的参数更新到服务器模型的对应位置
                for client_idx in range(client_num):
                    models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key]) # 将聚合后的参数同步回每个客户端的模型
    return server_model, models # 返回值：返回更新后的 server_model（服务器模型）和 models（客户端模型）

volve_train,volve_train_y_max,volve_train_y_min, volve_train_de_max,volve_train_de_min = data_load(volve_5_7_10_12)
volve_test,volve_test_y_max,volve_test_y_min, volve_test_de_max,volve_test_de_min =  data_load(volve_9A)

xj_train,xj_train_y_max,xj_train_y_min, xj_train_de_max,xj_train_de_min =  data_load(xj_3)
xj_test,xj_test_y_max,xj_test_y_min, xj_test_de_max,xj_test_de_min =  data_load(xj_1)

bh_train,bh_train_y_max,bh_train_y_min, bh_train_de_max,bh_train_de_min =  data_load(bh_7_15)
bh_test,bh_test_y_max,bh_test_y_min, bh_test_de_max,bh_test_de_min =  data_load(bh_2)

train_loaders = [volve_train, xj_train, bh_train]
test_loaders = [volve_test,xj_test,bh_test]

train_y_maxs = [volve_train_y_max,xj_train_y_max,bh_train_y_max]
train_y_mins = [volve_train_y_min,xj_train_y_min,bh_train_y_min]
train_de_maxs = [volve_train_de_max,xj_train_de_max,bh_train_de_max]
train_de_mins = [volve_train_de_min,xj_train_de_min,bh_train_de_min]

test_y_max = [volve_test_y_max,xj_test_y_max,bh_test_y_max]
test_y_min = [volve_test_y_min,xj_test_y_min,bh_test_y_min]
test_de_max = [volve_test_de_max,xj_test_de_max,bh_test_de_max]
test_de_min = [volve_test_de_min,xj_test_de_min,bh_test_de_min]

datasets= ['volve', 'xj', 'bh']
client_num = len(datasets) # 客户端的数量
client_weights = [1 / client_num for i in range(client_num)]

server_model=TransAm().to(device)
models = [copy.deepcopy(server_model).to(device) for idx in range(client_num)]
optimizers = [optim.Adam(params=models[idx].parameters(), lr=model_tf_lr, weight_decay=0.001) for idx in range(client_num)]

for epoch in range(server_model_epoch):
    print('-' * 40, 'server model epoch', epoch, '-' * 40)
    for wi in range(clint_model_epoch):
        print('    ', 'clint model epoch', wi, '    ')
        for client_idx in range(client_num):
            models[client_idx].train()
            train(models[client_idx], train_loaders[client_idx],optimizers[client_idx])

            models[client_idx].eval()
            train_acc,train_r2, train_mse,train_mae,train_loss, true_train, pre_train, train_depth = test(models[client_idx],train_loaders[client_idx],
                                                                                                           train_y_maxs[client_idx],train_y_mins[client_idx],
                                                                                                           train_de_maxs[client_idx],train_de_mins[client_idx])

            test_acc, test_r2, test_mse, test_mae, test_loss, true_test, pre_test, test_depth = test(models[client_idx],test_loaders[client_idx],
                                                                                                     test_y_max[client_idx],test_y_min[client_idx],
                                                                                                     test_de_max[client_idx],test_de_min[client_idx])
            # 获取当前时间
            now = datetime.now()
            print(datasets[client_idx])
            print('  train:loss =','{:.4f}'.format(train_loss), ' acc =', '{:.4f}'.format(train_acc), ' r2 =', '{:.4f}'.format(train_r2), 'time = ', now.strftime("%Y-%m-%d %H:%M:%S"))
            print('  test:loss =','{:.4f}'.format(test_loss), ' acc =', '{:.4f}'.format(test_acc), ' r2 =', '{:.4f}'.format(test_r2), 'time = ', now.strftime("%Y-%m-%d %H:%M:%S"))

            client_key = f'client_{client_idx}'  # 动态生成键名
            loss_acc_r2_mse_mae_metrics[client_key]['train']['acc_size'].append(train_acc)
            loss_acc_r2_mse_mae_metrics[client_key]['train']['r2_size'].append(train_r2)
            loss_acc_r2_mse_mae_metrics[client_key]['train']['mse_size'].append(train_mse)
            loss_acc_r2_mse_mae_metrics[client_key]['train']['mae_size'].append(train_mae)
            loss_acc_r2_mse_mae_metrics[client_key]['train']['loss_size'].append(train_loss)
            loss_acc_r2_mse_mae_metrics[client_key]['test']['acc_size'].append(test_acc)
            loss_acc_r2_mse_mae_metrics[client_key]['test']['r2_size'].append(test_r2)
            loss_acc_r2_mse_mae_metrics[client_key]['test']['mse_size'].append(test_mse)
            loss_acc_r2_mse_mae_metrics[client_key]['test']['mae_size'].append(test_mae)
            loss_acc_r2_mse_mae_metrics[client_key]['test']['loss_size'].append(test_loss)

            de_t_p_metrics[client_key]['train']['depth'] = train_depth
            de_t_p_metrics[client_key]['train']['true'] = true_train
            de_t_p_metrics[client_key]['train']['pre'] = pre_train
            de_t_p_metrics[client_key]['test']['depth'] = test_depth
            de_t_p_metrics[client_key]['test']['true'] = true_test
            de_t_p_metrics[client_key]['test']['pre'] = pre_test

    server_model, models = communication(server_model, models, client_weights)

    for client_idx in range(client_num):
        server_model.eval()
        test_acc, test_r2, test_mse, test_mae, test_loss, true_test, pre_test, test_depth = test(server_model,test_loaders[client_idx],
                                                                                                 test_y_max[client_idx],test_y_min[client_idx],
                                                                                                 test_de_max[client_idx],test_de_min[client_idx])
        client_key = f'test_{datasets[client_idx]}'  # 动态生成键名
        loss_acc_r2_mse_mae_metrics['server_model'][client_key]['acc_size'].append(test_acc)
        loss_acc_r2_mse_mae_metrics['server_model'][client_key]['r2_size'].append(test_r2)
        loss_acc_r2_mse_mae_metrics['server_model'][client_key]['mse_size'].append(test_mse)
        loss_acc_r2_mse_mae_metrics['server_model'][client_key]['mae_size'].append(test_mae)
        loss_acc_r2_mse_mae_metrics['server_model'][client_key]['loss_size'].append(test_loss)

        de_t_p_metrics['server_model'][client_key]['depth'] = test_depth
        de_t_p_metrics['server_model'][client_key]['true'] = true_test
        de_t_p_metrics['server_model'][client_key]['pre'] = pre_test

        # 获取当前时间
        now = datetime.now()

        print('    ', 'server model', '    ')
        print()
        print(datasets[client_idx],'test:loss =', '{:.4f}'.format(test_loss), ' acc =', '{:.4f}'.format(test_acc), ' r2 =','{:.4f}'.format(test_r2),
              ' mse =', '{:.4f}'.format(test_mse), ' mae =', '{:.4f}'.format(test_mae),'time = ', now.strftime("%Y-%m-%d %H:%M:%S"))

    server_volve_loss_acc_mse_mae_dict = {'test_loss': loss_acc_r2_mse_mae_metrics['server_model']['test_volve']['loss_size'],
                                    'test_acc': loss_acc_r2_mse_mae_metrics['server_model']['test_volve']['acc_size'],
                                    'test_r2': loss_acc_r2_mse_mae_metrics['server_model']['test_volve']['r2_size'],
                                    'test_mse': loss_acc_r2_mse_mae_metrics['server_model']['test_volve']['mse_size'],
                                    'test_mae': loss_acc_r2_mse_mae_metrics['server_model']['test_volve']['mae_size'] }
    server_volve_pre_ture_test_dict = {'depth': de_t_p_metrics['server_model']['test_volve']['depth'],
                                 'true': de_t_p_metrics['server_model']['test_volve']['true'],
                                 'pre': de_t_p_metrics['server_model']['test_volve']['pre']}
    server_xj_loss_acc_mse_mae_dict = {'test_loss': loss_acc_r2_mse_mae_metrics['server_model']['test_xj']['loss_size'],
                                    'test_acc': loss_acc_r2_mse_mae_metrics['server_model']['test_xj']['acc_size'],
                                    'test_r2': loss_acc_r2_mse_mae_metrics['server_model']['test_xj']['r2_size'],
                                    'test_mse': loss_acc_r2_mse_mae_metrics['server_model']['test_xj']['mse_size'],
                                    'test_mae': loss_acc_r2_mse_mae_metrics['server_model']['test_xj']['mae_size'] }
    server_xj_pre_ture_test_dict = {'depth': de_t_p_metrics['server_model']['test_xj']['depth'],
                                       'true': de_t_p_metrics['server_model']['test_xj']['true'],
                                       'pre': de_t_p_metrics['server_model']['test_xj']['pre']}
    server_bh_loss_acc_mse_mae_dict = {'test_loss': loss_acc_r2_mse_mae_metrics['server_model']['test_bh']['loss_size'],
                                    'test_acc': loss_acc_r2_mse_mae_metrics['server_model']['test_bh']['acc_size'],
                                    'test_r2': loss_acc_r2_mse_mae_metrics['server_model']['test_bh']['r2_size'],
                                    'test_mse': loss_acc_r2_mse_mae_metrics['server_model']['test_bh']['mse_size'],
                                    'test_mae': loss_acc_r2_mse_mae_metrics['server_model']['test_bh']['mae_size'] }
    server_bh_pre_ture_test_dict = {'depth': de_t_p_metrics['server_model']['test_bh']['depth'],
                                       'true': de_t_p_metrics['server_model']['test_bh']['true'],
                                       'pre': de_t_p_metrics['server_model']['test_bh']['pre']}

    clint0_volve__loss_acc_mse_mae_dict = {'test_loss': loss_acc_r2_mse_mae_metrics['client_0']['test']['loss_size'],
                                'test_acc': loss_acc_r2_mse_mae_metrics['client_0']['test']['acc_size'],
                                'test_r2': loss_acc_r2_mse_mae_metrics['client_0']['test']['r2_size'],
                                'test_mse': loss_acc_r2_mse_mae_metrics['client_0']['test']['mse_size'],
                                'test_mae': loss_acc_r2_mse_mae_metrics['client_0']['test']['mae_size'],
                                'train_loss': loss_acc_r2_mse_mae_metrics['client_0']['train']['loss_size'],
                                'train_acc': loss_acc_r2_mse_mae_metrics['client_0']['train']['acc_size'],
                                'train_r2': loss_acc_r2_mse_mae_metrics['client_0']['train']['r2_size'],
                                'train_mse': loss_acc_r2_mse_mae_metrics['client_0']['train']['mse_size'],
                                'train_mae': loss_acc_r2_mse_mae_metrics['client_0']['train']['mae_size'],
                                }
    clint0_volve_pre_ture_test_dict = {'depth': de_t_p_metrics['client_0']['test']['depth'],
                                    'true': de_t_p_metrics['client_0']['test']['true'],
                                    'pre': de_t_p_metrics['client_0']['test']['pre'] }
    clint1_xj_loss_acc_mse_mae_dict = {'test_loss': loss_acc_r2_mse_mae_metrics['client_1']['test']['loss_size'],
                                'test_acc': loss_acc_r2_mse_mae_metrics['client_1']['test']['acc_size'],
                                'test_r2': loss_acc_r2_mse_mae_metrics['client_1']['test']['r2_size'],
                                'test_mse': loss_acc_r2_mse_mae_metrics['client_1']['test']['mse_size'],
                                'test_mae': loss_acc_r2_mse_mae_metrics['client_1']['test']['mae_size'],
                                'train_loss': loss_acc_r2_mse_mae_metrics['client_1']['train']['loss_size'],
                                'train_acc': loss_acc_r2_mse_mae_metrics['client_1']['train']['acc_size'],
                                'train_r2': loss_acc_r2_mse_mae_metrics['client_1']['train']['r2_size'],
                                'train_mse': loss_acc_r2_mse_mae_metrics['client_1']['train']['mse_size'],
                                'train_mae': loss_acc_r2_mse_mae_metrics['client_1']['train']['mae_size'],
                                }
    clint1_xj_pre_ture_test_dict = {'depth': de_t_p_metrics['client_1']['test']['depth'],
                                    'true': de_t_p_metrics['client_1']['test']['true'],
                                    'pre': de_t_p_metrics['client_1']['test']['pre'] }
    clint2_bh_loss_acc_mse_mae_dict = {'test_loss': loss_acc_r2_mse_mae_metrics['client_2']['test']['loss_size'],
                                'test_acc': loss_acc_r2_mse_mae_metrics['client_2']['test']['acc_size'],
                                'test_r2': loss_acc_r2_mse_mae_metrics['client_2']['test']['r2_size'],
                                'test_mse': loss_acc_r2_mse_mae_metrics['client_2']['test']['mse_size'],
                                'test_mae': loss_acc_r2_mse_mae_metrics['client_2']['test']['mae_size'],
                                'train_loss': loss_acc_r2_mse_mae_metrics['client_2']['train']['loss_size'],
                                'train_acc': loss_acc_r2_mse_mae_metrics['client_2']['train']['acc_size'],
                                'train_r2': loss_acc_r2_mse_mae_metrics['client_2']['train']['r2_size'],
                                'train_mse': loss_acc_r2_mse_mae_metrics['client_2']['train']['mse_size'],
                                'train_mae': loss_acc_r2_mse_mae_metrics['client_2']['train']['mae_size'],
                                }
    clint2_bh_pre_ture_test_dict = {'depth': de_t_p_metrics['client_2']['test']['depth'],
                                    'true': de_t_p_metrics['client_2']['test']['true'],
                                    'pre': de_t_p_metrics['client_2']['test']['pre'] }

    server_volve_loss_acc_mse_mae = pd.DataFrame(server_volve_loss_acc_mse_mae_dict)
    server_volve_pre_ture_test = pd.DataFrame(server_volve_pre_ture_test_dict)
    server_xj_loss_acc_mse_mae = pd.DataFrame(server_xj_loss_acc_mse_mae_dict)
    server_xj_pre_ture_test = pd.DataFrame(server_xj_pre_ture_test_dict)
    server_bh_loss_acc_mse_mae = pd.DataFrame(server_bh_loss_acc_mse_mae_dict)
    server_bh_pre_ture_test = pd.DataFrame(server_bh_pre_ture_test_dict)

    clint0_volve_loss_acc_mse_mae = pd.DataFrame(clint0_volve__loss_acc_mse_mae_dict)
    clint0_volve_pre_ture_test = pd.DataFrame(clint0_volve_pre_ture_test_dict)
    clint1_xj_loss_acc_mse_mae = pd.DataFrame(clint1_xj_loss_acc_mse_mae_dict)
    clint1_xj_pre_ture_test = pd.DataFrame(clint1_xj_pre_ture_test_dict)
    clint2_bh_loss_acc_mse_mae = pd.DataFrame(clint2_bh_loss_acc_mse_mae_dict)
    clint2_bh_pre_ture_test = pd.DataFrame(clint2_bh_pre_ture_test_dict)

    server_volve_loss_acc_mse_mae.to_csv('./output0518/server/volve/server_volve_loss_acc_mse_mae.csv', sep=",", index=True)
    server_volve_pre_ture_test.to_csv('./output0518/server/volve/server_volve_pre_ture_test.csv', sep=",", index=True)
    server_xj_loss_acc_mse_mae.to_csv('./output0518/server/xj/server_xj_loss_acc_mse_mae.csv', sep=",", index=True)
    server_xj_pre_ture_test.to_csv('./output0518/server/xj/server_xj_pre_ture_test.csv', sep=",", index=True)
    server_bh_loss_acc_mse_mae.to_csv('./output0518/server/bh/server_bh_loss_acc_mse_mae.csv', sep=",", index=True)
    server_bh_pre_ture_test.to_csv('./output0518/server/bh/server_bh_pre_ture_test.csv', sep=",", index=True)

    clint0_volve_loss_acc_mse_mae.to_csv('./output0518/client/volve/clint0_volve_loss_acc_mse_mae.csv', sep=",", index=True)
    clint0_volve_pre_ture_test.to_csv('./output0518/client/volve/clint0_volve_pre_ture_test.csv', sep=",", index=True)
    clint1_xj_loss_acc_mse_mae.to_csv('./output0518/client/xj/clint1_xj_loss_acc_mse_mae.csv', sep=",", index=True)
    clint1_xj_pre_ture_test.to_csv('./output0518/client/xj/clint1_xj_pre_ture_test.csv', sep=",", index=True)
    clint2_bh_loss_acc_mse_mae.to_csv('./output0518/client/bh/clint2_bh_loss_acc_mse_mae.csv', sep=",", index=True)
    clint2_bh_pre_ture_test.to_csv('./output0518/client/bh/clint2_bh_pre_ture_test.csv', sep=",", index=True)

    acc_loss_plot_one(server_volve_loss_acc_mse_mae['test_loss'], 'loss','./output0518/server/volve/server_test_loss.png')
    acc_loss_plot_one(server_volve_loss_acc_mse_mae['test_r2'], 'r2', './output0518/server/volve/server_test_r2.png')
    acc_loss_plot_one(server_xj_loss_acc_mse_mae['test_loss'], 'loss','./output0518/server/xj/server_test_loss.png')
    acc_loss_plot_one(server_xj_loss_acc_mse_mae['test_r2'], 'r2', './output0518/server/xj/server_test_r2.png')
    acc_loss_plot_one(server_bh_loss_acc_mse_mae['test_loss'], 'loss','./output0518/server/bh/server_test_loss.png')
    acc_loss_plot_one(server_bh_loss_acc_mse_mae['test_r2'], 'r2', './output0518/server/bh/server_test_r2.png')

    acc_loss_plot_two(clint0_volve_loss_acc_mse_mae['train_r2'], clint0_volve_loss_acc_mse_mae['test_r2'], 'r2','./output0518/client/volve/clint0_volve_r2.png')
    acc_loss_plot_two(clint0_volve_loss_acc_mse_mae['train_loss'], clint0_volve_loss_acc_mse_mae['test_loss'], 'r2','./output0518/client/volve/clint0_volve_loss.png')
    acc_loss_plot_two(clint1_xj_loss_acc_mse_mae['train_r2'], clint1_xj_loss_acc_mse_mae['test_r2'], 'r2','./output0518/client/xj/clint1_xj_r2.png')
    acc_loss_plot_two(clint1_xj_loss_acc_mse_mae['train_loss'], clint1_xj_loss_acc_mse_mae['test_loss'], 'r2','./output0518/client/xj/clint1_xj_loss.png')
    acc_loss_plot_two(clint2_bh_loss_acc_mse_mae['train_r2'], clint2_bh_loss_acc_mse_mae['test_r2'], 'r2','./output0518/client/bh/clint2_bh_r2.png')
    acc_loss_plot_two(clint2_bh_loss_acc_mse_mae['train_loss'], clint2_bh_loss_acc_mse_mae['test_loss'], 'r2','./output0518/client/bh/clint2_bh_loss.png')

    true_test_plot(server_volve_pre_ture_test['depth'], server_volve_pre_ture_test['true'], server_volve_pre_ture_test['pre'], 'test',
                   './output0518/server/volve/server_volve_pre_ture_test.png')
    true_test_plot(server_xj_pre_ture_test['depth'], server_xj_pre_ture_test['true'], server_xj_pre_ture_test['pre'], 'test',
                   './output0518/server/xj/server_xj_pre_ture_test.png')
    true_test_plot(server_bh_pre_ture_test['depth'], server_bh_pre_ture_test['true'], server_bh_pre_ture_test['pre'], 'test',
                   './output0518/server/bh/server_bh_pre_ture_test.png')
    true_test_plot(clint0_volve_pre_ture_test['depth'], clint0_volve_pre_ture_test['true'], clint0_volve_pre_ture_test['pre'], 'test',
                   './output0518/client/volve/clint0_volve_pre_ture_test.png')
    true_test_plot(clint1_xj_pre_ture_test['depth'], clint1_xj_pre_ture_test['true'], clint1_xj_pre_ture_test['pre'], 'test',
                   './output0518/client/xj/clint1_xj_pre_ture_test.png')
    true_test_plot(clint2_bh_pre_ture_test['depth'], clint2_bh_pre_ture_test['true'], clint2_bh_pre_ture_test['pre'], 'test',
                   './output0518/client/bh/clint2_bh_pre_ture_test.png')

    torch.save(models[0].state_dict(), './output0518/model/model_0_volve.pkl')
    torch.save(models[1].state_dict(), './output0518/model/model_1_xj.pkl')
    torch.save(models[2].state_dict(), './output0518/model/model_2_bh.pkl')
    print('save clint model')
    torch.save(server_model.state_dict(), './output0518/model/server_model.pkl')
    print('save server_model')
