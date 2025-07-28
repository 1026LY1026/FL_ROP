from functions import *
import torch.nn as nn
import torch.optim as optim

from datetime import datetime

from utility_volve import *



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
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(MULTI_GPU)
deviceCount = torch.cuda.device_count()
torch.cuda.set_device(device)
print(deviceCount)
print(device)

bh_train,bh_train_y_max,bh_train_y_min, bh_train_de_max,bh_train_de_min = data_load(volve_5_7_10_12)
bh_test,bh_test_y_max,bh_test_y_min, bh_test_de_max,bh_test_de_min =  data_load(volve_9A)

model=LSTMModel().to(device)

criterion = nn.MSELoss()  # 占位符 索引为0.9


optimizer = optim.Adam(model.parameters(), lr=model_tf_lr, weight_decay=0.001)

# # 定义学习率衰减策略
# scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.2)


def train(TModel, loader):
    epoch_loss = 0
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

def initiate():
    train_acc_size = []
    train_r2_size = []
    train_mse_size = []
    train_mae_size = []
    train_loss_size = []

    test_acc_size = []
    test_r2_size = []
    test_mse_size = []
    test_mae_size = []
    test_loss_size = []
    total_acc = 0
    for epoch in range(1000):
        start = datetime.now()
        model.train()
        train(model, bh_train)
        model.eval()
        train_acc, train_r2, train_mse, train_mae, train_loss, true_train, pre_train, train_depth = test(model,
                                                                                                         bh_train,
                                                                                                         bh_train_y_max, bh_train_y_min,
                                                                                                         bh_train_de_max, bh_train_de_min)
        train_mse_size.append(train_mse)
        train_mae_size.append(train_mae)
        train_acc_size.append(train_acc)
        train_r2_size.append(train_r2)
        train_loss_size.append(train_loss)
        print('Epoch:', 'train %04d' % epoch, 'loss =', '{:.6f}'.format(train_loss), ' acc =',
              '{:.6f}'.format(train_acc), ' r2 =', '{:.6f}'.format(train_r2),
              ' mse =', '{:.6f}'.format(train_mse), ' mae =', '{:.6f}'.format(train_mae), 'time = ', start)

        test_acc, test_r2, test_mse, test_mae, test_loss, true_test, pre_test, test_depth = test(model, bh_test,
                                                                                                 bh_test_y_max, bh_test_y_min,
                                                                                                 bh_test_de_max, bh_test_de_min)
        test_mse_size.append(test_mse)
        test_mae_size.append(test_mae)
        test_acc_size.append(test_acc)
        test_r2_size.append(test_r2)
        test_loss_size.append(test_loss)
        print('Epoch:', 'test %04d' % epoch, 'loss =', '{:.6f}'.format(test_loss), ' acc =', '{:.6f}'.format(test_acc),
              ' r2 =', '{:.6f}'.format(test_r2),
              ' mse =', '{:.6f}'.format(test_mse), ' mae =', '{:.6f}'.format(test_mae), 'time = ', start)

        loss_acc_mse_mae_dict = {'train_loss': train_loss_size, 'test_loss': test_loss_size,
                                 'train_acc': train_acc_size, 'test_acc': test_acc_size,
                                 'train_r2': train_r2_size, 'test_r2': test_r2_size,
                                 'train_mse': train_mse_size, 'train_mae': train_mae_size,
                                 'test_mse': test_mse_size, 'test_mae': test_mae_size, }
        loss_acc_mse_mae = pd.DataFrame(loss_acc_mse_mae_dict)

        train_de = pd.DataFrame(train_depth, columns=['train_depth'])
        train_t = pd.DataFrame(true_train, columns=['train_true'])
        train_p = pd.DataFrame(pre_train, columns=['train_pre'])
        test_de = pd.DataFrame(test_depth, columns=['test_depth'])
        test_t = pd.DataFrame(true_test, columns=['test_true'])
        test_p = pd.DataFrame(pre_test, columns=['test_pre'])

        csv_train = pd.concat([train_de, train_t, train_p], axis=1)
        csv_test = pd.concat([test_de, test_t, test_p], axis=1)

        loss_acc_mse_mae.to_csv('./output/volve/out/loss_acc_mse_mae_.csv', sep=",", index=True)

        torch.save(model.state_dict(), './output/volve/model/Model_volve.pkl')

        csv_train.to_csv('./output/volve/out/rel_pre_train.csv', sep=",", index=True)
        csv_test.to_csv('./output/volve/out/rel_pre_test.csv', sep=",", index=True)

        acc_loss_plot_two(loss_acc_mse_mae['train_loss'], loss_acc_mse_mae['test_loss'], 'loss',
                      './output/volve/out/loss.png')
        acc_loss_plot_two(loss_acc_mse_mae['train_r2'], loss_acc_mse_mae['test_r2'], 'r2',
                      './output/volve/out/acc.png')
        true_test_plot(csv_test['test_depth'], csv_test['test_true'], csv_test['test_pre'], 'test',
                       './output/volve/out/pre_true_test.png')
        true_test_plot(csv_train['train_depth'], csv_train['train_true'], csv_train['train_pre'], 'train',
                        './output/volve/out/pre_true_train.png')


initiate()