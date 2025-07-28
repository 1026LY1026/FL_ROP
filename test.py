from test_functions import *
from datetime import datetime


data_dataloader,y_max,y_min, de_max,de_min = data_load(xj_1)

model=TransAm().to(device)
model_Path = "./out1/model/server_model.pkl"
model.load_state_dict(torch.load(model_Path, map_location=torch.device(device)))
criterion = nn.MSELoss()  # 忽略 占位符 索引为0.9


def initiate():
    test_acc_size = []
    test_r2_size = []
    test_mse_size = []
    test_mae_size = []
    test_loss_size = []

    start = datetime.now()

    model.eval()

    test_acc, test_r2, test_mse, test_mae, true_test, pre_test, test_depth = test(model, data_dataloader, y_max,
                                                                                             y_min, de_max, de_min)
    test_mse_size.append(test_mse)
    test_mae_size.append(test_mae)
    test_acc_size.append(test_acc)
    test_r2_size.append(test_r2)
    print(' acc =', '{:.6f}'.format(test_acc), ' r2 =', '{:.6f}'.format(test_r2),
          ' mse =', '{:.6f}'.format(test_mse), ' mae =', '{:.6f}'.format(test_mae), 'time = ', start)

    acc_mse_mae_dict = {'test_acc': test_acc_size, 'test_r2': test_r2_size,
                             'test_mse': test_mse_size, 'test_mae': test_mae_size, }
    acc_mse_mae = pd.DataFrame(acc_mse_mae_dict)

    test_de = pd.DataFrame(test_depth, columns=['test_depth'])
    test_t = pd.DataFrame(true_test, columns=['test_true'])
    test_p = pd.DataFrame(pre_test, columns=['test_pre'])

    csv_test = pd.concat([test_de, test_t, test_p], axis=1)

    # acc_mse_mae.to_csv('./output0204/test/xj/300_300/acc_mse_mae_xj_2.csv', sep=",", index=True)
    #
    # csv_test.to_csv('./output0204/test/xj/300_300/rel_pre_test_xj_2.csv', sep=",", index=True)
    #
    # true_test_plot(csv_test['test_depth'], csv_test['test_true'], csv_test['test_pre'], 'test',
    #                './output0204/test/xj/300_300/xj_2.png')
    # acc_mse_mae.to_csv('./output0204/test/volve/300_300/acc_mse_mae_volve_9A.csv', sep=",", index=True)
    #
    # csv_test.to_csv('./output0204/test/volve/300_300/rel_pre_test_volve_9A.csv', sep=",", index=True)
    #
    # true_test_plot(csv_test['test_depth'], csv_test['test_true'], csv_test['test_pre'], 'test',
    #                './output0204/test/volve/300_300/volve_9A.png')
    # acc_mse_mae.to_csv('./output0204/test/bh/300_300/acc_mse_mae_bh_2.csv', sep=",", index=True)
    #
    # csv_test.to_csv('./output0204/test/bh/300_300/rel_pre_test_bh_2.csv', sep=",", index=True)
    #
    # true_test_plot(csv_test['test_depth'], csv_test['test_true'], csv_test['test_pre'], 'test',
    #                './output0204/test/bh/300_300/bh_2.png')

initiate()
