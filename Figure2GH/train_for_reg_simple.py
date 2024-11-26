# 只有尝试传统模型的部分

from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)  # 消除 FutureWarning

from xgboost import XGBRegressor

from load_data import Load_for_blood_fat_meta
from models_for_reg import Try_on_model, EmbeddedSelection_for_reg

# y_i = 4  # 第七列代表任务2 n分类

y_names = ['CHO', 'TG', 'HDL', 'LDL', 'APOB']
# y 的顺序为 'CHO', 'TG', 'HDL', 'LDL', 'APOB' 注意这里可能和之前结论的顺序不一样

x_columns = None


def try_on_models_on_task1_4():
    data_loader = Load_for_blood_fat_meta()

    def try_on_models_on_data(data):
        x, y, x_columns = data

        print(x.shape)
        print(y.shape)

        # tom = Try_on_model(x, y, 0)
        # tom.try_on_models()
        for yi_index in range(5):
            print('now the name is ', y_names[yi_index])
            tom = Try_on_model(x, y, yi_index)
            tom.try_on_models()

    try_on_models_on_data(data_loader.task1())
    try_on_models_on_data(data_loader.task2())
    try_on_models_on_data(data_loader.task3())
    try_on_models_on_data(data_loader.task4())


def try_on_models_on_task9():
    data_loader = Load_for_blood_fat_meta()
    x, y = data_loader.task9_res()
    print(x.shape)
    print(y.shape)

    # tom = Try_on_model(x, y, 0)
    # tom.try_on_models()
    for yi_index in range(5):
        print('now the name is ', y_names[yi_index])
        tom = Try_on_model(x, y, yi_index)
        tom.try_on_models()


def get_param(i: int):
    """
    根据y的索引获得当时调到的超参数 Y有5个 所以长度均为5
    """

    params = {
        # task1的参数
        1: {'y_thods': [0.008, 0.009, 0.017, 0.02, 0.02],  # 前5个的阈值
            'n_est': [98, 18, 98, 10, 10],
            'lr': [0.04, 0.25, 0.06, 0.37, 0.25],
            'md': [8, 5, 6, 6, 2]},

        # task2的参数
        2: {'y_thods': [0.002, 0.012, 0.011, 0.003, 0.04],  # 前5个的阈值
            'n_est': [90, 54, 34, 12, 28],
            'lr': [0.18, 0.05, 0.17, 0.25, 0.29],
            'md': [2, 6, 3, 5, 4]},

        # task3的参数
        3: {'y_thods': [0.002, 0.0066, 0.01, 0.003, 0.007],  # 前5个的阈值
            'n_est': [98, 54, 68, 12, 38],
            'lr': [0.1, 0.34, 0.06, 0.26, 0.29],
            'md': [2, 6, 5, 5, 6]},

        # task4的参数
        4: {'y_thods': [0.005, 0.005, 0.012, 0.006, 0.007],  # 前5个的阈值
            'n_est': [40, 14, 10, 28, 18],
            'lr': [0.14, 0.32, 0.42, 0.22, 0.29],
            'md': [3, 7, 6, 5, 6]
            }
    }

    pms = params[i]
    return pms['y_thods'], pms['n_est'], pms['lr'], pms['md']


# 作shape解释时候重新训练 按照之前的参数
def retrain_for_shape_task(task_nub, y_index):
    data_loader = Load_for_blood_fat_meta()

    if task_nub == 1:
        x_values, y_values, x_columns = data_loader.task1()
    if task_nub == 2:
        x_values, y_values, x_columns = data_loader.task2()
    if task_nub == 3:
        x_values, y_values, x_columns = data_loader.task3()
    if task_nub == 4:
        x_values, y_values, x_columns = data_loader.task4()

    print('here', x_columns)

    y_thods, n_est, lr, md = get_param(task_nub)

    y_values = y_values[:, y_index]

    emb = EmbeddedSelection_for_reg(x_values, y_values, x_columns, y_index, y_thods[y_index])
    model = emb.fit_on_basic_model(XGBRegressor())
    emb.use_xgboost_with_hypara(model, ne=n_est[y_index], lr=lr[y_index], md=md[y_index], y_index=y_index,
                                task_nub=task_nub)


if __name__ == '__main__':


    try_on_models_on_task1_4()
