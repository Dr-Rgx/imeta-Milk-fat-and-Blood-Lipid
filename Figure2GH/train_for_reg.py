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
    from warnings import simplefilter

    simplefilter(action='ignore', category=FutureWarning)  # 消除 FutureWarning

    import warnings

    warnings.filterwarnings("ignore")

    # for task_nub in range(4):
    #     for y_index in range(5):
    #         retrain_for_shape_task(task_nub+1, y_index)

    # for y_index in range(5):
    #     for tasks in range(4):
    #         tasks = tasks + 1
    #         retrain_for_shape_task(tasks, y_index)

    # x, y, y_col_name = Load_for_blood_fat_meta().task1()
    # xgb_reg = XGBoost_for_reg(x, y, y_col_name, 0.01)
    # xgb_reg.process(1, 0, 35)

    # 使用自动调参尝试
    train = True
    test = True
    std = False  # 反而掉点

    save_path = 'agModels-predictregs'

    from autogluon.tabular import TabularPredictor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    data, y_col_names = Load_for_blood_fat_meta().task4_autogl(0)

    train_data, test_data = train_test_split(data, test_size=0.2, random_state=0)

    label = y_col_names

    hyperparameters = {
        'NN_TORCH': {},
        'GBM': [
            {'extra_trees': True, 'ag_args': {'name_suffix': 'XT'}},
            {},
            'GBMLarge',
        ],
        'CAT': {},
        'XGB': [
            {'criterion': 'gini', 'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass']}},
            {'criterion': 'entropy', 'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass']}},
            {'criterion': 'squared_error', 'ag_args': {'name_suffix': 'MSE', 'problem_types': ['regression']}},
        ],
        'FASTAI': {},
        'RF': [
            {'criterion': 'gini', 'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass']}},
            {'criterion': 'entropy', 'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass']}},
            {'criterion': 'squared_error', 'ag_args': {'name_suffix': 'MSE', 'problem_types': ['regression']}},
        ],
        'XT': [
            {'criterion': 'gini', 'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass']}},
            {'criterion': 'entropy', 'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass']}},
            {'criterion': 'squared_error', 'ag_args': {'name_suffix': 'MSE', 'problem_types': ['regression']}},
        ],
        'KNN': [
            {'weights': 'uniform', 'ag_args': {'name_suffix': 'Unif'}},
            {'weights': 'distance', 'ag_args': {'name_suffix': 'Dist'}},
        ],
    }

    # eval_metric = 'r2'
    eval_metric = 'mean_squared_error'

    if std:
        std_scaler = StandardScaler()
        train_data.iloc[:, :-1] = std_scaler.fit_transform(train_data.iloc[:, :-1])  # 计算均值和方差
        test_data.iloc[:, :-1] = std_scaler.transform(test_data.iloc[:, :-1])  # 计算均值和方差

    if train:
        predictor = TabularPredictor(label=label, path=save_path, eval_metric=eval_metric, ) \
            .fit(train_data,
                 num_bag_folds=10,
                 num_stack_levels=0,
                 # presets=['best_quality'],
                 presets=['interpretable'],  # , 'interpretable'
                 # time_limit=5000,
                 time_limit=20,
                 num_cpus=128,
                 num_gpus=2,
                 # hyperparameters=hyperparameters
                 )

    # test stage

    predictor = TabularPredictor.load("agModels-predictregs/")

    model = predictor.get_model_best()

    y_test = test_data[label]
    x_test = test_data.drop(columns=[label])

    y_pred = predictor.predict(x_test)

    perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True,
                                          detailed_report=True
                                          )

    print(perf)
    # todo 吧识别错误的列修改一下 看下怎么改
    leaderboard = predictor.leaderboard(test_data)
    predictor.leaderboard(silent=True)

    # todo shap

    # feature_names = data.columns[:-1]
    # ag_wrapper = AutogluonWrapper(predictor, feature_names)
    # print(callable(ag_wrapper), callable(ag_wrapper))
    #
    #
    # explainer = shap.Explainer(ag_wrapper)
    # shap_values = explainer(x_test,max_evals=7177)
    # print(shap_values)
    #
    # # print_accuracy(ag_wrapper.predict)
    #
    # X_train_summary = shap.kmeans(train_data[:,:-1], 10)
    #
    # explainer = shap.KernelExplainer(ag_wrapper.predict, X_train_summary)
    #
    # NSHAP_SAMPLES = 50  # how many samples to use to approximate each Shapely value, larger values will be slower
    # N_VAL = 10
    #
    # ROW_INDEX = 0  # index of an example datapoint
    # single_datapoint = train_data[:,:-1].iloc[[ROW_INDEX]]
    # single_prediction = ag_wrapper.predict(single_datapoint)
    #
    # shap_values_single = explainer.shap_values(single_datapoint, nsamples=NSHAP_SAMPLES)
    # shap.force_plot(explainer.expected_value, shap_values_single, train_data[:,:-1].iloc[ROW_INDEX, :])

    # todo shuffle importance

    fea_imp = predictor.feature_importance(data=test_data,
                                           num_shuffle_sets=1)

    print(fea_imp)
