# First install package from terminal:
# pip install -U pip
# pip install -U setuptools wheel
# pip install autogluon  # autogluon==0.7.0
import os
import warnings
from typing import Dict

import numpy as np
import pandas as pd
import shap
import sklearn
from autogluon.tabular import TabularPredictor
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score

from Tool.tool_for_automl import *
from Tool.tool_for_automl import _save_test_real_plot
from load_data import Load_for_blood_fat_meta
from send_massage import send_masaage


def auto_anay_on_col(y_index, will_train, feat_importance, num_gpus, test,
                     stage, will_shap=False):  # 0代表第一次正常训练
    print('stage:=============', stage, '=' * 15)

    # 打印出这个函数当前的所有实参

    print('locals()', locals())
    print('-' * 20)

    if not USE_MY_DATA:
        pass

        exit(5)
        # X, y = shap.datasets.boston()
        # label = 'label'

    if USE_MY_DATA:
        data, y_col_name = Load_for_blood_fat_meta().task4_autogl(y_index)
        # train_data, test_data = train_test_split(data, test_size=0.2, random_state=0)
        label = y_col_name
        X = data.iloc[:, :-1]  # 去除掉最后一列 因为是y
        y = data.iloc[:, -1]

        model_stage_i_path = f'log/autogluon_models/stage{stage}/{best_quality_or_interpretable}/y_i_{y_index}__y_name_{y_col_name}'

        imp_stage_i_save_path = f'log/autogluon_res/stage{stage}/{best_quality_or_interpretable}/y_i_{y_index}__y_name_{y_col_name}.csv'  # 这是stage0的保存位置
        imp_stage_i_load_path = imp_stage_i_save_path if stage == 0 else f'log/autogluon_res/stage{stage - 1}/{best_quality_or_interpretable}/y_i_{y_index}__y_name_{y_col_name}.csv'

        # create_dir_if_not_exist(model_stage_i_path, imp_stage_i_save_path, imp_stage_i_load_path)

        # model_stage1_path = f'log/autogluon_models/stage1/{best_quality_or_interpretable}/y_i_{y_index}__y_name_{y_col_name}'

        # imp_stage_i_save_path = f'log/autogluon_res/stage{stage}/y_i_{y_index}__y_name_{y_col_name}.csv'  # 这是stage0的保存位置
        # imp_stage1_save_path = f'log/autogluon_res/stage1/y_i_{y_index}__y_name_{y_col_name}.csv'

        if stage != 0:
            name_imp = pd.read_csv(imp_stage_i_load_path)
            print(f'load finish data shape(before) = {X.shape}')
            name_imp = name_imp[name_imp['importance'] >= imp_thre]

            X = X[name_imp.iloc[:, 0].values]  # 把第一列的列名取出来作为筛选

            print(f'load finish X shape = {X.shape}')

    print('model_stage_i_path', model_stage_i_path)
    print('imp_stage_i_save_path', imp_stage_i_save_path)
    print('imp_stage_i_load_path', imp_stage_i_load_path)

    X_train, X_valid, y_train, y_valid = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=0)
    # print(y)

    # def print_accuracy(f):
    #     print("Root mean squared test error = {0}".format(np.sqrt(np.mean((f(X_valid) - y_valid) ** 2))))
    #     time.sleep(0.5)  # to let the print get out before any progress bars

    feature_names = X_train.columns

    train_data = X_train.copy()
    train_data[label] = y_train

    test_data = X_valid.copy()
    test_data[label] = y_valid

    eval_metric = 'r2'  # todo 换成分类指标
    # predictor = TabularPredictor(label=label, problem_type='regression', eval_metric=eval_metric) \
    #     .fit(train_data,
    #          presets=['best_quality'],
    #          # presets=['interpretable'],
    #          time_limit=10,
    #          )

    if will_train:
        predictor = TabularPredictor(label=label, path=model_stage_i_path, eval_metric=eval_metric, ) \
            .fit(train_data,
                 num_bag_folds=10,
                 num_stack_levels=0,
                 # presets=['best_quality'],
                 presets=[best_quality_or_interpretable],  # , 'interpretable'
                 time_limit=TIME_LIMIT,
                 num_cpus=60,
                 num_gpus=num_gpus,
                 keep_only_best=keep_only_best,
                 verbosity=-1
                 # hyperparameters=hyperparameters
                 )

    if LOAD:
        predictor = TabularPredictor.load(model_stage_i_path)

    if test:
        # leaderboard = predictor.leaderboard(test_data)
        # predictor.leaderboard(silent=False)

        y_test = test_data[label]
        x_test = test_data.drop(columns=[label])

        y_pred = predictor.predict(x_test)

        perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True,
                                              detailed_report=True
                                              )
        print('on test perf:', perf)
        end = f'y_i_{y_index}__y_name_{y_col_name}.csv'
        save_pref(perf, imp_stage_i_save_path.replace(end, f'r2_on_all_y_i_{y_index}__y_name_{y_col_name}.txt'))

        r2 = r2_score(y_test, y_pred)
        _save_test_real_plot(y_pred, y_test, y_index, y_col_name, r2=r2,
                             best_quality_or_interpretable=best_quality_or_interpretable)

        # model = predictor.get_model_best() # return str only

    if not LOAD:
        print('train finish not load will exit(2) ....')
        exit(2)

    if feat_importance:
        print('feature importance start ....')

        fea_imp = predictor.feature_importance(data=train_data,  # todo 注意这里用测试集可能会有一点不规范
                                               num_shuffle_sets=get_num_shuffle_sets(stage),
                                               time_limit=get_feature_time_limits(stage),
                                               include_confidence_band=False
                                               )['importance']
        if USE_MY_DATA:
            pd.DataFrame(fea_imp).to_csv(imp_stage_i_save_path)
        else:
            pd.DataFrame(fea_imp).to_csv(
                f'log/autogluon_res/y_i_{y_index}__y_name_test_{best_quality_or_interpretable}.csv'
            )

        # print(fea_imp)

    if will_shap:
        # do_shap_and_save(X, feature_names, predictor, y_col_name, y_index)
        do_shap_and_save(X, feature_names, model_stage_i_path, y_col_name, y_index,
                         best_quality_or_interpretable)  # 由于X太大 用X_valid算了


# 5列可以并行 直接多线程
def run_mult_process_for_stage_i(stage, will_train=True, feat_importance=True, test=True, shap=True, debug=False,
                                 debug_y: int = 0):
    if debug:
        auto_anay_on_col(debug_y, will_train, feat_importance, 0, test, stage, shap)
        return

    # print(locals())
    from multiprocessing import Process

    process = []
    # i,will_train=False, feat_importance=True, num_gpus=8,stage=0
    # (y_index, will_train, feat_importance, num_gpus, test,stage,shap=False)
    for y_index in range(5):
        process.append(
            Process(target=auto_anay_on_col, args=(y_index, will_train, feat_importance, 0, test, stage, shap)))

    [p.start() for p in process]  # 开启了进程
    [p.join() for p in process]  # 等待进程依次结束


# 5列可以并行 直接多线程 进行shap图的绘制和shap数据的导出
def run_mult_process_for_draw(yindex_stage_dict: Dict[int, int], debug: bool = False, y_index=0):
    if debug:
        #               y_index,   will_train,   feat_importance, num_gpus, test,   stage,  will_shap=False
        auto_anay_on_col(y_index, False, False, 8, False, yindex_stage_dict[y_index], True)
        return

    from multiprocessing import Process

    process = []
    # i,will_train=False, feat_importance=True, num_gpus=8,stage=0
    # y_index, will_train, feat_importance, num_gpus, test,stage, will_shap
    for y_index in range(5):
        process.append(
            Process(target=auto_anay_on_col, args=(
                y_index, False, False, 8, False, yindex_stage_dict[y_index], True)
                    )
        )  # 不用减一 因为在输入data的时候就已经进行阈值筛选了

    [p.start() for p in process]  # 开启了进程
    [p.join() for p in process]  # 等待进程依次结束


def run_mult_process_for_shap(yindex_stage_dict: Dict[int, int], debug: bool = False, y_index=0):
    if debug:
        #               y_index,   will_train,   feat_importance, num_gpus, test,   stage,  will_shap=False
        auto_anay_on_col(y_index, False, False, -1, False, yindex_stage_dict[y_index], True)
        return

    from multiprocessing import Process

    process = []
    # i,will_train=False, feat_importance=True, num_gpus=8,stage=0
    # y_index, will_train, feat_importance, num_gpus, test,stage, will_shap
    for y_index in range(5):
        process.append(
            Process(target=auto_anay_on_col, args=(
                y_index, False, False, -1, False, yindex_stage_dict[y_index], True)
                    )
        )  # 不用减一 因为在输入data的时候就已经进行阈值筛选了

    [p.start() for p in process]  # 开启了进程
    [p.join() for p in process]  # 等待进程结束再进行主线程


def loop_for_shap(yindex_stage_dict: Dict[int, int]):
    for y_index in [1, 0, 2, 3, 4]:
        auto_anay_on_col(
            y_index, False, False, -1, False, yindex_stage_dict[y_index], True
        )


# 5列可以并行 直接多线程
def loop_importance_for_stage_i(stage, will_train=True, feat_importance=True, test=True, shap=False
                                ):
    for y_index in range(5):
        auto_anay_on_col(y_index, will_train, feat_importance, 0, test, stage, shap)


if __name__ == '__main__':
    #  CUDA_VISIBLE_DEVICES=3 python Auto_ML_for_reg.py

    import argparse

    parser = argparse.ArgumentParser(description='Control mul process')
    parser.add_argument('--stage', type=int, default=0, required=False)

    args = parser.parse_args()
    stage = args.stage

    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    USE_MY_DATA = True
    keep_only_best = True

    TIME_LIMIT = 3600 * 5

    # TIME_LIMIT = 20  # for debug set
    # feature_time_limits = 20
    # num_shuffle_sets = 1

    # test
    # TIME_LIMIT = 5
    # num_shuffle_sets = 1
    # feature_time_limits = 10

    LOAD = True
    # feat_importance=True
    # shape_any = False

    # best_quality_or_interpretable = 'interpretable'  # 解释之前的，分为 1 改这个 2 改阈值筛选 3.主调函数
    best_quality_or_interpretable = 'best_quality'  # todo 注意 如果是去进行解释的话 用的这个设置 要吧阈值筛选设置成 》= 防止特征数量不对

    warnings.filterwarnings('ignore')
    print(f'USE_MY_DATA = {USE_MY_DATA}')

    imp_thre = 0
    # run_mult_process_for_stage_i(stage, will_train=True, feat_importance=True, test=True, debug=False, debug_y=0)  # 多线程不能用for循环来控制
    # run_mult_process_for_stage_i(stage, will_train=True, feat_importance=True, test=True)  # 多线程不能用for循环来控制
    # 这个只有importance
    # run_mult_process_for_stage_i(stage, will_train=True, feat_importance=True, test=True)  # 多线程不能用for循环来控制

    # y_index_stage_dict_quality = {  # 这是基于best_quality的对应的stage
    #     0: 14,  # 6的特征太多了
    #     1: 15,
    #     2: 15,
    #     3: 18,
    #     4: 17,
    # }

    # y_index_stage_dict_inter = {  # 这是基于best_quality的对应的stage
    #     0: 15,  # 6的特征太多了
    #     1: 7,
    #     2: 6,
    #     3: 7,
    #     4: 16,
    # }

    # run_mult_process_for_draw(y_index_stage_dict)
    # loop_for_shap(y_index_stage_dict_quality)

    # loop_importance_for_stage_i(stage, will_train=False, feat_importance=True, test=True, shap=False)
    #
    # send_masaage(13310247086, 'loop_for_shap done', '40')
    # polo_test(y_index_stage_dict)
    # _save_test_real_plot(debug=True)
    for i in range(20):
        run_mult_process_for_stage_i(i)
