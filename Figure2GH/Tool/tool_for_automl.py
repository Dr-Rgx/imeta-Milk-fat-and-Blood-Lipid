# 将所有的 /mnt/x1/luqy21/source/medical-data-analysis/log/autogluon_res 目录下的目录里面的文件移动到 best_quality和interpretable目录下，需要创建文件夹
import os
import shutil

import numpy as np
import pandas as pd
import shap
from autogluon.tabular import TabularPredictor
from matplotlib import pyplot as plt


class AutogluonWrapper:
    def __init__(self, predictor, feature_names):
        self.ag_model = predictor
        self.feature_names = feature_names

    def predict(self, X):
        if isinstance(X, pd.Series):
            X = X.values.reshape(1, -1)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        return self.ag_model.predict(X)


def mv2best_quality():
    root_dir = '/mnt/x1/luqy21/source/medical-data-analysis/log/autogluon_res'
    sub_dir_names = ['best_quality', 'interpretable']
    # 遍历所有root_dir下的文件夹
    for file in os.listdir(root_dir):
        file_dir = os.path.join(root_dir, file)
        if os.path.isdir(file_dir):

            # print(file_dir)  # /mnt/x1/luqy21/source/medical-data-analysis/log/autogluon_res/stage0
            # 删除文件夹和文件夹里面的文件
            # shutil.rmtree(os.path.join(file_dir, 'interpretable'), ignore_errors=True)
            # 创建文件夹
            for i in range(len(sub_dir_names)):
                os.makedirs(os.path.join(file_dir, sub_dir_names[i]), exist_ok=True)

            for sub_dir in os.listdir(file_dir):
                sub_dir_path = os.path.join(file_dir, sub_dir)
                print(sub_dir_path)
                print('os.path.isdir(sub_dir_path)', os.path.isdir(sub_dir_path))
                if not os.path.isdir(sub_dir_path):
                    shutil.move(sub_dir_path, os.path.join(file_dir, sub_dir_names[0]))


def mult_processor_calcu_shap(X, predictor_path, n_process=19, feature_names=None):
    from multiprocessing import Process, Queue
    import copy

    # 多线程计算shap值
    n_samples = X.shape[0]
    batch_size = n_samples // n_process

    # 用于存放计算结果的队列
    queue = Queue()

    # 创建并启动多个进程
    processes = []
    for i in range(n_process):
        start = i * batch_size
        end = (i + 1) * batch_size if i < n_process - 1 else n_samples
        print(f'process {i} start {start} end {end}')

        # 加载predictor
        predictor = TabularPredictor.load(predictor_path)

        # 获取子样本
        X_batch = X[start:end].copy()

        # 创建explainer
        ag_wrapper = AutogluonWrapper(predictor, feature_names)
        explainer = shap.KernelExplainer(ag_wrapper.predict, X)

        # 启动进程
        process = Process(target=_calcul_shap, args=(X_batch, explainer, queue))
        process.start()
        processes.append(process)

    # 获取并合并多个进程的计算结果
    shap_values_list = [queue.get() for _ in range(n_process)]
    shap_values = np.concatenate(shap_values_list, axis=0)

    # 等待所有进程完成
    for process in processes:
        process.join()

    print('calcul shap finish')
    return shap_values


# 单线程直接计算shap值
def single_processor_calcu_shap(X, predictor_path, feature_names=None):
    predictor = TabularPredictor.load(predictor_path)
    ag_wrapper = AutogluonWrapper(predictor, feature_names)
    explainer = shap.KernelExplainer(ag_wrapper.predict, X)
    shap_values = explainer.shap_values(X, nsamples=300)
    return shap_values


def _calcul_shap(X, explainer, queue):
    shap_values = explainer.shap_values(X, nsamples=250)
    queue.put(shap_values)


def _save_test_real_plot(y_pred=None, y_test=None, y_index=-1, y_col_name='test_col', r2=0.8,
                         best_quality_or_interpretable=None, smooth=True, debug=False):
    def smooth_array(array, window_size=50):
        print('smooth_array start ...')

        if window_size % 2 == 0:
            window_size += 1  # 确保窗口大小为奇数，以便在数组的每个元素周围有相同数量的元素
        window = np.ones(window_size) / window_size
        smoothed_array = np.convolve(array, window, mode='same')

        return smoothed_array

    def gaussian_filter(array, window_size=2, sigma=5.0):
        print('gaussian_filter start ...window_size:', window_size, 'sigma:', sigma)
        # 创建高斯窗口
        import scipy
        gauss_window = scipy.signal.windows.gaussian(window_size, std=sigma)

        # 归一化窗口，使其和为1
        gauss_window /= gauss_window.sum()


        # 使用卷积计算平滑数组
        smooth_array = np.convolve(array, gauss_window, mode='same')

        return smooth_array

    def polynomial_smooth(array, degree=5):
        # 获取数组长度
        n = len(array)

        # 生成x坐标
        x = np.arange(n)

        # 使用多项式拟合
        coefficients = np.polyfit(x, array, deg=degree)
        poly = np.poly1d(coefficients)

        # 计算拟合后的平滑数据
        smooth_array = poly(x)

        return smooth_array

    if debug:
        y_pred = np.array(
            [[2.59], [2.95], [4.4], [3.11], [3.3], [5.73], [4.84], [2.52], [2.29], [4.66], [3.54], [2.44], [4.34],
             [3.35], [4.], [5.51], [3.1], [3.], [4.92]])
        y_test = np.array(
            [[2.73], [3.01], [4.31], [3.5], [3.85], [5.67], [4.79], [2.51], [2.71], [4.77], [3.67], [2.36], [4.51],
             [3.49], [4.25], [5.63], [2.96], [2.94], [4.87]])


    else:
        y_pred = y_pred.values
        y_test = y_test.values

    if smooth:
        # y_pred = smooth_array(y_pred.values).reshape(-1, 1)
        # y_test = smooth_array(y_test.values).reshape(-1, 1)

        y_pred = polynomial_smooth(y_pred).reshape(-1, 1)
        y_test = polynomial_smooth(y_test).reshape(-1, 1)

    # 使用plt对y_pred和y_test进行绘制,纵坐标轴从0开始
    plt.plot(y_pred, label='y_pred')
    plt.plot(y_test, label='y_test')
    plt.ylim(0, np.max(y_test) + 0.1)
    plt.legend()
    # 加上r2指标，以文本的形式，在右下角，保留两位小数
    # 添加文本框
    plt.text(0.95, 0.05, f"R\u00b2 Score={r2:.2f}", fontsize=20, ha='right', va='bottom', transform=plt.gca().transAxes)
    plt.title(f'prediction and real value for {y_col_name} in test set')
    prefix = f'log/autogluon_res/{best_quality_or_interpretable}'
    os.makedirs(prefix, exist_ok=True)
    plt.savefig(f'{prefix}/y_i_{y_index}__y_name_{y_col_name}_y_pred_real.png')
    plt.show()

    y_pred_real = pd.DataFrame(columns=['y_pred', 'y_real'], data=np.concatenate([y_pred, y_test], axis=1))
    y_pred_real.to_csv(f'{prefix}/y_i_{y_index}__y_name_{y_col_name}_y_pred_real.csv', index=False)


# 创建根目录，防止提示不存在该目录
def create_dir_if_not_exist(model_stage_i_path, imp_stage_i_save_path, imp_stage_i_load_path):

    lis = [model_stage_i_path, imp_stage_i_save_path, imp_stage_i_load_path]

    for path in lis:
        if '.' in path.split('/')[-1]:  # 如果是文件的形式
            path = '/'.join(path.split('/')[:-1])
        if not os.path.exists(path):
            os.makedirs(path)


def get_num_shuffle_sets(stage):
    return 30 if stage == 0 else 10


def get_feature_time_limits(stage):
    return 3600 * 24 * 1 if stage == 0 else 3600 * 24 * 1


def save_pref(perf, path):
    print('saved path:', path)
    output_file = open(path, 'w')
    output_file.write(str(perf))
    output_file.close()


def do_shap_and_save(X, feature_names, predictor_path, y_col_name, y_index, best_quality_or_interpretable):
    # X = X[:5]

    import shap
    print('shap start ...')

    # NSHAP_SAMPLES = X.shape[
    #     0]  # how many samples to use to approximate each Shapely value, larger values will be slower

    # 在log下创建一个文件夹，用来存放shap的的结果，如果存在不矛盾
    res_dir = f'log/autogluon_res/shap_res/{best_quality_or_interpretable}'
    shap_dir = f'{res_dir}/y_i_{y_index}__y_name_{y_col_name}_shap_values.npy'
    os.makedirs(res_dir, exist_ok=True)

    if os.path.exists(shap_dir):
        print('loading shap values from file ...')
        shap_values = np.load(shap_dir)
    else:
        # shap_values = mult_processor_calcu_shap(X, predictor_path, n_process=15, feature_names=feature_names)
        shap_values = single_processor_calcu_shap(X, predictor_path,  feature_names=feature_names)

    # 保存shap值
    np.save(shap_dir, shap_values)
    # 将shap保存为csv
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    shap_df.to_csv(shap_dir.replace('.npy', '.csv'), index=False)

    # 绘制并保存图片
    # shap.force_plot(explainer.expected_value, shap_values, X, matplotlib=True, show=False)
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.savefig(f'{res_dir}/y_i_{y_index}__y_name_{y_col_name}_summary_plot_bar.png')  # 保存shap值
    # plt.close()

    # 有可能因为bar的没清空 导致后面还有bar的内容
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig(f'{res_dir}/y_i_{y_index}__y_name_{y_col_name}_summary_plot.png')
    plt.close()

    # 在X中，选取出前NSHAP_SAMPLES个特征，因为shap要求X的行数要大于列数，所以这里选取的是前NSHAP_SAMPLES个特征

    # X = X.iloc[:, :NSHAP_SAMPLES]

    # N_VAL = NSHAP_SAMPLES  # -1是debug的 因为现在暂时处理不了
    # N_VAL = 3  # 用1来debug
    #
    # print('N_VAL', N_VAL)
    # N_VAL = 15  # -1是debug的 因为现在暂时处理不了
    # X = X.iloc[:,:NSHAP_SAMPLES]
    # ROW_INDEX = 0  # index of an example datapoint
    # single_datapoint = X_train.iloc[[ROW_INDEX]]
    # single_prediction = ag_wrapper.predict(single_datapoint)
    # shap_values_single = explainer.shap_values(single_datapoint, nsamples=NSHAP_SAMPLES)
    # shap.force_plot(explainer.expected_value, shap_values_single, X_train.iloc[ROW_INDEX, :])
    # calcu shap value
    # shap_values = explainer.shap_values(X_valid.iloc[0:N_VAL, :], nsamples=NSHAP_SAMPLES)
    # N_VAL = 1  # 用1来debug

    # shap_values = explainer.shap_values(X.iloc[0:N_VAL, :], nsamples=NSHAP_SAMPLES)  # 取出X的前N_VAL行，这是因为
    # shap_values = explainer.shap_values(X.iloc[0:N_VAL-10, :], nsamples=NSHAP_SAMPLES)  # 取出X的前N_VAL行，这是因为

    # shap.summary_plot(shap_values, X, show=False, plot_type="dot")
    # plt.savefig(f'{res_dir}/y_i_{y_index}__y_name_{y_col_name}_dot.png')
    # plt.close()
    #
    # shap.summary_plot(shap_values, X, show=False, plot_type="violin")
    # plt.savefig(f'{res_dir}/y_i_{y_index}__y_name_{y_col_name}_violin.png')
    # plt.close()
    #
    # shap.summary_plot(shap_values, X, show=False, plot_type="compact_dot")
    # plt.savefig(f'{res_dir}/y_i_{y_index}__y_name_{y_col_name}_compact_dot.png')
    # plt.close()

    print('shap finish')

    # 将传入的预测和真实的y 保存在一个csv文件中
