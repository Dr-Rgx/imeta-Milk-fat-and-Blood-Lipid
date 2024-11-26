"""
在这里写一些工具函数
"""
# 快速生成调参范围 起点 终点（闭区间） 步长 两个函数表示是否为整数
import numpy as np
import pandas as pd


def paralist(begin, end, step):
    paras = np.arange(begin, end + step, step)
    return paras


def paralist_int(begin, end, step):
    paras = np.arange(begin, end + step, step)
    return paras


# 使用shap解释模型
def exp_by_shap(model, X, x_cols, y_index, task_nub):
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    res = sorted(np.mean(np.abs(shap_values), axis=0), reverse=True)  # 作为结果

    names = x_cols[np.argsort(np.mean(np.abs(shap_values), axis=0))]  # 这个默认降序

    X_to_drwa = pd.DataFrame(columns=x_cols, data=X)

    X_to_save = pd.DataFrame(columns=names, data=np.array(res).reshape(1, -1))
    X_to_save = X_to_save.T
    print(X_to_save)
    X_to_save.to_csv(f'log/shap/task{task_nub}/{y_index}.csv')  # 拿来给它的接口画图用

    shap.summary_plot(shap_values, X_to_drwa, show=False, plot_size=(25, 12), title=y_index,
                      # path=f'log/shap/task{task_nub}/shape{y_index}.png'
                      )
    # shap.force_plot(explainer.expected_value, shap_values[0, :], X[0, :], show=True)

    # plt.savefig('123.jpg')
    shap.summary_plot(shap_values, X_to_drwa, plot_type="bar", show=False, plot_size=(25, 12), title=y_index,
                      # path=f'log/shap/task{task_nub}/bar{y_index}.png'
                      )

    # load JS visualization code to notebook
    # shap.initjs()

    # 可视化解释性 (use matplotlib=True to avoid Javascript)
    # shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :],show=True,matplotlib=True)

    # shap.dependence_plot("RM", shap_values, X,show=True)
