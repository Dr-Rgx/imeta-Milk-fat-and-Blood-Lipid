# x, y, x_cols = load_for_mission2()
#
# print(x)
# import xgboost
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
import shap

from load_data import load_for_mission2, load_for_mission1




def exp_by_shap(model, X, x_cols):
    import shap

    # load JS visualization code to notebook
    # shap.initjs()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # 可视化解释性 (use matplotlib=True to avoid Javascript)
    # shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :],show=True,matplotlib=True)

    # shap.dependence_plot("RM", shap_values, X,show=True)

    res = sorted(np.mean(np.abs(shap_values), axis=0), reverse=True)  # 作为结果
    names = x_cols[np.argsort(np.mean(np.abs(shap_values), axis=0))]
    X_to_drwa = pd.DataFrame(columns=x_cols, data=X)

    shap.summary_plot(shap_values, X_to_drwa, show=True)
    shap.summary_plot(shap_values, X_to_drwa, plot_type="bar", show=True)
    print(X_to_drwa)

    return res, names


"""训练 XGBoost 模型，SHAP里提供了相关数据集"""

# X, y = shap.datasets.boston()

X, y, x_cols = load_for_mission1(3)

X = pd.DataFrame(columns=x_cols,data=X)
print(X)

other_params = {
    'n_estimators': 10,
    'learning_rate': 0.31,
    'max_depth': 4,
    # 'subsample': 0.71,
    # 'colsample_bytree': 0.87,
    # 'max_depth': 2,
    # 'random_state': 123,
    # 'min_child_weight': 1,
    # 'reg_lambda': 0.06,
    # 'reg_alpha': 0.011,
    # 'gamma': 0.1,
    # 'subsample':0.72,
    # 'reg_alpha': 0.84,
    # 'reg_lambda':0.99
    # 'min_samples_split': 6,  'min_samples_leaf': 2,
    # 'max_features': 'sqrt',
    # 'min_weight_fraction_leaf':0,
    # 'min_impurity_decrease': 0,
    # 'alpha': 0.9, 'max_leaf_nodes': None
}

model = XGBRegressor(**other_params)
model.fit(X, y)

# GS for regress
# 参数设定（静止参数）


# 参数设定（网格参数）
cv_params = {
    'n_estimators': [10, 20, 30],
    'learning_rate': np.arange(0, 1, 0.01),
    'max_depth': np.arange(1, 10, 1),

    # 'subsample': np.arange(0, 1, 0.01),
    # 'colsample_bytree': np.arange(0, 1, 0.01)  # 区间为 ( 0,1]
    # 'min_child_weight': paralist_int(1, 15, 1),
    # 'max_features': paralist_int(1, 141, 1),
    # 'alpha': paralist(0.1, 0.99, 0.01),
    # 'min_samples_split': paralist_int(2, 50, 1),
    # 'min_samples_leaf': paralist_int(1, 20, 1),
    # 'min_child_weight': paralist_int(1, 15, 1),
    # 'reg_alpha':  paralist(0, 1, 0.001),
    # 'reg_lambda':  paralist(1.6, 1.8, 0.01),  # [1, 100] 都是正常值
    # 'gamma': paralist(0, 1, 1e-2),
    #  'eta': paralist(0.01, 0.1, 1e-2),
    # 'colsample_bylevel': (0.1, 1, 0.01),
    # 'eta': paralist(1e-4, 1e-3, 1e-4)
}

gs = GridSearchCV(model, cv_params, verbose=1, refit=True, cv=3, n_jobs=-1, scoring='neg_mean_squared_error'
                  # 多分类问题使用f1_micro 2分类使用f1
                  )  # neg_mean_squared_error
gs.fit(X, y)  # X为训练数据的特征值，y为训练数据的label

model = gs.best_estimator_

# model = XGBRegressor(**other_params)
# model.fit(X,y)
print(gs.best_params_)
print(gs.best_score_)
s, n = exp_by_shap(model, X, x_cols)

print(n)

print('end')
"""
通过SHAP值来解释预测值
(同样的方法也适用于 LightGBM, CatBoost, and scikit-learn models)
"""
