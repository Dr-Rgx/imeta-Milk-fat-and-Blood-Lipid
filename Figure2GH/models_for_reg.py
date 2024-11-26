"""
在这里存放封装的用于解决回归问题的类

最简化在main中的调用，在main中应该形如以下：

    emb = EmbeddedSelection_for_clf(x, y, thre=0.04)

    model_reuse = emb.fit_on_basic_model(XGBClassifier())  # 先基于基模型进行筛选

    emb.reuse_on_xgboost(model_for_selection=model_reuse)  # 然后再使用模型调参看能达到什么程度
"""

import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")  # 不显示warning 方便采集结果（warning就只有版本差异的问题）

import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor

import pandas as pd
from load_data import data_process
from my_tool import paralist, paralist_int, exp_by_shap


class EmbeddedSelection_for_reg:  # 为分类问题提供的嵌入式筛选

    x_data = None
    y_data = None

    thre = None

    x_cols = None
    y_col_index = None

    def __init__(self, x_data, y_data, x_cols, y_col_index, thre, mult=False):  # 如果y本身不止一行，就设置mult为True

        if mult:  # 如果代表y传入要多列，在多列研究的环境下
            self.x_data = x_data
            self.y_data = y_data[:, y_col_index]

        else:
            self.x_data = x_data
            self.y_data = y_data

        self.thre = thre

        self.x_cols = x_cols

        self.y_col_index = y_col_index

    # 基于基模型筛选 希望传入的模型有因素衡量的属性 因为是初次筛选，所以不进行数据划分
    def fit_on_basic_model(self, model):  # 先在基模型上初筛选出模型

        model.fit(self.x_data, self.y_data)
        r2 = r2_score(model.predict(self.x_data), self.y_data)
        # print('筛选模型，训练集上的f1为：', np.mean(r2))

        # data_plot.y_column_index = self.y_col_index

        # data_plot.plot_feature_with_name(model, 'log/y%.0f/feature_importance.html' % self.y_col_index, self.x_cols,
        #                                  thre=self.thre)

        return model

    # 然后使用该模型训练，看最大预测效果
    def reuse_on_xgboost(self, model_for_selection, y_index=False):

        sfm = SelectFromModel(model_for_selection, prefit=True, threshold=self.thre)

        X_train, X_test, y_train, y_test = data_process(self.x_data, self.y_data, False)

        feature_bool_index = sfm.get_support()
        self.x_cols = self.x_cols[feature_bool_index]
        # print('sfm.get_support()',sfm.get_support()) # 得到筛选出来的布尔下标

        X_train = sfm.transform(X_train)

        X_test = sfm.transform(X_test)

        # print('self.x_cols',self.x_cols)

        print('筛选过后X的shape为', X_train.shape)
        # 参数设定（静止参数）
        other_params = {
            # 'n_estimators': 14,
            # 'learning_rate': 0.13,
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
        # 参数设定（网格参数）
        cv_params = {
            'n_estimators': [70],
            'max_depth': [5],
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
            # 'subsample': paralist(0, 1, 0.001),
            # 'colsample_bytree': paralist_int(0, 0.99, 0.001)  # 区间为 ( 0,1]
            'learning_rate': [0.2],
            # 'eta': paralist(1e-4, 1e-3, 1e-4)
        }

        regress_model = XGBRegressor(**other_params)  # 注意这里的两个 * 号,表示将这个dict转为入参形式
        gs = GridSearchCV(regress_model, cv_params, verbose=2, refit=True, cv=10, n_jobs=-1, scoring='r2'
                          # 多分类问题使用f1_micro 2分类使用f1
                          )  # neg_mean_squared_error
        gs.fit(X_train, y_train)  # X为训练数据的特征值，y为训练数据的label

        model = gs.best_estimator_
        # print(model)
        print('此次获得最佳参数为：', gs.best_params_)

        # 注意 对于测试集 不要使用交叉验证，否则样本数太少 （交叉验证会重新去训练）
        y_pred = model.predict(X_test)
        y_true = y_test
        acc = r2_score(y_pred=y_pred, y_true=y_true)
        print('thre', self.thre, "最佳模型得分:", gs.best_score_, '测试集上的r2为：', acc)

        # exp_by_shap(model, X_train, self.x_cols, y_index)

        # data_plot.plot_true_pred(y_true, y_pred, self.y_col_index)

    # 已经挑选出最佳超参数 使用xgboost重新获得该模型 只使用 n-est, lr, max_d
    def use_xgboost_with_hypara(self, model_for_selection, ne, lr, md, y_index=False, task_nub=-1):

        sfm = SelectFromModel(model_for_selection, prefit=True, threshold=self.thre)

        X_train, X_test, y_train, y_test = data_process(self.x_data, self.y_data, False)

        feature_bool_index = sfm.get_support()
        self.x_cols = self.x_cols[feature_bool_index]
        # print('sfm.get_support()',sfm.get_support()) # 得到筛选出来的布尔下标

        X_train = sfm.transform(X_train)

        X_test = sfm.transform(X_test)

        # print('self.x_cols',self.x_cols)

        print('筛选过后X的shape为', X_train.shape)
        # 参数设定（静止参数）
        other_params = {
            'n_estimators': ne,
            'learning_rate': lr,
            # 'subsample': 0.71,
            # 'colsample_bytree': 0.87,
            'max_depth': md,
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
        # 参数设定（网格参数）
        cv_params = {
            # 'n_estimators': [ne],
            # 'max_depth': [5],
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
            # 'subsample': paralist(0, 1, 0.001),
            # 'colsample_bytree': paralist_int(0, 0.99, 0.001)  # 区间为 ( 0,1]
            # 'learning_rate': [0.2],
            # 'eta': paralist(1e-4, 1e-3, 1e-4)
        }

        regress_model = XGBRegressor(**other_params)  # 注意这里的两个 * 号,表示将这个dict转为入参形式
        gs = GridSearchCV(regress_model, cv_params, verbose=2, refit=True, cv=10, n_jobs=-1, scoring='r2'
                          # 多分类问题使用f1_micro 2分类使用f1
                          )  # neg_mean_squared_error
        gs.fit(X_train, y_train)  # X为训练数据的特征值，y为训练数据的label

        model = gs.best_estimator_
        # print(model)
        print('此次获得最佳参数为：', gs.best_params_)

        # 注意 对于测试集 不要使用交叉验证，否则样本数太少 （交叉验证会重新去训练）
        y_pred = model.predict(X_test)
        y_true = y_test
        acc = r2_score(y_pred=y_pred, y_true=y_true)
        print('thre', self.thre, "最佳模型得分:", gs.best_score_, '测试集上的r2为：', acc)

        exp_by_shap(model, X_train, self.x_cols, y_index, task_nub)

        # data_plot.plot_true_pred(y_true, y_pred, self.y_col_index)


class Try_on_model:
    """"
    只是单纯的尝试某种模型效果,不做特征筛选（只有少部分时候用得到）
    并且就不画图了，先看比较好的模型假设
    """
    x_data = None
    y_data = None
    y_index = None

    def __init__(self, x_data, y_data, y_index=-1):
        self.x_data = x_data

        if y_index == -1:
            self.y_data = y_data
        else:
            self.y_data = y_data[:, y_index]

        self.y_index = y_index

    def try_on_svR(self):
        # 参数设定（静止参数）
        X_train, X_test, y_tain, y_test = data_process(self.x_data, self.y_data, False)

        other_params = {
        }

        cv_params = [
            # {'kernel': ['rbf'], 'gamma': paralist(0, 1, 1e-2),
            #   'C': paralist(0.01, 5, 1)},
            {'kernel': ['linear'],
             'C': paralist(1, 100, 10)}
        ]

        regress_model = SVR(**other_params)  # **other_params 注意这里的两个 * 号,表示将这个dict转为入参形式
        gs = GridSearchCV(regress_model, cv_params, verbose=2, refit=True, cv=10, n_jobs=15,
                          scoring='r2')  # neg_mean_squared_error
        gs.fit(X_train, y_tain)  # X为训练数据的特征值，y为训练数据的label

        model = gs.best_estimator_
        # print('此次获得最佳参数为：', gs.best_params_)

        # 注意 对于测试集 不要使用交叉验证，否则样本数太少 （交叉验证会重新去训练）
        r2 = r2_score(y_pred=model.predict(X_test), y_true=y_test)
        print("SVR训练集准确率:", gs.best_score_, 'SVR测试集上的r2为：', r2)

    def try_on_Lasso(self):
        # 参数设定（静止参数）
        X_train, X_test, y_train, y_test = data_process(self.x_data, self.y_data, False)



        cv_params = {
            'alpha': paralist(1e-2, 1, 1e-1)
        }

        regress_model = Lasso()  # **other_params 注意这里的两个 * 号,表示将这个dict转为入参形式
        gs = GridSearchCV(regress_model, cv_params, verbose=0, refit=True, cv=10, n_jobs=15,
                          scoring='r2')  # neg_mean_squared_error
        gs.fit(X_train, y_train)  # X为训练数据的特征值，y为训练数据的label

        model = gs.best_estimator_
        print('Lasso获得最佳参数为：', gs.best_params_)

        # 注意 对于测试集 不要使用交叉验证，否则样本数太少 （交叉验证会重新去训练）
        r2 = r2_score(y_pred=model.predict(X_test), y_true=y_test)
        print("Lasso训练集准确率:", gs.best_score_, '测试集上的r2为：', r2)

    def try_on_Ridge(self):
        # 参数设定（静止参数）
        X_train, X_test, y_tain, y_test = data_process(self.x_data, self.y_data, False)



        cv_params = {
            'alpha': paralist(1e-2, 1, 1e-2)
        }

        regress_model = Ridge()  # **other_params 注意这里的两个 * 号,表示将这个dict转为入参形式
        gs = GridSearchCV(regress_model, cv_params, verbose=0, refit=True, cv=10, n_jobs=15,
                          scoring='r2')  # neg_mean_squared_error
        gs.fit(X_train, y_tain)  # X为训练数据的特征值，y为训练数据的label

        model = gs.best_estimator_
        print('Ridge：', gs.best_params_)

        # 注意 对于测试集 不要使用交叉验证，否则样本数太少 （交叉验证会重新去训练）
        r2 = r2_score(y_pred=model.predict(X_test), y_true=y_test)
        print("Ridge训练集准确率:", gs.best_score_, 'Ridge测试集上的r2为：', r2)

    def try_on_RandomForest(self):
        # 参数设定（静止参数）
        X_train, X_test, y_tain, y_test = data_process(self.x_data, self.y_data, False)

        print(y_tain.shape)
        other_params = {
            'n_estimators': 10,
            'max_features': X_train.shape[1]
        }

        cv_params = {
            # 'n_estimators': paralist(1, 200, 1)
            # 'max_features':paralist(1,X_train.shape[1],1)
            'max_depth': paralist(1, 80, 2)
        }

        regress_model = RandomForestRegressor(**other_params)  # **other_params 注意这里的两个 * 号,表示将这个dict转为入参形式
        gs = GridSearchCV(regress_model, cv_params, verbose=0, refit=True, cv=10, n_jobs=-1,
                          scoring='r2')  # neg_mean_squared_error
        gs.fit(X_train, y_tain)  # X为训练数据的特征值，y为训练数据的label

        model = gs.best_estimator_
        print('随机森林,此次搜索结果：', gs.best_params_)

        # 注意 对于测试集 不要使用交叉验证，否则样本数太少 （交叉验证会重新去训练）
        r2 = r2_score(y_pred=model.predict(X_test), y_true=y_test)
        print("随机森林训练集准确率:", gs.best_score_, '随机森林测试集上的r2为：', r2)

    def try_on_xgboost(self):
        # 参数设定（静止参数）
        X_train, X_test, y_tain, y_test = data_process(self.x_data, self.y_data, False)

        print(y_tain.shape)
        other_params = {
            # 'n_estimators': paralist(1, 80, 10),
            # 'minchildweight':paralist(0.1,0.9,0.01),
            # 'max_depth':paralist(1, 10, 1)
        }

        cv_params = {
            'n_estimators': paralist(1, 80, 10),
            'min_child_weight': paralist_int(1, 40, 1),
            'max_depth': paralist(1, 10, 1)
        }

        regress_model = XGBRegressor(**other_params)  # **other_params 注意这里的两个 * 号,表示将这个dict转为入参形式
        gs = GridSearchCV(regress_model, cv_params, verbose=1, refit=True, cv=10, n_jobs=-1,
                          scoring='r2')  # neg_mean_squared_error
        gs.fit(X_train, y_tain)  # X为训练数据的特征值，y为训练数据的label

        model = gs.best_estimator_
        print('xgboost,此次搜索结果：', gs.best_params_)

        # 注意 对于测试集 不要使用交叉验证，否则样本数太少 （交叉验证会重新去训练）
        r2 = r2_score(y_pred=model.predict(X_test), y_true=y_test)
        print("xgboost训练集准确率:", gs.best_score_, 'xgboost测试集上的r2为：', r2)

    def try_on_models(self):
        print('now anylisis y', self.y_index, ':-----------------------------')
        self.try_on_xgboost()
        self.try_on_svR()
        self.try_on_Lasso()
        self.try_on_Ridge()
        self.try_on_RandomForest()


class AB_for_reg:
    """AdaBoost,对每个回归进行方法保存参数，后续进行模型保存画图后合为通用"""

    def __init__(self, x_data, y_data, x_columns):
        self.x_data = x_data
        self.y_data = y_data
        self.x_columns = x_columns

        self.y_train = None
        self.y_test = None
        self.y_pred_test = None
        self.y_pred_train = None

    def process_for_y0(self):
        # 标准化数据
        scaler = StandardScaler()
        X = scaler.fit_transform(self.x_data)
        y = self.y_data

        sfm = SelectFromModel(
            AdaBoostRegressor(base_estimator=DecisionTreeRegressor(random_state=13), random_state=13).fit(X, y),
            prefit=True, threshold=0.01)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123)
        X_train = sfm.transform(X_train)
        X_test = sfm.transform(X_test)

        params = {
            # 'learning_rate':np.arange(0.1,1.1,0.1)
            # 'n_estimators':np.arange(20,60,1)
        }
        others = {
            'base_estimator': None,
            'n_estimators': 33,
            'learning_rate': 0.4,
            'loss': 'linear',
            'random_state': 13
        }
        reg = AdaBoostRegressor(**others)
        gs = GridSearchCV(reg, params, cv=10, scoring='r2')
        gs.fit(X_train, y_train)
        print('AdaBoost此次搜索结果：', gs.best_params_)

        model = gs.best_estimator_
        print('AdaBoost训练集r2评分：', gs.best_score_)
        print('AdaBoost测试集r2分数', gs.score(X_test, y_test))
        print(model)
        print(model.feature_importances_)
        print('AdaBoost权重不为0的特征数目：', len(np.nonzero(model.feature_importances_)[0]))
        print('AdaBoost权重不为0的特征：', self.x_columns[np.nonzero(model.feature_importances_)[0]])

    def process_for_y1(self):
        # 标准化数据
        scaler = StandardScaler()
        X = scaler.fit_transform(self.x_data)
        y = self.y_data

        sfm = SelectFromModel(
            AdaBoostRegressor(base_estimator=DecisionTreeRegressor(random_state=13), random_state=13).fit(X, y),
            prefit=True, threshold=0.012)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123)
        X_train = sfm.transform(X_train)
        X_test = sfm.transform(X_test)

        params = {
            # 'learning_rate':np.arange(0.1,1.1,0.1)
            # 'n_estimators':np.arange(20,60,1)
        }
        others = {
            'base_estimator': None,
            'n_estimators': 39,
            'learning_rate': 0.3,
            'loss': 'linear',
            'random_state': 13
        }
        reg = AdaBoostRegressor(**others)
        gs = GridSearchCV(reg, params, cv=10, scoring='r2')
        gs.fit(X_train, y_train)
        print('AdaBoost此次搜索结果：', gs.best_params_)

        model = gs.best_estimator_
        print('AdaBoost训练集r2评分：', gs.best_score_)
        print('AdaBoost测试集r2分数', gs.score(X_test, y_test))
        print(model)
        print(model.feature_importances_)
        print('AdaBoost权重不为0的特征数目：', len(np.nonzero(model.feature_importances_)[0]))
        print('AdaBoost权重不为0的特征：', self.x_columns[np.nonzero(model.feature_importances_)[0]])

    def process_for_y2(self):
        # 标准化数据
        scaler = StandardScaler()
        X = scaler.fit_transform(self.x_data)
        y = self.y_data

        sfm = SelectFromModel(
            AdaBoostRegressor(base_estimator=DecisionTreeRegressor(random_state=13), random_state=13).fit(X, y),
            prefit=True, threshold=0.012)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123)
        X_train = sfm.transform(X_train)
        X_test = sfm.transform(X_test)

        params = {
            # 'learning_rate':np.arange(0.1,1.1,0.1)
            # 'n_estimators':np.arange(20,60,1)
        }
        others = {
            'base_estimator': None,
            'n_estimators': 22,
            'learning_rate': 0.6,
            'loss': 'linear',
            'random_state': 13
        }
        reg = AdaBoostRegressor(**others)
        gs = GridSearchCV(reg, params, cv=10, scoring='r2')
        gs.fit(X_train, y_train)
        print('AdaBoost此次搜索结果：', gs.best_params_)

        model = gs.best_estimator_
        print('AdaBoost训练集r2评分：', gs.best_score_)
        print('AdaBoost测试集r2分数', gs.score(X_test, y_test))
        print(model)
        print(model.feature_importances_)
        print('AdaBoost权重不为0的特征数目：', len(np.nonzero(model.feature_importances_)[0]))
        print('AdaBoost权重不为0的特征：', self.x_columns[np.nonzero(model.feature_importances_)[0]])

    def process_for_y3(self):
        # 标准化数据
        scaler = StandardScaler()
        X = scaler.fit_transform(self.x_data)
        y = self.y_data

        sfm = SelectFromModel(
            AdaBoostRegressor(base_estimator=DecisionTreeRegressor(random_state=13), random_state=13).fit(X, y),
            prefit=True, threshold=0.01)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123)
        X_train = sfm.transform(X_train)
        X_test = sfm.transform(X_test)

        params = {
            # 'learning_rate':np.arange(0.1,1.1,0.1)
            # 'n_estimators':np.arange(20,60,1)
        }
        others = {
            'base_estimator': None,
            'n_estimators': 55,
            'learning_rate': 0.6,
            'loss': 'linear',
            'random_state': 13
        }
        reg = AdaBoostRegressor(**others)
        gs = GridSearchCV(reg, params, cv=10, scoring='r2')
        gs.fit(X_train, y_train)
        print('AdaBoost此次搜索结果：', gs.best_params_)

        model = gs.best_estimator_
        print('AdaBoost训练集r2评分：', gs.best_score_)
        print('AdaBoost测试集r2分数', gs.score(X_test, y_test))
        print(model)
        print(model.feature_importances_)
        print('AdaBoost权重不为0的特征数目：', len(np.nonzero(model.feature_importances_)[0]))
        print('AdaBoost权重不为0的特征：', self.x_columns[np.nonzero(model.feature_importances_)[0]])

    def process_for_y4(self):
        # 标准化数据
        scaler = StandardScaler()
        X = scaler.fit_transform(self.x_data)
        y = self.y_data

        sfm = SelectFromModel(
            AdaBoostRegressor(base_estimator=DecisionTreeRegressor(random_state=13), random_state=13).fit(X, y),
            prefit=True, threshold=0.01)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123)
        X_train = sfm.transform(X_train)
        X_test = sfm.transform(X_test)

        params = {
            # 'learning_rate':np.arange(0.1,1.1,0.1)
            # 'n_estimators':np.arange(20,60,1)
        }
        others = {
            'base_estimator': None,
            'n_estimators': 55,
            'learning_rate': 0.6,
            'loss': 'linear',
            'random_state': 13
        }
        reg = AdaBoostRegressor(**others)
        gs = GridSearchCV(reg, params, cv=10, scoring='r2')
        gs.fit(X_train, y_train)
        print('AdaBoost此次搜索结果：', gs.best_params_)

        model = gs.best_estimator_
        print('AdaBoost训练集r2评分：', gs.best_score_)
        print('AdaBoost测试集r2分数', gs.score(X_test, y_test))
        print(model)
        print(model.feature_importances_)
        print('AdaBoost权重不为0的特征数目：', len(np.nonzero(model.feature_importances_)[0]))
        print('AdaBoost权重不为0的特征：', self.x_columns[np.nonzero(model.feature_importances_)[0]])


class XGBoost_for_reg:
    def __init__(self, x_data, y_data, x_columns, thre):
        self.x_data = x_data
        self.y_data = y_data
        self.x_columns = x_columns
        self.thre = thre

    def process(self, task_num, y_col, n_jobs):
        """task_num:任务号，y_col：标签列号"""
        X_train, X_test, y_train, y_test = data_process(self.x_data, self.y_data, test_size=0.2)

        y_train, y_test = y_train[:, y_col], y_test[:, y_col]

        sfm = SelectFromModel(XGBRegressor(verbose=0, n_jobs=n_jobs).fit(X_train, y_train, verbose=2),
                              prefit=True,
                              threshold=self.thre)
        print('pretrain finished')

        X_train = sfm.transform(X_train)
        X_test = sfm.transform(X_test)
        print('筛选过后X的shape为', X_train.shape)
        features = self.x_columns[sfm.get_support()]
        print("筛选特征数：", len(features))
        print(features)

        # 参数设定（静止参数）
        other_params = {
            # 'n_estimators': 10,
            # 'learning_rate': 0.31,
            # 'max_depth': 4,
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
        # 参数设定（网格参数）
        cv_params = {
            'n_estimators': np.arange(10, 80, 5),
            # 'learning_rate': np.arange(0, 1, 0.1),
            # 'max_depth': np.arange(1, 10, 1),

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
        regress_model = XGBRegressor(**other_params)
        gs = GridSearchCV(regress_model, cv_params, cv=10, scoring='r2', n_jobs=n_jobs, verbose=10)
        gs.fit(X_train, y_train)
        print('turing finishd')

        model = gs.best_estimator_
        model.score(X_test, y_test)
        print(model)

        print('此次获得最佳参数为：', gs.best_params_)
        print('XGBoost训练集r2评分：', gs.best_score_)
        print('XGBoost测试集r2分数: ', gs.score(X_test, y_test))

        print(model.feature_importances_)

        # 数据存储
        df = pd.DataFrame(columns=['indicator', 'importance'])
        df['indicator'] = features
        df['importance'] = model.feature_importances_
        df.to_csv(f"log/blood_fat_met_results/{task_num}/{y_col}_xgboost_feature_importance.csv")
