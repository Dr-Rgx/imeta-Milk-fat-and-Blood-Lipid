"""
在这里存放封装的用于解决回归问题的类

最简化在main中的调用，在main中应该形如以下：

    emb = EmbeddedSelection_for_clf(x, y, thre=0.04)

    model_reuse = emb.fit_on_basic_model(XGBClassifier())  # 先基于基模型进行筛选

    emb.reuse_on_xgboost(model_for_selection=model_reuse)  # 然后再使用模型调参看能达到什么程度
"""

import warnings

warnings.filterwarnings("ignore")  # 不显示warning 方便采集结果（warning就只有版本差异的问题）
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, plot_roc_curve
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from xgboost import XGBClassifier

import data_plot
from sklearn.metrics import plot_confusion_matrix
from load_data import data_process
from my_tool import paralist, paralist_int


class EmbeddedSelection_for_clf:  # 为分类问题提供的嵌入式筛选

    x_data = None
    y_data = None

    thre = None

    x_cols = None
    y_col_index = None

    def __init__(self, x_data, y_data, x_cols, y_col_index, thre, mult=False):  # mult 代表传入的y是否有多列

        self.x_data = x_data

        if mult:
            self.y_data = y_data[:, y_col_index]
        else:
            self.y_data = y_data

        self.thre = thre

        self.x_cols = x_cols

        self.y_col_index = y_col_index

    # 基于基模型筛选 希望传入的模型有因素衡量的属性 因为是初次筛选，所以不进行数据划分
    def filt_on_basic_model(self, model):  # 先在基模型上初筛选出模型

        model.fit(self.x_data, self.y_data)
        f1 = f1_score(y_pred=model.predict(self.x_data), y_true=self.y_data, average='micro')
        print('筛选模型，训练集上的f1为：', np.mean(f1))

        data_plot.y_column_index = self.y_col_index

        data_plot.plot_feature_with_name(model, 'log/y%.0f/feature_importance.html' % self.y_col_index, self.x_cols,
                                         thre=self.thre)

        return model

    # 然后使用该模型训练，看最大预测效果
    def reuse_on_xgboost(self, model_for_selection):

        sfm = SelectFromModel(model_for_selection, prefit=True, threshold=self.thre)

        X_train, X_test, y_train, y_test = data_process(self.x_data, self.y_data, False)

        X_train = sfm.transform(X_train)

        X_test = sfm.transform(X_test)

        print('筛选过后X的shape为', X_train.shape)
        # 参数设定（静止参数）
        other_params = {
            'n_estimators': 21,
            'learning_rate': 0.13,
            'subsample': 0.71,
            'colsample_bytree': 0.87,
            'max_depth': 2,
            'random_state': 123,
            'min_child_weight': 1,
            'reg_lambda': 0.06,
            'reg_alpha': 0.011,
            'gamma': 0.1,
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
            'n_estimators': paralist_int(1, 150, 1),
            'max_depth': paralist_int(1, 10, 1),
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
            # 'learning_rate': paralist(0.01, 2, 0.01),
            # 'eta': paralist(1e-4, 1e-3, 1e-4)
        }

        print(X_train.shape)
        print(y_train.shape)

        regress_model = XGBClassifier(**other_params)  # 注意这里的两个 * 号,表示将这个dict转为入参形式
        gs = GridSearchCV(regress_model, cv_params, verbose=1, refit=True, cv=10, n_jobs=15, scoring='f1_micro'
                          # 多分类问题使用f1_micro 2分类使用f1
                          )  # neg_mean_squared_error
        gs.fit(X_train, y_train)  # X为训练数据的特征值，y为训练数据的label

        model = gs.best_estimator_
        print(model)
        print('此次获得最佳参数为：', gs.best_params_)

        # 注意 对于测试集 不要使用交叉验证，否则样本数太少 （交叉验证会重新去训练）
        acc = f1_score(y_pred=model.predict(X_test), y_true=y_test, average='micro')
        print('thre', self.thre, "最佳模型得分:", gs.best_score_, '测试集上的f1为：', acc)
        # plot_feature(model, 'log/y%.f/reuse_xgboost_feature_importance.html' % self.y_col_index) 不再进行特征的绘制了

        data_plot.plot_roc_with_auc(model, X_train, X_test, y_train, y_test)

    # 然后使用该模型训练，看最大预测效果
    def reuse_on_svm(self, model_for_selection):
        sfm = SelectFromModel(model_for_selection, prefit=True, threshold=self.thre)

        X_train, X_test, y_tain, y_test = data_process(self.x_data, self.y_data, False)

        X_train = sfm.transform(X_train)

        X_test = sfm.transform(X_test)

        print('筛选过后X的shape为', X_train.shape)
        # 参数设定（静止参数）
        other_params = {
            'kernel': 'linear'
        }
        # 参数设定（网格参数）
        cv_params = {
            'C': paralist(0.1, 5.1, 0.01)
        }

        print(X_train.shape)
        print(y_tain.shape)

        regress_model = SVC(**other_params)  # 注意这里的两个 * 号,表示将这个dict转为入参形式
        gs = GridSearchCV(regress_model, cv_params, verbose=1, refit=True, cv=10, n_jobs=15, scoring='f1_micro'
                          # 多分类问题使用f1_micro 2分类使用f1
                          )  # neg_mean_squared_error
        gs.fit(X_train, y_tain)  # X为训练数据的特征值，y为训练数据的label

        model = gs.best_estimator_
        print(model)
        print('此次获得最佳参数为：', gs.best_params_)

        # 注意 对于测试集 不要使用交叉验证，否则样本数太少 （交叉验证会重新去训练）
        acc = accuracy_score(y_pred=model.predict(X_test), y_true=y_test)
        print('thre', self.thre, "最佳模型得分:", gs.best_score_, '测试集上的f1为：', acc)
        # plot_feature(model, 'log/y%.f/reuse_xgboost_feature_importance.html' % self.y_col_index)


class Try_on_model:
    """"
    只是单纯的尝试某种模型效果,不做特征筛选（只有少部分时候用得到）
    并且就不画图了，先看比较好的模型假设
    """
    x_data = None
    y_data = None
    y_index = None

    def __init__(self, x_data, y_data, y_index=-1):  # mult 代表是否需要对 y_data进行切分，对于之前的任务需要，这里不需要

        self.x_data = x_data

        self.y_data = y_data

        self.y_index = y_index

    def try_on_svC(self):
        # 参数设定（静止参数）
        X_train, X_test, y_tain, y_test = data_process(self.x_data, self.y_data, False)

        other_params = {
        }

        cv_params = [
            # {'kernel': ['rbf'], 'gamma': paralist(0, 1, 1e-2),
            #   'C': paralist(0.01, 5, 1)},
            {'kernel': ['linear'], 'C': paralist(1, 1000, 10)}
        ]

        regress_model = SVC(**other_params)  # **other_params 注意这里的两个 * 号,表示将这个dict转为入参形式
        gs = GridSearchCV(regress_model, cv_params, verbose=0, refit=True, cv=10,n_jobs=15,
                          scoring='f1_micro')  # neg_mean_squared_error
        gs.fit(X_train, y_tain)  # X为训练数据的特征值，y为训练数据的label

        model = gs.best_estimator_
        # print('此次获得最佳参数为：', gs.best_params_)

        # 注意 对于测试集 不要使用交叉验证，否则样本数太少 （交叉验证会重新去训练）
        f1 = f1_score(y_pred=model.predict(X_test), y_true=y_test, average='micro')
        print("SVC训练集准确率:", gs.best_score_, 'SVC测试集上的f1为：', f1)

    def try_on_RandomForest(self):
        # 参数设定（静止参数）
        X_train, X_test, y_tain, y_test = data_process(self.x_data, self.y_data, False)

        other_params = {
            'n_estimators': 101,
            'max_features': 800
        }

        cv_params = {
            # 'n_estimators': paralist(1, 200, 1)
            # 'max_features':paralist(1,1000,1)
            'max_depth': paralist(1, 80, 2)
        }

        regress_model = RandomForestClassifier(**other_params)  # **other_params 注意这里的两个 * 号,表示将这个dict转为入参形式
        gs = GridSearchCV(regress_model, cv_params, verbose=0, refit=True, cv=10, n_jobs=-1,
                          scoring='f1_micro')  # neg_mean_squared_error
        gs.fit(X_train, y_tain)  # X为训练数据的特征值，y为训练数据的label

        model = gs.best_estimator_
        print('随机森林,此次搜索结果：', gs.best_params_)

        # 注意 对于测试集 不要使用交叉验证，否则样本数太少 （交叉验证会重新去训练）
        f1 = f1_score(y_pred=model.predict(X_test), y_true=y_test, average='micro')
        print("随机森林训练集准确率:", gs.best_score_, '随机森林测试集上的f1为：', f1)

    def try_on_models(self):
        self.try_on_RandomForest()
        self.try_on_svC()


class LR_for_clf:
    """逻辑回归+分类,预测x_diet.csv的invention(2为一类)"""

    def __init__(self, x_data, y_data, x_columns):
        self.x_data = x_data
        self.y_data = y_data
        self.x_columns = x_columns

        self.y_train = None
        self.y_test = None
        self.y_pred_test = None
        self.y_pred_train = None

    def process_for_two(self):
        # # 选取重要特征
        # rfe = RFE(LogisticRegression(multi_class='multinomial'), n_features_to_select=15)
        # rfe.fit(self.x_data, self.y_data)
        # X = rfe.transform(self.x_data)
        # features = self.x_columns[rfe.get_support()]
        # print(features)

        X_train, X_test, self.y_train, self.y_test = train_test_split(self.x_data, self.y_data, test_size=0.1, random_state=13)

        sfm = SelectFromModel(LogisticRegression(multi_class='multinomial').fit(self.x_data, self.y_data), prefit=True, threshold=0.26)
        features = self.x_columns[sfm.get_support()]
        print("筛选特征数：",len(features))
        print(features)
        X_train = sfm.transform(X_train)
        X_test = sfm.transform(X_test)

        params = {}
        others = {'solver':'lbfgs',
                  'multi_class':'multinomial',
                  'random_state':7}
        clf = LogisticRegression(**others)
        gs = GridSearchCV(clf, params, cv=10, scoring='f1_micro')
        gs.fit(X_train, self.y_train)
        print('LogisticRegression此次搜索结果：', gs.best_params_)

        model = gs.best_estimator_
        print("LogisticRegression训练集f1评分:", gs.best_score_)
        print("LogisticRegression测试集f1评分:", gs.score(X_test,self.y_test))

        task="task2"
        self.plot_confusion_matrix_for_train(model, X_train, self.y_train,task)
        self.plot_confusion_matrix_for_test(model, X_test, self.y_test,task)

    def process_for_three(self):
        # 标准化数据
        # scaler = StandardScaler()
        # X = scaler.fit_transform(self.x_data)
        X=self.x_data

        # 选取重要特征
        rfe = RFE(LogisticRegression(), n_features_to_select=9)
        rfe.fit(X, self.y_data)
        X = rfe.transform(X)
        features = self.x_columns[rfe.get_support()]
        print(features)

        X_train, X_test, self.y_train, self.y_test = train_test_split(X, self.y_data, test_size=0.1, random_state=13)

        params = {}
        others = {'random_state':7}
        clf = LogisticRegression(**others)
        gs = GridSearchCV(clf, params, cv=10, scoring='f1_micro')
        gs.fit(X_train, self.y_train)
        print('LogisticRegression此次搜索结果：', gs.best_params_)

        model = gs.best_estimator_
        print("LogisticRegression训练集f1评分:", gs.best_score_)
        print("LogisticRegression测试集f1评分:", gs.score(X_test,self.y_test))

        task="task3"
        self.plot_roc_for_test(model, X_test, self.y_test,task)
        self.plot_confusion_matrix_for_train(model, X_train, self.y_train,task)
        self.plot_confusion_matrix_for_test(model, X_test, self.y_test,task)
        # k2 = cross_val_score(model, X, self.y_data, cv=10, scoring="f1_micro")
        # print(f"交叉验证f1_micro评价：{k2}")
        # print(f"交叉验证f1_micro评价均值：{k2.mean()}")

    def plot_confusion_matrix_for_train(self,model,X_train,y_train,task):
        plot_confusion_matrix(model, X_train, y_train,cmap=plt.cm.Blues)
        plt.savefig(f"log/picture/{task}/LogisticRegression_confusion_matrix_for_train.png")
        plt.show()

    def plot_confusion_matrix_for_test(self,model,X_test,y_test,task):
        plot_confusion_matrix(model, X_test, y_test,cmap=plt.cm.Blues)
        plt.savefig(f"log/picture/{task}/LogisticRegression_confusion_matrix_for_test.png")
        plt.show()

    def plot_roc_for_test(self,model,X_test,y_test,task):
        plot_roc_curve(model, X_test, y_test)
        plt.savefig(f"log/picture/{task}/LogisticRegression_roc_for_test.png")
        plt.show()

class AB_for_clf:
    """AdaBoost"""
    def __init__(self, x_data, y_data, x_columns):
        self.x_data = x_data
        self.y_data = y_data
        self.x_columns = x_columns

        self.y_train = None
        self.y_test = None
        self.y_pred_test = None
        self.y_pred_train = None

    def process_for_two(self):
        X=self.x_data
        y=self.y_data
        X_train, X_test, self.y_train, self.y_test = train_test_split(X,y, test_size=0.1, random_state=13)

        params = {}
        others = {'base_estimator': None,
                  'n_estimators': 32,
                  'learning_rate': 0.5,
                  'algorithm': 'SAMME.R',
                  'random_state': 13}
        clf = AdaBoostClassifier(**others)
        gs = GridSearchCV(clf, params, cv=10, scoring='f1_micro')
        gs.fit(X_train, self.y_train)
        print('AdaBoost此次搜索结果：', gs.best_params_)

        model = gs.best_estimator_
        print("AdaBoost训练集f1评分:", gs.best_score_)
        print("AdaBoost测试集f1评分:", gs.score(X_test,self.y_test))

        print('AdaBoost权重：',model.feature_importances_)
        print('AdaBoost权重不为0的特征数目：',len(np.nonzero(model.feature_importances_)[0]))
        print('AdaBoost权重不为0的特征：', self.x_columns[np.nonzero(model.feature_importances_)[0]])

        k2 = cross_val_score(model, self.x_data, self.y_data, cv=10, scoring="f1_micro")
        print(f"交叉验证f1_micro评价：{k2}")
        print(f"交叉验证f1_micro评价均值：{k2.mean()}")

        task="task2"
        self.plot_confusion_matrix_for_train(model, X_train, self.y_train,task)
        self.plot_confusion_matrix_for_test(model, X_test, self.y_test,task)

    def process_for_three(self):
        X = self.x_data
        y = self.y_data
        X_train, X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.1, random_state=13)

        params = {}
        others = {'base_estimator': None,
                  'n_estimators': 50,
                  'learning_rate': 1,
                  'algorithm': 'SAMME.R',
                  'random_state': 13}
        clf = AdaBoostClassifier(**others)
        gs = GridSearchCV(clf, params, cv=10, scoring='f1_micro')
        gs.fit(X_train, self.y_train)
        print('AdaBoost此次搜索结果：', gs.best_params_)

        model = gs.best_estimator_
        print("AdaBoost训练集f1评分:", gs.best_score_)
        print("AdaBoost测试集f1评分:", gs.score(X_test,self.y_test))

        print('AdaBoost权重：', model.feature_importances_)
        print('AdaBoost权重不为0的特征数目：', len(np.nonzero(model.feature_importances_)[0]))
        print('AdaBoost权重不为0的特征：', self.x_columns[np.nonzero(model.feature_importances_)[0]])

        # k2 = cross_val_score(model, self.x_data, self.y_data, cv=10, scoring="f1_micro")
        # print(f"交叉验证f1_micro评价：{k2}")
        # print(f"交叉验证f1_micro评价均值：{k2.mean()}")

    def plot_confusion_matrix_for_train(self,model,X_train,y_train,task):
        plot_confusion_matrix(model, X_train, y_train,cmap=plt.cm.Blues)
        plt.savefig(f"log/picture/{task}/AdaBoost_confusion_matrix_for_train.png")
        plt.show()

    def plot_confusion_matrix_for_test(self,model,X_test,y_test,task):
        plot_confusion_matrix(model, X_test, y_test,cmap=plt.cm.Blues)
        plt.savefig(f"log/picture/{task}/AdaBoost_confusion_matrix_for_test.png")
        plt.show()

    def plot_roc_for_test(self,model,X_test,y_test,task):
        plot_roc_curve(model, X_test, y_test)
        plt.savefig(f"log/picture/{task}/AdaBoost_roc_for_test.png")
        plt.show()