# 可视化特征重要性 将x轴变成了列名 并且如果传入了阈值则根据阈值进行绘制
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost

# import plot_in_echart as pie  # 自己封装的 使用pyechart的可视化

y_column_index = -1

y_names = ['CHO', 'TG', 'HDL', 'LDL', 'APOB']


def locate_x_file_by_index(x_column_indexes, index: int):
    i = 0  # 从0开始搜索
    while index > x_column_indexes[i + 1]:
        i += 1
    return i


def plot_feature_with_name(model, path, x_columns, thre=-1.):
    """"
    传入模型，保存路径，x列名，阈值 如果阈值为默认值，则不进行按阈值筛选
    支持feature_importances_ 和 coef_
    """

    impotances = None
    if hasattr(model, 'feature_importances_'):
        impotances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        print('using coef_')
        print('model.coef_.shape', model.coef_.shape)
        impotances = np.mean(model.coef_, axis=1)  # coef需要转成一维
        print('impotances', impotances)
    else:
        print('the model can\'t select features')

    # print('model.feature_importances_',model.feature_importances_)

    feat_imp = pd.Series(impotances).sort_values(ascending=False)  # 先对重要程度进行排序

    print('x_columns', x_columns.shape)
    print('the sorted feature_importances_', feat_imp.shape)

    x = x_columns[feat_imp.index]

    i = 0
    if thre != -1.0:  # 如果传入了阈值，则根据阈值进行筛选

        while feat_imp.values[i] > thre:  # 找到划分的分界限
            print('feat_imp.values[i], thre', feat_imp.values[i], thre)

            i += 1
        x = x[:i]
        feat_imp = feat_imp[:i]

    print('i', i, 'thre', thre)

    print('len(feat_imp)', len(feat_imp))
    # 将结论保存为csv
    df = pd.DataFrame(columns=['indicator', 'importance'])
    df['indicator'] = x
    df['importance'] = feat_imp.values

    df.to_csv('log/y%.0f/lin_SVC_feature_importance.csv' % y_column_index)

    # 绘图
    # pie.plot_one(x.values, df['importance'].values, path)  # 只看特征是否平滑


def plot_feature(model, path):
    impotances = None
    if hasattr(model, 'feature_importances_'):
        impotances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        print('using coef_')
        impotances = model.coef_.ravel()  # coef需要转成一维
    else:
        print('the model can\'t select features')

    impotances = pd.Series(impotances).sort_values(ascending=False)  # 先对重要程度进行排序

    pie.plot_one([i for i in range(len(model.feature_importances_))], impotances.values, path)  # 只看特征是否平滑


# 可视化 将重要性聚合起来 得出每个文件的重要性贡献
# def plot_feature_importance_per_file(model, path, x_file_name):
#     feat_imp = pd.Series(model.feature_importances_)  # 使用pd是为了保留索引关系
#
#     # 统计每个文件上的特征的重要性 使用数组模拟字典
#     importance_per_file = [0 for _ in range(9)]
#
#     for i, v in feat_imp.items():  # 遍历这个Series
#
#         importance_per_file[locate_x_file_by_index(i)] += v
#
#
#     pie.plot_in_bar(x_file_name, importance_per_file, path)


# 绘制混淆矩阵，用于分类
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# 绘制多分类（可相加兼容2分类）的ROC曲线，并且标注了AUC
def plot_roc_with_auc(model, X_train, X_test, y_train, y_test):
    # 引入必要的库
    import numpy as np
    import matplotlib.pyplot as plt
    from itertools import cycle

    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    from sklearn.multiclass import OneVsRestClassifier
    from scipy import interp

    # 将标签二值化
    y_test = label_binarize(y_test, classes=[1, 2, 3, 4, 5, 6])
    y_train = label_binarize(y_train, classes=[1, 2, 3, 4, 5, 6])

    print(y_test)

    # 设置种类
    n_classes = 6

    # Learn to predict each class against the other
    classifier = OneVsRestClassifier(xgboost.XGBClassifier())

    print(X_train.shape, y_train.shape)

    y_pred = classifier.fit(X_train, y_train). \
        predict_proba(X_test)

    # 计算每一类的ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area（方法二）
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area（方法一）
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    lw = 2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (auc = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (auc = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (auc = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.savefig('./log/picture')
    plt.show()


def plot_true_pred(y_true, y_pred, yi):
    plt.style.use('fivethirtyeight')
    print(y_pred)
    print(y_true)
    xticksig = np.arange(1, 1 + len(y_pred)).astype(dtype=np.str)

    plt.plot(xticksig, y_pred, label='y_pred', linestyle=':', linewidth=2)
    plt.plot(xticksig, y_true, label='y_true', linewidth=2)
    plt.xlabel('test sample')
    plt.ylabel('value')
    plt.grid(False)
    plt.legend()

    plt.title(y_names[yi])

    plt.savefig('{}{}'.format(y_names[yi], '.png'))
    plt.show()

    df = pd.DataFrame()
    df['y_true'] = y_true
    df['y_pred'] = y_pred

    df.to_csv('log/{}true_pred.csv'.format(y_names[yi]))
    print('printed')
