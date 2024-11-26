import warnings

warnings.filterwarnings("ignore")  # 不显示warning 方便采集结果（warning就只有版本差异的问题）

y_column_index = 7  # 第七列代表任务2 n分类

x_columns = None

# test the merge
if __name__ == '__main__':
    # x, y, x_cols = load_for_mission2()
    #
    # print(x)
    import xgboost
    import shap

    # load JS visualization code to notebook
    shap.initjs()

    """训练 XGBoost """

    X, y = shap.datasets.boston()
    model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100)

    """
    通过SHAP值来解释预测值
    (同样的方法也适用于 LightGBM, CatBoost, and scikit-learn models)
    """

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # 可视化解释性 (use matplotlib=True to avoid Javascript)
    # shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :],show=True,matplotlib=True)

    # shap.dependence_plot("RM", shap_values, X,show=True)
    # shap.summary_plot(shap_values, X,show=True)
    #
    # model_reuse =

    # emb_model.reuse_on_xgboost(model_for_selection=model_reuse)  # 然后再使用模型调参看能达到什么程度
    # emb.reuse_on_svm(model_for_selection=model_reuse)  # 然后再使用模型调参看能达到什么程度

    # tom = Try_on_model(x, y)
    # tom.try_on_models()
