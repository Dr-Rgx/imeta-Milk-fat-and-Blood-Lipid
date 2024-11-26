XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bynode=1, colsample_bytree=0.87, gamma=0.1, gpu_id=-1,
       importance_type='gain', interaction_constraints='',
       learning_rate=0.13, max_delta_step=0, max_depth=2,
       min_child_weight=1, missing=nan, monotone_constraints='()',
       n_estimators=21, n_jobs=16, num_parallel_tree=1,
       objective='binary:logistic', random_state=123, reg_alpha=0.011,
       reg_lambda=0.06, scale_pos_weight=1, subsample=0.71,
       tree_method='exact', use_label_encoder=True, validate_parameters=1,
       verbosity=None)
此次获得最佳参数为： {}
thre 0.04 最佳模型得分: 0.9759036144578314 测试集上的r2为： 1.0

Process finished with exit code 0
