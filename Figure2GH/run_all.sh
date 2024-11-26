#task 1-4 在传统模型上的尝试
python train_for_reg_simple.py

# task 1-4 复现 基于筛选 集成学习的XGBRegressor的结果
python try_on_models.py

#task 6、7、8、9、10   其中task9结果在  log/blood_fat_met_results/task9/alpha_and_CHO.etc_（相关系数）.csv
python do_Test.py

# 基于自动机器学习的结果 将所有特征列在一起并且逐渐总结特征置换重要度 然后筛选 然后做shap 迭代20轮

python Auto_ML_for_reg.py