# 医学数据分析

通过生物指标、Alpha多样性、微生物和代谢等，预测血脂及特征重要性分析。

[//]: # ()

[//]: # (#### 软件架构)

[//]: # (软件架构说明：)

[//]: # (![输入图片说明]&#40;https://images.gitee.com/uploads/images/2021/0525/174507_785074b7_9078337.png "不同文件说明和目前规划.png"&#41;)

[//]: # ()

[//]: # (---)

[//]: # ()

[//]: # (关于yi y0-y4 代表任务1前5列，)

[//]: # ()

[//]: # (y6表示对intervention，是否为mlk的2分类)

[//]: # ()

[//]: # (y7表示任务2，对intervention的n分类)

[//]: # ()

[//]: # (y8表示任务3，对diet的2分类)


使用自动机器学习刷比之前高很多的性能，详见readme
按照 ['CHO', 'TG', 'HDL', 'LDL', 'APOB'] 的顺序 在不可见的测试集上的性能为：


{'r2': 0.9550677446754604, 'root_mean_squared_error': -0.21318581595514277, 'mean_squared_error':
-0.04544819212446001, 'mean_absolute_error': -0.15956909832201505, 'pearsonr': 0.9837115407078194, '
median_absolute_error': -0.12187183380126942}


{'r2': 0.7241573839337527, 'root_mean_squared_error': -0.2847731420716417, 'mean_squared_error': -0.08109574244535545, '
mean_absolute_error': -0.2216621920936986, 'pearsonr': 0.9352529746675574, 'median_absolute_error':
-0.16359610557556148}


{'r2': 0.842763417165714, 'root_mean_squared_error': -0.15543679981938432, 'mean_squared_error':
-0.024160598738091354, 'mean_absolute_error': -0.12517768809669896, 'pearsonr': 0.9395541872644874, '
median_absolute_error': -0.09266549110412603}


{'r2': 0.9495099820029773, 'root_mean_squared_error': -0.0458027860851577, 'mean_squared_error':
-0.0020978952131627156, 'mean_absolute_error': -0.03445201917698509, 'pearsonr': 0.9850147256623284, '
median_absolute_error': -0.028078222274780296}


{'r2': 0.6729471288420752, 'root_mean_squared_error': -0.11529223047952887, 'mean_squared_error':
-0.013292298408944806, 'mean_absolute_error': -0.08142746435968497, 'pearsonr': 0.9495674752370328, '
median_absolute_error': -0.05315562486648562}