# 医学数据分析

# 简介
本项目是一个医学数据分析的项目，主要用于医学数据的分析（预测和归因）

我们将任务分为10个子任务 分别为

血脂和菌群有什么关系？
## task1：g__xxxx 预测 CHO TG HDL LDL APOB 

血脂和代谢产物有什么关系？
## task2：使用meta_xxxx 预测 CHO TG HDL LDL APOB

血脂和（菌群+代谢产物）有什么关系？
## task3：使用g__xxxx和meta_xxxx 预测 CHO TG HDL LDL APOB

血脂和综合检测指标有什么关系？
## task4：使用diet,interventions,bodyweight,alpha多样性,g__xxxx和meta_xxxx，预测 CHO TG HDL LDL APOB


------------------离散化


菌群和代谢产物有什么关系？（前提，代谢产物与菌群有密切关系，但不都是由菌群产生，人体也可以）
task5：计算g__xxxx和meta_xxxx的spearman系数
## 不同的控制下（diet，interventions），血脂是否有明显差异？

## task6：     diet：N（原假设）和H（备选假设），T检验或？
    CHO HDL LDL 有明显变化 P值远小于0.05



## task7：interventions：
ctl（原假设）和其他（备选假设），T检验或？
*  （原假设）和其他（备选假设），T检验或？
ans:仅考虑不同影响造成的显著性上升/下降的影响 P<0.05
        若不控制 非control
        mlk让血脂的5个指标中的4个造成了上升
        fat让其中3上升
        whp和ltf让5个中的2个造成了影响。其中WHP让TG下降 其余皆为上升
       [README.md](..%2Fmedical-data-analysis%20-%20%B8%B1%B1%BE%2FREADME.md) 即：
        在控制的情况下 mlk可以让血脂4个指标的明显低于非控制的对照组

## task8：diet和interventions组合情况：不列举了。
ans:
    在节食的情况下，mlk fat whp会造成血脂的指标升高，
    其中出现偶然情况，whp让TG显著下降。
    而在不节食的情况下，主要是mlk让血脂升高。


血脂和alpha多样性，是否有关联？
## task9：计算alpha多样性和CHO TG HDL LDL APOB的相关性，spearman系数？
ans:代谢 菌群 内部相关高 两者之间无明显高相关性。无论使用哪种相关度系数都是如此。
    除了 goods_coverage 在交叉的相关系数里最高 数值大约为0.1左右 全为正数 其余几乎为负数

血脂和alpha多样性，是否有关联？
## task10：按照bodyweight分为两组 先画图看一下集中程度（假设是正态分布） 然后根据中心分组。在低体重组里看mlk的影响

# 代码完成的目标
完成上面所有task 并且添加了 使用自动机器学习的全预测方法

# 代码执行
首先 
```bash
pip install -r requirements.txt
```

然后根据 [run_all.sh](run_all.sh) 里注释的每一行 按需求执行即可
其中对于自动机器学习部分 shap的实验结果放在目录`log/autogluon_res/shap_res`  下