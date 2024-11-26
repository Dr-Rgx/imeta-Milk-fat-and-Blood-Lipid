import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from load_data import Load_for_blood_fat_meta

data = Load_for_blood_fat_meta()

"""
这里用于封装T检测的的各个过程
"""


def task6():
    """
    Returns
    statisticfloat or array
    t-statistic.

    pvaluefloat or array
    Two-sided p-value. 双侧表示单独的差异。无论大小

    """
    pd_n, pd_h = data.task6()

    yis = ['CHO', 'TG', 'HDL', 'LDL', 'APOB']
    # print('length:', data_h.shape, data_n.shape)

    for yi in yis:
        print()

        data_n = pd_n[yi]
        data_h = pd_h[yi]
        data_n = data_n[:47]

        print('now analysis on:', yi)
        s, p = stats.ttest_rel(data_n, data_h)
        print(p)
        # print('the var que:',stats.levene(data_n,data_h))

        plt.scatter([i for i in range(data_n.shape[0])], data_n, color='red', label='diet = N')
        plt.scatter([i for i in range(data_h.shape[0])], data_h, color='green', label='diet = H')
        plt.title('%s , p value = %.3f' % (yi, p))
        plt.legend()
        plt.savefig('log/blood_fat_met_results/task6/%s.png' % yi)
        plt.show()


def task7():
    yis = ['CHO', 'TG', 'HDL', 'LDL', 'APOB']
    x, control_name, effected_names = data.task7()

    control = x[x['interventions'] == control_name]  # 这是对照组

    res = pd.DataFrame(columns=yis, index=[effected_names])

    compare_res = pd.DataFrame(columns=yis, index=[effected_names])

    for effected in effected_names:  # 对于每个加了影响的组

        interventions = x[x['interventions'] == effected]

        # cas 13 ctl 16 fat 17 lef 14 mlk 15 whp 20  所以全部取13个，才能进行配对样本对照

        for yi in yis:
            print('now analysis on effect:', effected, 'yi:', yi, '.............')  # 取出了每个控制分组

            print('compare:', 'higher' if np.mean(interventions[yi][:13]) > np.mean(control[yi][:13]) else 'lower')

            s, p = stats.ttest_rel(interventions[yi][:13], control[yi][:13])  # 加入影响之后的和控制组进行t检验

            res[yi].loc[effected] = p
            compare_res[yi].loc[effected] = 'higher' if np.mean(interventions[yi][:13]) > np.mean(
                control[yi][:13]) else 'lower'

            # print(p)

        print()

    print(res)
    print(compare_res)


def task8():
    yis = ['CHO', 'TG', 'HDL', 'LDL', 'APOB']

    # 与task7 不同的是 这里经过了两次筛选

    for diet in ['H', 'N']:
        x, control_name, effected_names = data.task8()

        print('now the diet is :', diet)

        x = x[x['diet'] == diet]  # 这里对diet进行分类

        control = x[x['interventions'] == control_name]  # 这是对照组

        res = pd.DataFrame(columns=yis, index=[effected_names])

        compare_res = pd.DataFrame(columns=yis, index=[effected_names])

        for effected in effected_names:  # 对于每个加了影响的组

            interventions = x[x['interventions'] == effected]

            # cas 13 ctl 16 fat 17 lef 14 mlk 15 whp 20  所以全部取13个，才能进行配对样本对照

            for yi in yis:
                # print('now analysis on effect:', effected, 'yi:', yi, '.............')  # 取出了每个控制分组

                # print('compare:', 'higher' if np.mean(interventions[yi][:13]) > np.mean(control[yi][:13]) else 'lower')

                n_samples = min(len(interventions[yi]), len(control[yi]))  # 由这里定的截取长度，我们考虑两者最小的情况作为样本数目
                s, p = stats.ttest_rel(interventions[yi][:n_samples],
                                       control[yi][:n_samples])  # 加入影响之后的和控制组进行t检验 注意这里截取的长度也不同

                res[yi].loc[effected] = p
                compare_res[yi].loc[effected] = 'higher' if np.mean(interventions[yi][:13]) > np.mean(
                    control[yi][:13]) else 'lower'

                # print(p)

            print()

        print(res)
        print(compare_res)


def task10():
    yis = ['CHO', 'TG', 'HDL', 'LDL', 'APOB']

    x_y = data.task10()

    median = np.median(x_y['bodyweight'].values)

    # print(median)   # 中位数34.3

    thinner = x_y[x_y['bodyweight'] <= median]
    fatter = x_y[x_y['bodyweight'] > median]

    length = min(len(thinner), len(fatter))  # 差了3个 删除这3个
    thinner = thinner[:length]
    fatter = fatter[:length]

    # plt.scatter([i for i in range(length)], thinner[:length]['bodyweight'], color='red', label='thinner')
    # plt.scatter([i for i in range(length)], fatter[:length]['bodyweight'], color='g', label='fatter')
    # plt.legend()
    # plt.show()  # 验证了中位数的划分作用

    # mlk_set = x_y[x_y['interventions'] == 'Mlk']
    # print(mlk_set.shape)
    # thinner = mlk_set[mlk_set['bodyweight'] <= median]
    # fatter = mlk_set[mlk_set['bodyweight'] > median]  # 这说明mlk组 在胖瘦上划分均等

    y = 'APOB'
    thin_mlk = thinner[thinner['interventions'] == 'Mlk']  # 8
    thin_ctl = thinner[thinner['interventions'] == 'Ctl']  # 10
    thin_ctl = thin_ctl[:8]  # 数量对齐作对比

    b = 'greater'
    l = 'lower'  # 这是为了在表达式里方便判断

    fat_mlk = fatter[fatter['interventions'] == 'Mlk']  # 7
    fat_ctl = fatter[fatter['interventions'] == 'Ctl']  # 6  样本数大致均衡
    fat_mlk = fat_mlk[:6]  # 数量对齐作对比

    for y in yis:
        print(f'the {y} ,avg values(std) is {np.mean(x_y[y].values)}, median is {np.median(x_y[y].values)}')

        s, p = stats.ttest_rel(thin_mlk[y], thin_ctl[y], alternative='greater')
        print(f'now analysing on thin on {y},p = {p}, compare to ctl ,y in mlk is '
              f'{b if np.mean(thin_mlk[y].values) > np.mean(thin_ctl[y].values) else l}'
              f' the avg{y} is {np.mean(thin_mlk[y].values)} the set {y} is {thin_mlk[y].values}')

        s, p = stats.ttest_rel(fat_mlk[y], fat_ctl[y], alternative='greater')
        print(f'now analysing on {y} on fat ,p = {p}, compare to ctl ,y in mlk is '
              f'{b if np.mean(fat_mlk[y].values) > np.mean(fat_ctl[y].values) else l}, '
              f' the {y} is {np.mean(fat_mlk[y].values)} the set{y} is {fat_mlk[y].values}')

        print()

    # 接下来分别和正常体重的

    # plt.savefig('log/blood_fat_met_results/task6/%s.png' % yi)

    print('peek special', '-' * 10)

    # 看几个H组 但是瘦的
    s_ids = [11, 17, 159, 162, 167, 175, 183]
    data_with_id = data.get_all_data()
    s_data = data_with_id.loc[data_with_id['ID'].isin(s_ids)]
    print(s_data)
    s_data = s_data[yis]
    print(pd.DataFrame.mean(s_data))


if __name__ == '__main__':
    task6()
    task7()
    task8()
    data.task9_res()
    task10()
