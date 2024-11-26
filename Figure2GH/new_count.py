import json
import os

import pandas as pd

from load_data import Load_for_blood_fat_meta
import scipy.stats as stats




def p_1_2(x):
    def compare(group1, group2):
        results = {}
        for col in group1.columns:
            # 使用独立样本 t 检验比较每一个指标
            t_stat, p_val = stats.ttest_ind(group1[col], group2[col])
            results[col] = {
                'mean_difference': group1[col].mean() - group2[col].mean(),
                't_stat': t_stat,
                'p_val': p_val
            }
        return results

    def find_significant_results(res):
        # 找出所有p值小于0.05的情况
        significant_results = {}

        for group_name, results in res.items():
            for col, stats in results.items():
                if stats['p_val'] < 0.05:
                    # 如果该组还没有在significant_results字典中，就添加一个新的键
                    if group_name not in significant_results:
                        significant_results[group_name] = {}
                    # 将具有显著差异的结果添加到字典中
                    significant_results[group_name][col] = stats

        return significant_results

    def print_significant_results(significant_results):
        # 打印所有 p 值小于 0.05 的结果
        for group_name, results in significant_results.items():
            # 将结果按照 p 值的大小进行排序
            sorted_results = sorted(results.items(), key=lambda item: item[1]['p_val'])

            print(f"Significant results for {group_name} compared to control group:")
            for col, stats in sorted_results:
                print(
                    f"{col}: Mean Difference = {stats['mean_difference']}, t_stat = {stats['t_stat']}, p_val = {stats['p_val']}")

    def save_significant_results(significant_results, base_dir):
        for group_name, results in significant_results.items():
            # 将结果按照 p 值的大小进行排序
            sorted_results = sorted(results.items(), key=lambda item: item[1]['p_val'])

            # 创建保存结果的目录
            group_dir = os.path.join(base_dir, group_name)
            os.makedirs(group_dir, exist_ok=True)

            # 保存结果到文件中
            with open(os.path.join(group_dir, 'results.json'), 'w') as f:
                json.dump(sorted_results, f, indent=4)


    # 查看x的interventions所有可选的种类
    print(x['interventions'].unique())  # ['Ctl' 'Mlk' 'Fat' 'Whp' 'Cas' 'Ltf']
    # 筛选出x的interventions指标下 为milk的数据
    interventions = ['Ctl' 'Mlk' 'Fat' 'Whp' 'Cas' 'Ltf']
    Ctl = x[x['interventions'] == 'Ctl']
    Mlk = x[x['interventions'] == 'Mlk']
    Fat = x[x['interventions'] == 'Fat']
    Whp = x[x['interventions'] == 'Whp']
    Cas = x[x['interventions'] == 'Cas']
    Ltf = x[x['interventions'] == 'Ltf']
    groups = [Ctl, Mlk, Fat, Whp, Cas, Ltf]
    groups_names = ['Ctl', 'Mlk', 'Fat', 'Whp', 'Cas', 'Ltf']
    # 将Mlk的数据中筛选出meta_开头的特征
    for i, group in enumerate(groups):
        groups[i] = group[group.columns[group.columns.str.startswith('g_')]]


    res = {}  # 不同组对ctl的对比
    for i, group_i in enumerate(groups[1:], start=1):
        result = compare(group_i, groups[0])
        res[groups_names[i]] = result
    significant_results = find_significant_results(res)
    print_significant_results(significant_results)
    save_significant_results(significant_results, 'new_count/p2')


def p_3(x):
    # 假设 x 是你的 DataFrame，ys 是你关心的列的列表
    ys = ['CHO', 'TG', 'HDL', 'LDL', 'APOB']

    # 提取出 y 列
    ys_cols = x[ys]

    # 提取出所有以 'meta_' 开头的列
    meta_cols = x[x.columns[x.columns.str.startswith('g_')]]

    # 对每个 y 和每个 'meta_' 列进行迭代
    for y in ys_cols:
        results = []
        for meta_col in meta_cols:
            # 计算相关性
            correlation, p_value = stats.pearsonr(x[y], x[meta_col])

            # 如果 p 值小于或等于 0.05，那么保存到结果列表中
            if p_value <= 0.05:
                results.append({
                    'meta_column': meta_col,
                    'correlation': correlation,
                    'p_value': p_value
                })

        # 对结果列表按照 p 值进行排序，越小的排在越前面
        results.sort(key=lambda x: x['p_value'])

        saved_path = os.path.join('new_count', 'p3', 'g', f'{y}_results.json')
        os.makedirs(os.path.dirname(saved_path), exist_ok=True)
        # 保存为 JSON 文件，每个 y 一个文件
        with open(saved_path, 'w') as f:
            json.dump(results, f)
        print(f"Saved results for {y} to {saved_path}")




    ...


if __name__ == '__main__':
    loader = Load_for_blood_fat_meta()
    x = loader.get_all_data()
    p_3(x)
