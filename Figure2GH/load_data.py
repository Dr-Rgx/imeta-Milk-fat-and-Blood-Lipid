import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

"""
读取数据的都写在这里
返回类型目前都为
X:numpy, y:numpy, x.columns:numpy(列名)  可以直接放入sklearn训练

def load_metabolism()除外 返回的是 x和y的dataframe
"""

"""
merged.csv 融合了不包含x_diet.csv的所有部分和代谢指标 最后5列是5个y
因为前期的一些错误，x_f 被添加了两次，已经使用：
    X = X.T.drop_duplicates().T  进行删除
"""


# 从融合了所有x中的里面提取出代谢的数据 返回x y的dataframe
def load_metabolism():
    path = 'data/merged.csv'  # 位置在该路径下
    X = pd.read_csv(path)
    print(X.shape)
    x, y = X.iloc[:, 618:-5], X.iloc[:, -5:]
    return x, y


# load_metabolism()


# ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓任务1
# 读取需要回归的数据
def load_for_mission1(y_index: int):  # 根据传入的y_index 来返回列，
    print('now returning ', y_index)
    merged = pd.read_csv('data/merged.csv')
    print(merged.shape)

    X, y = merged.iloc[:, :-5], merged.iloc[:, -5:]  # 之前弄错了一个细节，df不能直接切片，要调用.iloc

    y = y.iloc[:, y_index]

    X = X.T.drop_duplicates().T  # 删除merged中的重复列

    return X.values, y.values, X.columns


# 读取X 不含代谢数据
def load_for_reg_without_newx():
    x_files = ['x_body.csv', 'x_alpha.csv', 'x_f.csv', 'x_c.csv', 'x_o.csv', 'x_diet.csv',
               'x_g1.csv', 'x_g2.csv', 'x_s.csv']

    x_pd = pd.DataFrame()  # 初始化空的pd 后面不断循环cat合并
    for file_name in x_files:
        file_path = os.path.join('data', file_name)  # 将名字拼接成相对路径
        x_pd_read = pd.read_csv(file_path)
        print('正在添加:', file_path, ' shape:', x_pd_read.shape)
        x_pd = pd.concat([x_pd, x_pd_read], axis=1)  # 1代表合并列

    y_pd = pd.read_csv('data/y.csv')

    return x_pd.values, y_pd.values, x_pd.columns


# ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓对mlk的2分类，任务2的基础
def load_for_clf():
    """
    预测diet中mlk 进行二分类
    对diet的interventions进行分类，返回可以直接训练的X和y
    返回的 x.columns 为列名，为了后面总结用
    """

    path = 'data/data_for_clf.csv'  # 位置在该路径下
    X = pd.read_csv(path)

    X = X.T.drop_duplicates().T  # 删除merged中的重复列

    x, y = X.iloc[:, :-5], X.iloc[:, -1]
    return x.values, y.values, x.columns


def load_for_clf_without_newX():
    """
    不带有代谢数据的X，用来预测diet中是否为mlk的情况
    新增：添加上diet的第一列也作为x
    """
    x_cols = []

    x_files = ['x_body.csv', 'x_alpha.csv', 'x_f.csv', 'x_c.csv', 'x_o.csv',
               'x_g1.csv', 'x_g2.csv', 'x_s.csv']

    x_pd = pd.DataFrame()  # 初始化空的pd 后面不断循环cat合并
    for file_name in x_files:
        file_path = os.path.join('data', file_name)  # 将名字拼接成相对路径
        x_pd_read = pd.read_csv(file_path)
        print('正在添加:', file_path, ' shape:', x_pd_read.shape)
        x_cols.append(x_pd_read.columns)
        x_pd = pd.concat([x_pd, x_pd_read], axis=1)  # 1代表合并列

    y = pd.read_csv('data/x_diet.csv')
    y_inv = y['interventions']  # 直接访问这一列
    for i in range(len(y_inv)):
        if y_inv[i] == 2:  # 如果为2 则为milk 则将其编码为1
            y_inv[i] = 1
        else:
            y_inv[i] = 0

    x_pd = pd.concat([x_pd, y['Diet']], axis=1)

    return x_pd.values, y_inv.values, x_pd.columns


# 5.30更新
# ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓任务 2
# 对diet中的mlk进行n分类 包含代谢指标
def load_for_mission2():
    """
    对diet中的mlk进行n分类
    包含代谢指标
    """
    path = 'data/merged.csv'  # 位置在该路径下
    merged = pd.read_csv(path)

    merged = merged.T.drop_duplicates().T  # 删除merged中的重复列

    y = merged['interventions']
    merged.drop(['interventions'], axis=1, inplace=True)

    print('now loading mission 2:n clf for interventions')

    x = merged.iloc[:, :-5]

    print('the number of nan：', x.isnull().sum().sum())

    return x.values, y.values, x.columns


# 对diet中的mlk进行n分类 不包含代谢指标
def load_for_mission2_without_newX():
    """
    不包含代谢数据的，对interventions的n分类问题
    """

    x_cols = []

    x_files = ['x_body.csv', 'x_alpha.csv', 'x_f.csv', 'x_f.csv', 'x_c.csv', 'x_o.csv',
               'x_g1.csv', 'x_g2.csv', 'x_s.csv']

    x_pd = pd.DataFrame()  # 初始化空的pd 后面不断循环cat合并
    for file_name in x_files:
        file_path = os.path.join('data', file_name)  # 将名字拼接成相对路径
        x_pd_read = pd.read_csv(file_path)
        print('正在添加:', file_path, ' shape:', x_pd_read.shape)
        x_cols.append(x_pd_read.columns)
        x_pd = pd.concat([x_pd, x_pd_read], axis=1)  # 1代表合并列

    y = pd.read_csv('data/x_diet.csv')
    y_inv = y['interventions']  # 直接访问这一列

    # 不进行编码了

    x_pd = pd.concat([x_pd, y['Diet']], axis=1)

    return x_pd.values, y.values, x_pd.columns


# ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓任务 3
# 对diet中的diet进行2分类 包含代谢指标
def load_for_mission3():
    """
    对diet中的diet 做2分类 因为已经 0 1 划分了 所以不用再继续划分了
    包含代谢指标
    """
    path = 'data/merged.csv'  # 位置在该路径下
    merged = pd.read_csv(path)

    merged = merged.T.drop_duplicates().T  # 删除merged中的重复列

    y = merged['Diet']
    merged.drop(['Diet'], axis=1, inplace=True)
    merged.drop(['Food consumption', 'protein', 'fat', 'carbohydrate'], axis=1, inplace=True)  # 对只有fat权重过高的修正

    x = merged.iloc[:, :-5]

    print('now loading mission 3:2 clf for interventions')

    return x.values, y.values, x.columns


# 对diet中的diet进行2分类 不包含代谢指标
def load_for_mission3_without_newX():
    """
    不包含代谢数据的，对diet中的diet 做2分类
    """

    x_cols = []

    x_files = ['x_body.csv', 'x_alpha.csv', 'x_f.csv', 'x_c.csv', 'x_o.csv',
               'x_g1.csv', 'x_g2.csv', 'x_s.csv']

    x_pd = pd.DataFrame()  # 初始化空的pd 后面不断循环cat合并
    for file_name in x_files:
        file_path = os.path.join('data', file_name)  # 将名字拼接成相对路径
        x_pd_read = pd.read_csv(file_path)
        print('正在添加:', file_path, ' shape:', x_pd_read.shape)
        x_cols.append(x_pd_read.columns)
        x_pd = pd.concat([x_pd, x_pd_read], axis=1)  # 1代表合并列

    y = pd.read_csv('data/x_diet.csv')
    y_inv = y['Diet']  # 直接换成这一列

    # 不进行编码了

    return x_pd.values, y_inv.values, x_pd.columns


class Load_for_blood_fat_meta:
    def __init__(self):
        self.path = 'data/blood_fat_meta.csv'

    def get_all_data(self):
        return pd.read_csv(self.path)

    def task1(self):
        """血脂和菌群有什么关系?"""
        x = pd.read_csv(self.path).loc[:, 'g__Ochrobactrum':'g__Pseudomonas']
        y = pd.read_csv(self.path, usecols=['CHO', 'TG', 'HDL', 'LDL', 'APOB'])

        return x.values, y.values, x.columns

    def task2(self):
        """血脂和代谢产物有什么关系？"""
        x = pd.read_csv(self.path).loc[:, 'meta_397':]
        y = pd.read_csv(self.path, usecols=['CHO', 'TG', 'HDL', 'LDL', 'APOB'])

        return x.values, y.values, x.columns

    def task3(self):
        """血脂和（菌群+代谢产物）有什么关系？"""
        x = pd.read_csv(self.path).loc[:, 'g__Ochrobactrum':]
        y = pd.read_csv(self.path, usecols=['CHO', 'TG', 'HDL', 'LDL', 'APOB'])

        return x.values, y.values, x.columns

    def task4(self, auto_g=False):
        """血脂和综合检测指标有什么关系？,无ID"""
        """diet:N-0,H-1"""
        x = pd.read_csv(self.path)
        # x["diet"] = pd.factorize(x["diet"])[0].astype(np.uint16)
        # x["interventions"] = pd.factorize(x["interventions"])[0].astype(np.uint16)

        y_col_names = ['CHO', 'TG', 'HDL', 'LDL', 'APOB']

        y = pd.read_csv(self.path, usecols=y_col_names)

        if auto_g:
            return x, y_col_names
        else:
            x = x.loc[:, 'diet':]
            # 将 'diet' 和 'interventions' 列转换为分类类型
            x['diet'] = x['diet'].astype('category')
            x['interventions'] = x['interventions'].astype('category')

            # 使用 get_dummies() 函数将 'diet' 和 'interventions' 列转换为 one-hot 编码
            x = pd.get_dummies(x, columns=['diet', 'interventions'])
            x = x.astype(float)

            return x.values, y.values, x.columns

    def task4_autogl(self, y_index):
        """血脂和综合检测指标有什么关系？,无ID"""
        """diet:N-0,H-1"""
        x = pd.read_csv(self.path)
        # x["diet"] = pd.factorize(x["diet"])[0].astype(np.uint16)
        # x["interventions"] = pd.factorize(x["interventions"])[0].astype(np.uint16)

        y_col_names = ['CHO', 'TG', 'HDL', 'LDL', 'APOB']
        y_col_name = y_col_names[y_index]

        # x和y分别取出来
        y = x[y_col_name]
        x = x.loc[:, 'diet':]

        print(f'load y_index:{y_index}, y:{y_col_name}')

        return pd.concat([x, y], axis=1), y_col_name

    def task5(self):
        """计算g__xxxx和meta_xxxx的spearman系数"""
        x = pd.read_csv(self.path).loc[:, 'g__Ochrobactrum':]
        x.corr('spearman').to_csv("log/blood_fat_met_results/task5/g_and_m_spearman.csv")

    def task9_res(self):
        """计算alpha多样性和CHO TG HDL LDL APOB的相关性，spearman系数？"""
        """未进行相关性分析"""
        x = pd.read_csv(self.path).loc[:, 'OTUs':'simpson']
        y = pd.read_csv(self.path, usecols=['CHO', 'TG', 'HDL', 'LDL', 'APOB'])
        new = pd.concat([x, y], axis=1)
        for method in ['pearson', 'kendall', 'spearman']:
            new.corr(method).to_csv("log/blood_fat_met_results/task9/alpha_and_CHO.etc_%s.csv" % method)

    def task6(self):
        """
        diet：N（原假设）和H（备选假设），T检验或？
        方法： Ttest检测 N 和 H 区别下的血脂差异的显著性水平

        如果使用机器学习方法，这个好像既不是回归问题也不是分类问题
        返回为N的部分和为H的部分 ,在外面进行T检验对比差异
        """

        data = pd.read_csv(self.path)

        yis = ['CHO', 'TG', 'HDL', 'LDL', 'APOB']

        yis = data[yis]

        diet = data['diet']

        x = pd.concat([diet, yis], axis=1)

        pd_n = x[x['diet'] == 'N']

        pd_h = x[x['diet'] == 'H']

        return pd_n, pd_h

    def task7(self):
        """
        不同 interventions 对血脂的影响
        思路 全部和对照组进行T检验  刚好 cas 13 ctl 16 fat 17 lef 14 mlk 15 whp 20 比较均匀且样本量小于30
        """

        data = pd.read_csv(self.path)

        yis = ['CHO', 'TG', 'HDL', 'LDL', 'APOB']

        yis = data[yis]

        interventions = data['interventions']

        x = pd.concat([interventions, yis], axis=1)

        # control = data[data['interventions'] == 'Ctl']  # 这组作为对照

        effected_names = ['Mlk', 'Fat', 'Whp', 'Cas', 'Ltf']  # 这些代表收到影响了的情况

        return x, 'Ctl', effected_names  # 返回 interventions和y拼接的数据矩阵， 对照组名字， 受到影响的名字

    def task8(self):
        """
        diet和interventions组合情况， 对血脂的影响

        和task7的区别就是多返回一个diet 然后思路区别就是分别对H 和 N 进行区分
        """

        data = pd.read_csv(self.path)

        yis = ['CHO', 'TG', 'HDL', 'LDL', 'APOB']

        yis = data[yis]

        d_in_ind = ['diet', 'interventions']

        diet_interventions = data[d_in_ind]

        x = pd.concat([diet_interventions, yis], axis=1)

        # control = data[data['interventions'] == 'Ctl']  # 这组作为对照

        effected_names = ['Mlk', 'Fat', 'Whp', 'Cas', 'Ltf']  # 这些代表收到影响了的情况

        return x, 'Ctl', effected_names  # 返回 interventions和y拼接的数据矩阵， 对照组名字， 受到影响的名字

    def task9(self):  # 尝试使用多样性来预测 看信息增益

        data = pd.read_csv(self.path)  #
        yis = ['CHO', 'TG', 'HDL', 'LDL', 'APOB']
        xis = ['OTUs', 'chao1', 'goods_coverage', 'observed_species', 'PD_whole_tree', 'shannon', 'simpson']

        y = data[yis]  # 这里包含了不同列

        x = data[xis]

        print('yis.shape', y.shape)

        print('x.shape', x.shape)

        return x.values, y.values

    def task10(self):
        """
        不同 interventions 对血脂的影响
        思路 全部和对照组进行T检验  刚好 cas 13 ctl 16 fat 17 lef 14 mlk 15 whp 20 比较均匀且样本量小于30
        """

        data = pd.read_csv(self.path)

        yis = ['CHO', 'TG', 'HDL', 'LDL', 'APOB']

        yis = data[yis]

        inter_bodyweight = data[['interventions', 'bodyweight']]

        x = pd.concat([inter_bodyweight, yis], axis=1)

        # control = data[data['interventions'] == 'Ctl']  # 这组作为对照

        # effected_names = ['Mlk', 'Fat', 'Whp', 'Cas', 'Ltf']  # 这些代表收到影响了的情况

        return x


# --------------------------对数据的预处理,划分
def data_process(X, y, need_scale=False, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123, test_size=0.2)

    # 进行归一化
    if need_scale:
        standardScaler = StandardScaler()
        standardScaler.fit(X_train)  # 都使用train的归一化标准
        X_train = standardScaler.transform(X_train)
        X_test = standardScaler.transform(X_test)

    return X_train, X_test, y_train, y_test
