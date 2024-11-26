import glob
import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def draw(best_quality_or_interpretable, stage_num=-1):
    print('attention：call cat them all before')
    lis = glob.glob(f'./stage*/{best_quality_or_interpretable}')
    # print(lis)
    l = 0
    y_names = ['CHO', 'TG', 'HDL', 'LDL', 'APOB']

    if stage_num == -1:
        stage_num = 0
        for li in lis:
            if len(li.split('/')[-2]) == len('stage7'):
                l = 1
            elif len(li.split('/')[-2]) == len('stage10'):
                l = 2
            else:
                continue
            if not os.path.exists(f'{li}/all.txt'):
                continue

            stage_num += 1
    print('stage_num', stage_num)

    r2_all = []
    for li in [f'./stage{i}/{best_quality_or_interpretable}' for i in range(stage_num)]:
        print(os.path.join(li, 'all.txt'))
        with open(os.path.join(li, 'all.txt')) as all_f:
            line = [dict(eval(r)) for r in all_f.readlines()]

            r2s = [di['r2'] for di in line]
            r2_all.append(r2s)
        all_f.close()

        plt.plot(y_names, r2s, label=f"{li.split('/')[1]}")

    plt.legend()
    plt.show()

    r2_all = np.array(r2_all)
    for y_index in range(5):
        # print(f'y_names: {y_names[y_index]},{np.argmax(r2_all[:, y_index])}')
        plt.plot([i for i in range(r2_all.shape[0])], r2_all[:, y_index])
        plt.tittle = y_names[y_index]

        plt.show()

    data_frame = pd.DataFrame(r2_all, [f'stage_{i}' for i in range(stage_num)], y_names)
    # 找出每列最大值所有的stage号（用argmax）,并且将这个结果放在最后一行 去除掉stage_这个前缀
    data_frame.loc['max_stage'] = data_frame.idxmax(axis=0)

    # 统计每一列第三大减去第三小的值，加载到最后一行
    # data_frame.loc['max-min'] = data_frame.apply(lambda x: x.nlargest(3).values[0] - x.nsmallest(3).values[0], axis=0)

    data_frame.loc['max_stage'] = data_frame.loc['max_stage'].apply(lambda x: x.split('_')[1]) #这个apply是将每个元素都进行一次函数的操作
    data_frame.to_csv(os.path.join(best_quality_or_interpretable,'r2_all_stage.csv'))
    print('save finish')







if __name__ == '__main__':
    draw(best_quality_or_interpretable='interpretable',stage_num=20)
