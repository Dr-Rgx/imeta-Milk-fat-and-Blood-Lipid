import glob
import os

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

csv_root_path='./'
# 找到该路径下所有的csv文件
csv_files = glob.glob(os.path.join(csv_root_path, '*.csv'))
def get_name(file_path):
    name = file_path.split('_')[6]
    return name



def polynomial_smooth(array, degree=6):
    # 获取数组长度
    n = len(array)

    # 生成x坐标
    x = np.arange(n)

    # 使用多项式拟合
    coefficients = np.polyfit(x, array, deg=degree)
    poly = np.poly1d(coefficients)

    # 计算拟合后的平滑数据
    smooth_array = poly(x)

    return smooth_array


def get_r2(r2_dict, name):
    return r2_dict[name]


if __name__ == '__main__':

    # fonts = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
    # print(fonts)
    import matplotlib as mpl

    mpl.rcParams['font.family'] = 'Times New Roman'
    # exit(2)
    r2_dict = {'CHO': 0.96, 'TG': 0.79, 'HDL': 0.89, 'LDL': 0.96, 'APOB': 0.73}

    for csv_file in csv_files:
        name = get_name(csv_file)
        csv = pd.read_csv(csv_file)
        y_pred = csv.y_pred.values
        y_real = csv.y_real.values

        r2 = get_r2(r2_dict, name)

        x = [i+1 for i in range(len(y_pred))]
        # y_real = polynomial_smooth(y_real)
        # y_pred = polynomial_smooth(y_pred)

        import matplotlib.pyplot as plt
        import numpy as np
        import matplotlib.ticker as ticker

        # Set the figure size to wider ratio
        fig = plt.figure(figsize=(6 * 1.618, 6))

        plt.ylim(0, np.max(y_real) * 1.5)

        # Define the width of the bars and the positions of the x ticks
        width = 0.35
        x_ticks = np.arange(len(x))

        # Increase the linewidth of the plotted bars and plot them side by side
        plt.bar(x_ticks - width / 2, y_pred, width, label='Predicted value', linewidth=3,
                color=(114 / 255, 134 / 255, 180 / 255))
        plt.bar(x_ticks + width / 2, y_real, width, label='True value', linewidth=3,
                color=(220 / 255, 103 / 255, 106 / 255))

        # Adjust the subplot layout to leave space for the title
        fig.subplots_adjust(left=0.2, bottom=0.2)

        # Add a vertical title on the left with bigger font size
        plt.text(-0.11, 0.5, f'{name} value(mmol/L)', horizontalalignment='center',
                 rotation=90, verticalalignment='center', fontsize=20,
                 transform=plt.gca().transAxes)

        # Get the current axes and increase the linewidth of the spines
        ax = plt.gca()

        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        for spine in ax.spines.values():
            spine.set_linewidth(3)

        # Increase the size of the tick labels
        plt.tick_params(axis='both', which='major', labelsize=20)

        # Set the x ticks positions and labels
        plt.xticks(x_ticks, x)

        plt.legend(fontsize=18, loc='upper right', frameon=False)
        plt.figtext(0.5, 0.1, f'Test cases,  R\u00B2 score={r2:.2f}', ha='center', va='center', fontsize=20)

        # 保存图片
        plt.savefig(f'{name}.png')
        plt.show()

print()