import pyecharts.options as opts
from pyecharts.charts import Line
from pyecharts import options as opts
from pyecharts.charts import Bar
from pyecharts.commons.utils import JsCode
from pyecharts.globals import ThemeType

"""
可交互的图形，数据多了更容易去细分找到极值,不然plt可能一堆糊的
目前主要用来绘制训练集和测试集上的线条，代替plt
第一次使用 pip(3) install pyecharts
"""

# 输入x y1 y2 进行可交互的绘图
def plot(x_axis: list, y1: list, y2: list, save_path: str):
    x_axis = ['%f' % k for k in x_axis]  # 注意这里要输入字符串才行
    y1_plot = ['%.3f' % i for i in y1]  # 保留3位小数
    y2_plot = ['%.3f' % i for i in y2]

    (
        Line(init_opts=opts.InitOpts(width="1600px", height="800px"))
        .add_xaxis(xaxis_data=x_axis)

        .add_yaxis(
            series_name="train_score",
            y_axis=y1_plot,
            markpoint_opts=opts.MarkPointOpts(
                data=[
                    opts.MarkPointItem(type_="max", name="最大值")
                    # opts.MarkPointItem(type_="min", name="最小值"),
                ]
            ),
            markline_opts=opts.MarkLineOpts(
                data=[opts.MarkLineItem(type_="average", name="平均值")]
            ),
        )

        .add_yaxis(
            series_name="test_score",
            y_axis=y2_plot,
            markpoint_opts=opts.MarkPointOpts(
                data=[
                    opts.MarkPointItem(type_="max", name="最大值")
                    # opts.MarkPointItem(type_="min", name="最小值"),
                ]
            ),
            markline_opts=opts.MarkLineOpts(
                data=[opts.MarkLineItem(type_="average", name="平均值")]
            ),
            # markpoint_opts=opts.MarkPointOpts(
            #     data=[opts.MarkPointItem(value=-2, name="最低", x=1, y=-1.5)]
            # ),
            # markline_opts=opts.MarkLineOpts(
            #     data=[
            #         opts.MarkLineItem(type_="average", name="平均值"),
            #         opts.MarkLineItem(symbol="none", x="90%", y="max"),
            #         opts.MarkLineItem(symbol="circle", type_="max", name="最高点"),
            #     ]
            # ),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="此次分析结果", subtitle=""),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            toolbox_opts=opts.ToolboxOpts(is_show=True),
            xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False),
        )
        .render(save_path)
    )

def plot_one(x_axis: list, y1: list, save_path: str):
    # x_axis = ['%f' % k for k in x_axis]  # 注意这里要输入字符串才行
    y1_plot = ['%.3f' % i for i in y1]  # 保留3位小数

    (
        Line(init_opts=opts.InitOpts(width="1600px", height="800px"))
        .add_xaxis(xaxis_data=x_axis)

        .add_yaxis(
            series_name='importance',
            y_axis=y1_plot,
            markpoint_opts=opts.MarkPointOpts(
                data=[
                    opts.MarkPointItem(type_="max", name="最大值")
                    # opts.MarkPointItem(type_="min", name="最小值"),
                ]
            ),
            markline_opts=opts.MarkLineOpts(
                data=[opts.MarkLineItem(type_="average", name="平均值")]
            ),
        )

        .set_global_opts(
            title_opts=opts.TitleOpts(title="此次分析结果", subtitle=""),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            toolbox_opts=opts.ToolboxOpts(is_show=True),
            xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False),
        )
        .render(save_path)
    )


def plot_in_bar(x_axis: list, y1: list, save_path: str):  # 柱状图

    y1 = ['%.3f' % i for i in y1]  # 保留3位小数

    print('x_axis', x_axis)
    print(len(x_axis), len(y1))

    c = (
        Bar()
            .add_xaxis(x_axis)
            .add_yaxis("贡献", y1)
            .set_global_opts(
            xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=-15)),
            title_opts=opts.TitleOpts(title="每个X文件上的特征贡献度", subtitle=""),
            toolbox_opts=opts.ToolboxOpts(is_show=True)
        )
            .render(save_path)
    )