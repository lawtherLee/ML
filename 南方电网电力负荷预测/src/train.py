# -*- coding: utf-8 -*-
import os
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from utils.log import Logger
from utils.common import data_preprocessing
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

plt.rcParams["font.size"] = 15
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "Microsoft YaHei"]


# 1. 定义电力负荷模型类, 配置日志, 获取数据源
class PowerLoadModel:
    # 1.1 初始化属性信息
    def __init__(self):
        # 1.2 拼接日志文件名
        logfile_name = "train_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        # 1.3 创建日志对象
        self.logfile = Logger("../", log_name=logfile_name).get_logger()
        # 1.4 获取数据源
        self.data_source = data_preprocessing()


# 2. 查看数据的整体分布情况(数据分析)
def ana_data(data):  # analysis: 分析
    """
    1.查看数据整体情况
    2.负荷整体的分布情况
    3.各个小时的平均负荷趋势，看一下负荷在一天中的变化情况
    4.各个月份的平均负荷趋势，看一下负荷在一年中的变化情况
    5.工作日与周末的平均负荷情况，看一下工作日的负荷与周末的负荷是否有区别
    :param data: 数据源
    :return:
    """
    _data = data.copy()
    # 1. 查看数据整体情况
    _data.info()
    # 先将 time 列转换为 datetime 类型
    _data["time"] = pd.to_datetime(_data["time"])
    # 2. 负荷整体的分布情况 直方图
    # 2.1 创建画布
    fig = plt.figure(figsize=(10, 20))
    # 2.2 创建子图
    ax1 = fig.add_subplot(4, 1, 1)
    ax1.set_title("负荷整体的分布情况")
    ax1.set_xlabel("负荷")
    ax1.set_ylabel("频数")
    ax1.hist(_data["power_load"], bins=100)

    # 3. 各个小时的平均负荷趋势，看一下负荷在一天中的变化情况
    # 3.1 新增一列, 充当小时
    _data["hour"] = _data["time"].dt.hour
    # 3.2 根据小时分组, 计算平均值
    hour_load_mean = _data.groupby("hour", as_index=False)["power_load"].mean()
    # 3.3 绘制折线图
    ax2 = fig.add_subplot(4, 1, 2)
    ax2.set_title("各个小时的平均负荷趋势")
    ax2.plot(hour_load_mean["hour"], hour_load_mean["power_load"])
    ax2.set_xlabel("小时")
    ax2.set_ylabel("平均负荷")

    # 4. 各个月份的平均负荷趋势，看一下负荷在一年中的变化情况
    # 4.1 新增一列, 充当月份
    _data["month"] = _data["time"].dt.month
    # 4.2 根据月份分组, 计算平均值
    month_load_mean = _data.groupby("month", as_index=False)["power_load"].mean()
    # 4.3 绘制折线图
    ax3 = fig.add_subplot(4, 1, 3)
    ax3.set_title("各个月份的平均负荷趋势")
    ax3.plot(month_load_mean["month"], month_load_mean["power_load"])
    ax3.set_xlabel("月份")
    ax3.set_ylabel("平均负荷")

    # 5. 工作日与周末的平均负荷情况，看一下工作日的负荷与周末的负荷是否有区别
    # 5.1 新增一列, 充当工作日与周末
    _data["weekday"] = _data["time"].dt.weekday + 1
    _data["is_holiday"] = _data["weekday"].apply(lambda x: 1 if x in [6, 7] else 0)
    # 5.2 根据工作日与周末分组, 计算平均值
    work_load_mean = _data[_data["is_holiday"] == 0]["power_load"].mean()
    holiday_load_mean = _data[_data["is_holiday"] == 1]["power_load"].mean()
    # 5.3 绘制柱状图
    ax4 = fig.add_subplot(4, 1, 4)
    ax4.set_title("工作日与周末的平均负荷情况")
    ax4.bar(
        x=["工作日", "周末"],
        height=[work_load_mean, holiday_load_mean],
    )
    ax4.set_xlabel("工作日与周末")
    ax4.set_ylabel("平均负荷")
    plt.tight_layout()
    plt.show()


# 3. 特征工程(重点)
def feature_engineering(data, logger):
    """
    对给定的数据源，进行特征工程处理，提取出关键的特征
    1.提取出时间特征：小时、月份
    2.提取出相近时间窗口中的负荷特征：step大小窗口的负荷
    3.提取昨日同时刻负荷特征
    4.剔除出现空值的样本
    5.整理时间特征，并返回
    :param data: 数据源
    :param logger: 日志
    :return:
    """
    # 1. 提取出时间特征：小时、月份
    # 2. 提取出相近时间窗口中的负荷特征：step大小窗口的负荷
    # 3. 提取昨日同时刻负荷特征
    # 4. 剔除出现空值的样本
    # 5. 整理时间特征，并返回
    pass


# 5. 模型训练, 评估

if __name__ == "__main__":
    plm = PowerLoadModel()
    # ana_data(plm.data_source)
    feature_engineering(plm.data_source, plm.logfile)
