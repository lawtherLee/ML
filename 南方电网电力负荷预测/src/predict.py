import joblib
import matplotlib.pyplot as plt
import datetime

import pandas as pd

from utils.common import data_preprocessing
from utils.log import Logger

plt.rcParams["font.size"] = 15
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "Microsoft YaHei"]


# 1. 配置电力负荷预测类
class PowerLoadPredict:
    def __init__(self, file_path):
        # 配置日志记录
        logfile_name = "predict_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.logger = Logger("../", logfile_name).get_logger()
        # 获取数据源
        self.data_source = data_preprocessing(file_path)
        # 历史数据转字典: key: 时间 value: 负荷 避免频繁操作dataframe
        self.time_load_dict = dict(
            zip(self.data_source["time"], self.data_source["power_load"])
        )


# 2. 预测数据解析特征, 保持与模型训练时的特征列名一致
def pred_feature_extract(data_dict, time, logger):
    """
    预测数据解析特征，保持与模型训练时的特征列名一致
    1.解析时间特征
    2.解析时间窗口特征
    3.解析昨日同时刻特征
    :param data_dict:历史数据，字典格式，key：时间，value:负荷
    :param time:预测时间，字符串类型，格式为2024-12-20 09:00:00
    :param logger:日志对象
    :return:
    """
    logger.info(f"=========解析预测时间为：{time}所对应的特征==============")
    # 特征列清单
    feature_names = [
        "hour_00",
        "hour_01",
        "hour_02",
        "hour_03",
        "hour_04",
        "hour_05",
        "hour_06",
        "hour_07",
        "hour_08",
        "hour_09",
        "hour_10",
        "hour_11",
        "hour_12",
        "hour_13",
        "hour_14",
        "hour_15",
        "hour_16",
        "hour_17",
        "hour_18",
        "hour_19",
        "hour_20",
        "hour_21",
        "hour_22",
        "hour_23",
        "month_01",
        "month_02",
        "month_03",
        "month_04",
        "month_05",
        "month_06",
        "month_07",
        "month_08",
        "month_09",
        "month_10",
        "month_11",
        "month_12",
        "前1小时",
        "前2小时",
        "前3小时",
        "yesterday_load",
    ]

    # 1. 解析时间特征, 即: time字段(预测时间)对应的数据样本
    # 1.1 截取要预测的time字段的 小时信息
    pred_hour = time[11:13]
    hour_list = []
    for i in range(24):
        if pred_hour == feature_names[i][5:7]:
            hour_list.append(1)
        else:
            hour_list.append(0)
    # print(hour_list)
    # 1.2 截取要预测的time字段的 月份信息
    pred_month = time[5:7]
    month_list = []
    for i in range(24, 36):
        if pred_month == feature_names[i][6:8]:
            month_list.append(1)
        else:
            month_list.append(0)
    # print(month_list)

    # 2. 解析时间窗口特征 - 前1/2/3小时的负荷
    last_1h_load, last_2h_load, last_3h_load = [
        data_dict.get(
            (pd.to_datetime(time) - pd.to_timedelta(f"{h}h")).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            500,
        )
        for h in [1, 2, 3]
    ]
    # 3. 解析昨日同时刻特征


# 4. 测试
if __name__ == "__main__":
    # 4.1 创建电力负荷预测类对象
    plp = PowerLoadPredict("../data/test.csv")
    # 4.2 加载模型对象
    estimator = joblib.load("../model/xgb_20260515.pkl")
    # 4.3 确定要预测的时间段（2015 - 08 - 01 00: 00:00及以后的时间）
    pred_times = plp.data_source["time"][
        plp.data_source["time"] >= "2015-08-01 00:00:00"
    ]
    # 4.4 为了模拟实际场景的预测，把要预测的时间以及以后的负荷都掩盖掉，因此新建一个数据字典，只保存预测时间以前的数据字典
    for pred_time in pred_times:
        print(f"正在预测时间：{pred_time}所对应的负荷...")
        time_load_dict_masked = {
            k: v for k, v in plp.time_load_dict.items() if k < pred_time
        }
        pred_feature_extract(time_load_dict_masked, pred_time, plp.logger)
    # 4.5 预测负荷
    # 4.5.1 解析特征（定义解析特征方法）
    # 4.5.2 利用加载的模型预测
    # 4.6 保存预测时间对应的真实负荷
    # 4.7 结果保存到evaluate_list，三个元素分别是预测时间、真实负荷、预测负荷，方便后续进行预测结果评价
    # 4.8 循环结束后，evaluate_list转为DataFrame
