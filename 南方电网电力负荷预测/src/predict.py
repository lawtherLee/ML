# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import datetime
from utils.log import Logger
from utils.common import data_preprocessing
from sklearn.metrics import mean_absolute_error
import matplotlib.ticker as mick
import joblib
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['font.size'] = 15


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
    logger.info(f'=========解析预测时间为：{time}所对应的特征==============')
    # 特征列清单
    feature_names = ['hour_00', 'hour_01', 'hour_02', 'hour_03', 'hour_04', 'hour_05',
                     'hour_06', 'hour_07', 'hour_08', 'hour_09', 'hour_10', 'hour_11',
                     'hour_12', 'hour_13', 'hour_14', 'hour_15', 'hour_16', 'hour_17',
                     'hour_18', 'hour_19', 'hour_20', 'hour_21', 'hour_22', 'hour_23',
                     'month_01', 'month_02', 'month_03', 'month_04', 'month_05', 'month_06',
                     'month_07', 'month_08', 'month_09', 'month_10', 'month_11', 'month_12',
                     '前1小时', '前2小时', '前3小时', 'yesterday_load']
    # 小时特征数据，使用列表保存起来
    hour_part = []
    pred_hour = time[11:13]
    for i in range(24):
        if pred_hour == feature_names[i][5:7]:
            hour_part.append(1)
        else:
            hour_part.append(0)
    # 月份特征数据，使用列表保存起来
    month_part = []
    pred_month = time[5:7]
    for i in range(24, 36):
        if pred_month == feature_names[i][6:8]:
            month_part.append(1)
        else:
            month_part.append(0)
    # 历史负荷数据，使用列表保存起来
    his_part = []
    # 前1小时负荷
    last_1h_time = (pd.to_datetime(time) - pd.to_timedelta('1h')).strftime('%Y-%m-%d %H:%M:%S')
    last_1h_load = data_dict.get(last_1h_time, 600)
    # 前2小时负荷
    last_2h_time = (pd.to_datetime(time) - pd.to_timedelta('2h')).strftime('%Y-%m-%d %H:%M:%S')
    last_2h_load = data_dict.get(last_2h_time, 600)
    # 前3小时负荷
    last_3h_time = (pd.to_datetime(time) - pd.to_timedelta('3h')).strftime('%Y-%m-%d %H:%M:%S')
    last_3h_load = data_dict.get(last_3h_time, 600)

    # 昨日同时刻负荷
    last_day_time = (pd.to_datetime(time) - pd.to_timedelta('1d')).strftime('%Y-%m-%d %H:%M:%S')
    last_day_load = data_dict.get(last_day_time, 600)

    his_part = [last_1h_load, last_2h_load, last_3h_load, last_day_load]
    # 特征数据，包含小时特征数据，月份特征数据，历史负荷数据
    feature_list = [hour_part + month_part + his_part]
    # feature_list需要转成dataframe并返回，所以这里用append变成一个二维列表
    feature_df = pd.DataFrame(feature_list, columns=feature_names)
    return feature_df, feature_names


def prediction_plot(data):
    """
    绘制时间与预测负荷折线图，时间与真实负荷折线图，展示预测效果
    :param data: 数据一共有三列：时间、真实值、预测值
    :return:
    """
    # 绘制在新数据下
    fig = plt.figure(figsize=(40, 20))
    ax = fig.add_subplot()
    # 绘制时间与真实负荷的折线图
    ax.plot(data['时间'], data['真实值'], label='真实值')
    # 绘制时间与预测负荷的折线图
    ax.plot(data['时间'], data['预测值'], label='预测值')
    ax.set_ylabel('负荷')
    ax.set_title('预测负荷以及真实负荷的折线图')
    # 横坐标时间若不处理太过密集，这里调大时间展示的间隔
    ax.xaxis.set_major_locator(mick.MultipleLocator(50))
    # 时间展示时旋转45度
    plt.xticks(rotation=45)
    plt.legend()
    plt.savefig('../data/fig/预测效果.png')


class PowerLoadPredict(object):
    def __init__(self, filename):
        # 配置日志记录
        logfile_name = "predict_" + datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        self.logfile = Logger('../', logfile_name).get_logger()
        # 获取数据源
        self.data_source = data_preprocessing(filename)
        # 历史数据转为字典，key:时间，value:负荷，目的是为了避免频繁操作dataframe，提高效率。实际开发场景中可以使用redis进行缓存
        self.data_dict = self.data_source.set_index('time')['power_load'].to_dict()


if __name__ == '__main__':
    """
    模型预测
    1.导包、配置绘图字体
    2.定义电力负荷预测类，配置日志，获取数据源、历史数据转为字典（避免频繁操作dataframe，提高效率）
    3.加载模型
    4.模型预测
        4.1 确定要预测的时间段（2015-08-01 00:00:00及以后的时间）
        4.2 为了模拟实际场景的预测，把要预测的时间以及以后的负荷都掩盖掉，因此新建一个数据字典，只保存预测时间以前的数据字典
        4.3 预测负荷
            4.3.1 解析特征（定义解析特征方法）
            4.3.2 利用加载的模型预测
        4.4 保存预测时间对应的真实负荷
        4.5 结果保存到evaluate_list，三个元素分别是预测时间、真实负荷、预测负荷，方便后续进行预测结果评价
        4.6 循环结束后，evaluate_list转为DataFrame
    5.预测结果评价
        5.1 计算预测结果与真实结果的MAE
        5.2 绘制折线图（预测时间-真实负荷折线图，预测时间-预测负荷折线图），查看预测效果 
    """
    # 2.定义电力负荷预测类(PowerLoadPredict)，配置日志，获取数据源、历史数据转为字典（避免频繁操作dataframe，提高效率）
    input_file = os.path.join('../data', 'test.csv')
    pred_obj = PowerLoadPredict(input_file)
    # 3.加载模型
    model = joblib.load('../model/xgb.pkl')
    # 4.模型预测
    evaluate_list = []
    # 4.1确定要预测的时间段：2015-08-01 00:00:00及以后的时间
    pred_times = pred_obj.data_source[pred_obj.data_source['time'] >= '2015-08-01 00:00:00']['time']
    for pred_time in pred_times:
        print(f"开始预测时间为：{pred_time}的负荷")
        pred_obj.logfile.info(f"开始预测时间为：{pred_time}的负荷")
        # 4.2为了模拟实际场景的预测，把要预测的时间以及以后的负荷都掩盖掉，因此新建一个数据字典，只保存预测时间以前的数据字典
        data_his_dict = {k: v for k, v in pred_obj.data_dict.items() if k < pred_time}
        # 4.3预测负荷
        # 4.3.1解析特征
        processed_data, feature_cols = pred_feature_extract(data_his_dict, pred_time, pred_obj.logfile)
        # 4.3.2 模型预测
        pred_value = model.predict(processed_data[feature_cols])
        # 4.4真实负荷
        true_value = pred_obj.data_dict.get(pred_time)
        pred_obj.logfile.info(f"真实负荷为：{true_value}, 预测负荷为：{pred_value}")
        # 4.5结果保存到evaluate_list，三个元素分别是预测时间、真实负荷、预测负荷
        evaluate_list.append([pred_time, true_value, pred_value[0]])
    # 4.6evaluate_list转为DataFrame
    evaluate_df = pd.DataFrame(evaluate_list, columns=['时间', '真实值', '预测值'])
    # 5.预测结果评价
    # 5.1计算预测结果与真实结果的MAE
    mae_score = mean_absolute_error(evaluate_df['真实值'], evaluate_df['预测值'])
    print(f"模型对新数据进行预测的平均绝对误差：{mae_score}")
    pred_obj.logfile.info(f"模型对新数据进行预测的平均绝对误差：{mae_score}")
    # 5.2绘制折线图，查看预测效果
    prediction_plot(evaluate_df)
