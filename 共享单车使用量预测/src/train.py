"""
案例:
    公共自行车使用量预测 - 回归
回顾:
    有监督学习, 特征 + 连续/计数标签 y
    评估常用: MAE, MSE, RMSE
    流程: 加载 -> 预处理/特征工程 -> 训练 -> 评估 -> 预测
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.size"] = 15
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "Microsoft YaHei"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR, "自行车数据集", "train.csv")
TEST_PATH = os.path.join(BASE_DIR, "自行车数据集", "test.csv")


# 1. 加载数据
def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    return train, test


# 2. 探索性分析
def ana_data(train):
    """
    1. 查看数据整体情况 info / describe
    2. y 的分布（直方图）
    3. 各 hour 的平均 y（折线图）——看早晚高峰
    4. 工作日 vs 非工作日 的平均 y（柱状图）
    5. 各 weather 的平均 y（柱状图）
    6. 两 city 的 y 分布或均值对比
    :param train: 训练集
    """
    _data = train.copy()

    fig = plt.figure(figsize=(10, 24))

    # 2. y的分布
    ax1 = fig.add_subplot(5, 1, 1)
    ax1.set_title("租车量 y 的分布")
    ax1.hist(_data["y"], bins=50)
    ax1.set_xlabel("租车量")
    ax1.set_ylabel("频数")

    # 3. 各 小时 的平均 租车量
    hour_mean = _data.groupby("hour", as_index=False)["y"].mean()
    ax2 = fig.add_subplot(5, 1, 2)
    ax2.set_title("各小时平均租车量")
    ax2.plot(hour_mean["hour"], hour_mean["y"])
    ax2.set_xlabel("小时")
    ax2.set_ylabel("平均租车量")
    ax2.set_xticks(range(24))

    # 4. 工作日 vs 非工作日 的平均租车量
    work_mean = _data.groupby("is_workday")["y"].mean()
    ax3 = fig.add_subplot(5, 1, 3)
    ax3.set_title("工作日与周末的平均租车量")
    ax3.bar(["周末(0)", "工作日(1)"], [work_mean.get(0), work_mean.get(1)])

    # 5. 各 weather 的平均租车量
    weather_mean = _data.groupby("weather", as_index=False)["y"].mean()
    ax4 = fig.add_subplot(5, 1, 4)
    ax4.set_title("各天气状况的平均租车量")
    ax4.bar(weather_mean["weather"], weather_mean["y"])
    ax4.set_xticks(weather_mean["weather"])
    ax4.set_xticklabels(["晴(1)", "阴(2)", "小雨(3)", "大雨(4)"])

    # 6. 两城市平均租车量
    city_mean = _data.groupby("city")["y"].mean()
    ax5 = fig.add_subplot(5, 1, 5)
    ax5.set_title("各城市租车量")
    ax5.bar(["city=0", "city=1"], city_mean.values)

    plt.tight_layout()
    fig_dir = os.path.join(BASE_DIR, "data", "fig")
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, "租车量分布.png"))
    plt.show()


# 3. 特征工程
def feature_engineering(data):
    """
    1. 去掉 id
    2. 分离 y（测试集没有 y）
    3. hour 周期编码
    4. 温度相关处理
    5. weather、city 做 one-hot
    """
    _data = data.copy()

    # 1. 去掉 id
    _data.drop(columns=["id"], inplace=True)
    # 2. 分离标签：train 有 y，test 没有
    y = _data.pop("y") if "y" in _data.columns else None
    # 3. hour 周期编码：0点和23点在「钟面」上挨着
    _data["hour_sin"] = np.sin(2 * np.pi * _data["hour"] / 24)
    _data["hour_cos"] = np.cos(2 * np.pi * _data["hour"] / 24)
    _data.drop(columns=["hour"], inplace=True)
    # 4. 温度相关处理
    _data["temp_diff"] = _data["temp_2"] - _data["temp_1"]
    _data.drop(columns=["temp_2"], inplace=True)
    # 5. weather、city 做 one-hot
    _data = pd.get_dummies(_data, columns=["weather", "city"], dtype=int)

    feat_cols = list(_data.columns)
    return _data, y, feat_cols


if __name__ == "__main__":
    train_data, test_data = load_data()
    # ana_data(train_data)
    data, y, feat_cols = feature_engineering(train_data)
