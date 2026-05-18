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
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
import joblib

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


# 4. 训练模型 与 评估
def model_train(X, y, feat_cols):
    """
    1. 数据集切分
    2. 模型实例化（可选网格搜索）
    3. 模型训练
    4. 模型评估
    5. 模型保存
    """
    # 1. 数据集切分
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # log1p 变换：压缩右偏分布，让模型更容易拟合
    y_train = np.log1p(y_train)

    # 直接用调好的参数
    param_grid = {
        "n_estimators": [100, 300, 500],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1, 0.2],
    }
    gs = GridSearchCV(
        XGBRegressor(random_state=42),
        param_grid,
        cv=5,
        scoring="neg_mean_absolute_error",
    )
    gs.fit(x_train, y_train)
    print("最优参数:", gs.best_params_)

    estimator = gs.best_estimator_
    estimator.fit(x_train, y_train)
    # expm1 还原 log1p 变换，在原始尺度上评估
    y_pred = np.expm1(estimator.predict(x_test))

    print(f"MAE:  {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"RMSE: {root_mean_squared_error(y_test, y_pred):.4f}")
    print(f"R²: {r2_score(y_test, y_pred):.4f}")

    model_path = os.path.join(BASE_DIR, "model", "bike_xgb.pkl")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(estimator, model_path)
    print(f"模型已保存: {model_path}")


# 5. 预测并保存结果
def predict(estimator, test_data, feat_cols):
    """
    1. 对 test_data 做特征工程
    2. 列对齐
    3. 预测
    4. 保存 id + y 到 result.csv
    """
    test_x, _, _ = feature_engineering(test_data)
    # 重新索引特征列, 保证特征对齐
    test_x = test_x.reindex(columns=feat_cols, fill_value=0)
    y_pred = estimator.predict(test_x)
    # 预测值取整
    y_pred = np.round(np.expm1(y_pred)).astype(int)

    # 保存预测结果
    res = pd.DataFrame({"id": test_data["id"], "y": y_pred})
    res_path = os.path.join(BASE_DIR, "data", "result.csv")
    os.makedirs(os.path.dirname(res_path), exist_ok=True)
    res.to_csv(res_path, index=False)
    print(f"预测结果已保存: {res_path}")


if __name__ == "__main__":
    train_data, test_data = load_data()
    # ana_data(train_data)
    train_x, y, feat_cols = feature_engineering(train_data)

    # model_train(train_x, y, feat_cols)
    predict(joblib.load("../model/bike_xgb.pkl"), test_data, feat_cols)
