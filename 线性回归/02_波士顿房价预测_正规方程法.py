"""
案例:
    演示 正规方程法 线性回归对象 完成 波士顿房价预测案例.

回顾:
    线性回归算法 属于 有监督学习之  有特征, 有标签, 且标签是连续的.
    线性回归分类:
        一元线性回归: 1个特征列, 1个标签列.
        多元线性回归: 多个特征列, 1个标签列.
    线性回归大白话解释:
       它是用线性公式来描述 特征 和 标签之间关系的, 方便做预测, 公式如下:
       一元线性回归: y = w * x + b
       多元线性回归: y = w1 * x1 + w2 * x2 + w3 * x3 + ... + wn * xn + b = w的转置 * x + b
    如何衡量线性回归模型的好坏?
        思路:
            预测值和真实值之间的误差, 误差越小, 模型越好 => 损失函数
        具体的方案:
            1. 最小二乘.        每个(样本)误差平方和
            2. 均方误差(MSE)    每个(样本)误差平方和 / 样本总数
            3. 均方根误差(RMSE)  每个(样本)误差平方和 / 样本总数 的 平方根
            4. 平均绝对误差(MAE) 每个(样本)误差绝对值和 / 样本总数
    如何让损失函数最小?
        思路1: 梯度下降法.     =>   全梯度下降(Full Gradient Descent, FGD), 随机梯度下降(SGD), 小批量梯度下降(推荐, Min-Batch), 随机平均梯度下降(SAG)
        思路2: 正规方程法.

    机器学习开发流程:
        1. 加载数据.
        2. 数据的预处理.
        3. 特征工程(特征提取, 特征预处理...)
        4. 模型训练.
        5. 模型预测.
        6. 模型评估
"""

# 导包
# from sklearn.datasets import load_boston  # 数据
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
)  # 均方误差评估, RMSE, MAE


data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\\s+", skiprows=22, header=None)
data = np.hstack(
    [raw_df.values[::2, :], raw_df.values[1::2, :2]]
)  # hstack()函数作用: 水平拼接数组
target = raw_df.values[1::2, 2]

# print(f"特征: {data.shape}")  # (506, 13)
# print(f"标签: {target.shape}")  # (506,)

# print(f"特征数据: {data[:5]}")
# print(f"标签数据: {target[:5]}")

# 2. 数据的预处理.  切分训练集 和 测试集.
# 参1: 特征数据. 参2: 标签数据. 参3: 测试集占训练集的比例. 参4: 随机种子.
x_train, x_test, y_train, y_test = train_test_split(
    data, target, test_size=0.2, random_state=23
)

# 3. 特征工程(特征提取, 特征预处理...)
# 3.1 创建标准化对象.
transfer = StandardScaler()
# 3.2 对训练集进行标准化.
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

# 4. 模型训练.
# 4.1 创建 线性回归 正规方程 模型对象.
estimator = LinearRegression(
    fit_intercept=True
)  # fit_intercept: 是否需要截距(Bias, 偏置), 默认是True
# 4.2 模型训练.
estimator.fit(x_train, y_train)
# 4.3 打印模型计算出的 w(权重, weight) 和 b(偏置, bias).
print(f"权重: {estimator.coef_}")
print(f"偏置: {estimator.intercept_}")
# 5. 模型预测.
y_pre = estimator.predict(x_test)
print(f"预测结果为: {y_pre}")

# 6. 模型评估
# 参1: 测试集的标签数据. 参2: 预测结果.
print(
    f"均方误差为: {mean_squared_error(y_test, y_pre)}"
)  # MSE: 均方误差, 公式: 每个样本的误差平方和 / 样本总数
print(
    f"均方根误差为: {root_mean_squared_error(y_test, y_pre)}"
)  # RMSE: 均方根误差, 公式: 每个样本的误差平方和 / 样本总数, 开平方根
print(
    f"平均绝对误差为: {mean_absolute_error(y_test, y_pre)}"
)  # MAE: 平均绝对误差, 公式: 每个样本的误差 绝对值和 / 样本总数
