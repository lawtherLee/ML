"""
案例:
    演示逻辑回归模型实现 癌症预测.

逻辑回归模型介绍:
    概述:
        属于有监督学习, 即: 有特征, 有标签, 且表示是离散的.
        主要适用于: 二分类.
    原理:
        把线性回归处理后的预测值 -> 通过 Sigmoid激活函数, 映射到[0, 1] 概率 -> 基于自定义的阈值, 结合概率来 分类.
    损失函数:
        极大似然估计函数的 负数形式.

回顾: 机器学习项目流程
    1. 加载数据.
    2. 数据预处理.
    3. 特征工程(提取, 预处理...)
    4. 模型训练.
    5. 模型预测.
    6. 模型评估.
"""

# 导包
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression  # 逻辑回归模型
from sklearn.preprocessing import StandardScaler  # 标准化
from sklearn.model_selection import train_test_split  # 训练集和测试集分割
from sklearn.metrics import accuracy_score  # 模型评估

# 1. 加载数据.
data = pd.read_csv("./data/breast-cancer-wisconsin.csv")
# data.info()

# 2. 数据预处理.
# 2.1 把 ? 替换成 np.nan,  参1: 要被替换的值,  参2: 用来替换的值,  参3: 是否替换源数据, 默认为False
data.replace("?", np.nan, inplace=True)
# 2.2 缺失值处理 -> 删除.
data.dropna(inplace=True)
# data.info()

# 3. 特征工程(提取, 预处理...)
# 3.1 特征提取之 提取特征和标签.
x = data.iloc[
    :, 1:-1
]  # 按照行号, 列索引获取数据,  :表示所有行   1:-1 表示从第1列到最后1列, 包左不包右
y = data.iloc[:, -1]  # 获取最后一列
# y = data['Class']       # 获取最后一列, 效果同上

# 3.2 查看下特征 和 标签.
print(x[:5])
print(y[:5])
print(x.shape, y.shape)

# 3.3 切割训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
# 3.4 特征工程: 标准化
# 3.4.1 创建标准化对象.
transfer = StandardScaler()
# 3.4.2 对训练集进行标准化.  训练 + 标准化
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

# 4. 模型训练.
# 4.1 创建模型对象 -> 逻辑回归模型
estimator = LogisticRegression()
estimator.fit(x_train, y_train)

# 5. 模型预测.
y_pre = estimator.predict(x_test)
print(f"预测值为: {y_pre}")

# 6. 模型评估.
# 正确率(准确率), 公式为: 预测对的 / 样本总数
print(f"预测前评估, 正确率: {estimator.score(x_test, y_test)}")  # 测试集的特征, 标签.
print(f"预测后评估, 正确率: {accuracy_score(y_test, y_pre)}")  # 测试集的标签, 预测值.

# 思考: 逻辑回归模型能用 准确率来评测吗?
# 答案: 可以, 但是结果不精准, 因为逻辑回归模型主要用于 二分类, 即: A类还是B类, 不能说 97%的A类, 3%的B类.
# 所以要通过 混淆矩阵来评测, 即: 精确率, 召回率, F1值(F1-Score), ROC曲
