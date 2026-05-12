"""
案例:
    通过 XGBoost 极限梯度提升树 完成 红酒品质分类案例.


回顾: XGBoost 极限梯度提升树
    概述:
        Extreme Gradient Boosting Tree, 底层采用 打分函数 决定是否分支.
    原理:
        Gain值 = 分枝前的打分 - (分支后左子树打分 + 分支后右子树打分)
        如果 Gain值 > 0, 考虑分枝, 否则: 不考虑分枝.
"""

import joblib  # 保存和加载模型
import numpy as np
import pandas as pd
import xgboost as xgb  # 极限梯度提升树对象
from collections import Counter  # 统计数据
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
)  # 训练集和测试集的划分
from sklearn.metrics import classification_report, accuracy_score  # 模型(分类)评估报告
from sklearn.model_selection import (
    StratifiedKFold,
)  # 分层K折交叉验证, 类似于 网格搜索时 cv=折数
from sklearn.utils import class_weight  # 计算样本权重


# 1. 定义函数, 对 红酒品质分类源数据 -> 拆分成 训练集和测试集, 并存储到csv文件中.
def dm01_data_split():
    # 1. 加载数据集.
    df = pd.read_csv("./data/红酒品质分类.csv")
    # 2. 查看数据集.
    df.info()
    # 3. 抽取特征数据 和 标签数据.
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1] - 3  # 最后1列是标签, 默认范围是: [3, 8]  -> [0, 5]
    # 4. 查看数据.
    # print(x[:5])
    # print(y[:5])
    print(f"查看 标签结果的分布情况, 是否均衡: {Counter(y)}")
    # 5. 切分 训练集和测试集.
    # 参1: 特征数据. 参2: 标签数据. 参3: 测试集的比例. 参4: 随机种子. 参5: 参考数据集的标签分布.
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6. 把上述的 训练集特征 和 标签数据拼接到一起,  测试集特征 和 标签数据拼接到一起.  最后写到文件中.
    pd.concat([x_train, y_train], axis=1).to_csv(
        "./data/红酒品质分类_train.csv", index=False
    )
    pd.concat([x_test, y_test], axis=1).to_csv(
        "./data/红酒品质分类_test.csv", index=False
    )


# 2. 定义函数, 训练模型, 并保存模型.
def dm02_train_model():
    # 1. 读取训练集和测试集.
    train_data = pd.read_csv("./data/红酒品质分类_train.csv")
    test_data = pd.read_csv("./data/红酒品质分类_test.csv")

    # 2. 提取 训练集和测试集的 特征数据 和 标签数据.
    x_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]
    x_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]

    # 3. 创建模型对象.
    estimator = xgb.XGBClassifier(
        n_estimators=100,  # 弱学习器的数量
        learning_rate=0.1,  # 学习率
        max_depth=5,  # 树最大深度
        objective="multi:softmax",  # 目标函数
        random_state=42,
    )

    # 加入 平衡权重, 因为数据集是 样本不均衡的.
    # 参1: 平衡权重, 参2: 标签数据(即: 参考标签数据分布, 平衡权重)
    class_weight.compute_sample_weight(class_weight="balanced", y=y_train)

    # 4. 模型训练.
    estimator.fit(x_train, y_train)

    # 5. 模型评估.
    print(f"准确率: {estimator.score(x_test, y_test)}")

    # 6. 保存模型.
    joblib.dump(
        estimator, "./model/红酒品质分类_model.pkl"
    )  # 后缀名也可以写 .pth, 都是pickle文件格式
    print(f"模型保存成功: {'./model/红酒品质分类_model.pkl'}")


# 3. 定义函数, 测试模型.
def dm03_use_model():
    # 1. 读取训练集和测试集.
    train_data = pd.read_csv("./data/红酒品质分类_train.csv")
    test_data = pd.read_csv("./data/红酒品质分类_test.csv")

    # 2. 提取 训练集和测试集的 特征数据 和 标签数据.
    x_train = train_data.iloc[:, :-1]  # 除了最后1列, 都是特征
    y_train = train_data.iloc[:, -1]  # 最后1列是标签

    x_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]

    # 3. 加载模型.
    estimator = joblib.load("./model/红酒品质分类_model.pkl")

    # 4. 创建网格搜索 + 交叉验证(结合分层采样数据), 找模型最优参数组合.
    # 4.1 定义变量, 记录: 参数组合.
    param_dict = {
        "max_depth": [5, 6, 7],
        "n_estimators": [80, 100, 120],
        "learning_rate": [0.08, 0.1, 0.12],
    }
    # 4.2 创建 分层采样 对象.
    # 参1: 折数, 参2: 是否打乱(数据), 参3: 随机种子.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 4.3 创建 网格搜索 + 交叉验证(结合分层采样数据) 对象.
    # 参1: 模型对象, 参2: 参数组合, 参3: 交叉验证对象.
    gs_estimator = GridSearchCV(estimator, param_grid=param_dict, cv=skf)

    # 5. 模型训练.
    gs_estimator.fit(x_train, y_train)

    # 6. 模型预测.
    y_pre = gs_estimator.predict(x_test)

    # 7. 打印模型评估系数.
    print(f"最优估计器对象组合: {gs_estimator.best_estimator_}")
    print(f"最优评分: {gs_estimator.best_score_}")
    print(f"准确率: {accuracy_score(y_test, y_pre)}")


if __name__ == "__main__":
    # dm01_data_split()
    # dm02_train_model()
    dm03_use_model()
