"""
案例:
    演示AdaBoost算法 之  葡萄酒案例.

AdaBoost算法介绍:
    它属于 Boosting思想, 即: 串行执行, 每次使用全部样本, 最后 加权投票.
    原理:
        1. 使用全部样本, 通过决策树模型(第1个弱分类器)进行训练, 获取结果.
            思路:
                预测正确 -> 权重下降
                预测错误 -> 权重上升
        2. 把第1个弱分类器的处理结果, 交给第2个弱分类器进行训练, 获取结果.
            思路:
                预测正确 -> 权重下降
                预测错误 -> 权重上升
        3. 依次类推, 串行执行, 直至获取最终结果.
"""

# 导包
import pandas as pd
from sklearn.preprocessing import LabelEncoder  # 标签编码器
from sklearn.model_selection import train_test_split  # 训练集、测试集分割
from sklearn.tree import DecisionTreeClassifier  # 决策树分类器
from sklearn.ensemble import (
    AdaBoostClassifier,
)  # AdaBoost分类器 -> 集成学习Boosting思想
from sklearn.metrics import accuracy_score  # 模型评估 -> 正确率

# 1. 获取数据集
df_wine = pd.read_csv("./data/wine0501.csv")
# df_wine.info()

# print(df_wine['Class label'].unique())    # [1, 2, 3]   葡萄酒类别有3种. 但是决策树只能识别 二叉树.

# 2. 数据预处理
# 2.1 从 标签列(Class label)中, 过滤掉 1类别, 剩下 2, 3 类别.
df_wine = df_wine[df_wine["Class label"] != 1]
# print(df_wine['Class label'].unique())      # [2, 3]

# 2.2 获取特征列 和 标签列.
x = df_wine[["Alcohol", "Hue"]]  # 酒精 和 色泽
y = df_wine["Class label"]  # 标签列.

# 2.3 打印数据.
# print(x[:5])
# print(y[:5])

# 2.4 通过 标签编码器, 把 标签列, 转换为 数值列. 这玩意能去重
le = LabelEncoder()
y = le.fit_transform(y)
# print(y)      # [2, 3] -> [0, 1]

# 2.5 训练集、测试集分割.
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=23, stratify=y
)

# 3. 特征工程, 此处略.

# 4. 模型训练, 预测, 评估.
# 场景1: 单一决策树 -> 充当弱分类器
# 4.1 创建模型对象
estimator1 = DecisionTreeClassifier(max_depth=3)
# 4.2 训练模型
estimator1.fit(x_train, y_train)
# # 4.3 模型预测
y_pre1 = estimator1.predict(x_test)
# print(f'单一决策树预测结果: {y_pre1}')
# 4.4 模型评估
print(f"单一决策树预测正确率: {accuracy_score(y_test, y_pre1)}")  # 0.91666666


# 场景2: AdaBoost -> 集成学习, CART树, 200棵
# 4.1 创建模型对象
# 参1: 弱分类器(决策树对象), 参2: 弱分类器个数, 参3: 学习率, 参4: 集成算法
estimator2 = AdaBoostClassifier(
    estimator=estimator1, n_estimators=200, learning_rate=0.1
)
# 4.2 训练模型
estimator2.fit(x_train, y_train)
# 4.3 模型预测
y_pre2 = estimator2.predict(x_test)
# print(f'AdaBoost集成学习预测结果: {y_pre2}')
# 4.4 模型评估
print(f"AdaBoost集成学习预测正确率: {accuracy_score(y_test, y_pre2)}")
