"""
案例:
    演示 集成学习之 Bagging思想 随机森林算法 代码.

集成学习:
    概述:
        把多个弱学习器 组成 1个强学习器的过程 -> 集成学习.
    思想:
        Bagging思想:
            1. 有放回的随机抽样.
            2. 平权投票.
            3. 可以并行执行.

        Boosting思想:
            1. 每次训练都会使用全部样本.
            2. 加权投票 -> 预测正确:权重降低, 预测错误: 权重增加.
            3. 只能串行执行.
    Bagging思想代表:
        随机森林算法.

随机森林算法:
    1. 每个弱学习器都是 CART树(必须是二叉树)
    2. 有放回的随机抽样, 平权投票, 并行执行.
"""

# 导包
import pandas as pd
from sklearn.model_selection import train_test_split  # 切分训练集和测试集
from sklearn.tree import DecisionTreeClassifier  # 决策树
from sklearn.ensemble import RandomForestClassifier  # 随机森林算法(分类器)
from sklearn.model_selection import GridSearchCV  # 网格搜索

# 1. 加载数据.
df = pd.read_csv("../决策树/data/train.csv")
# df.info()

# 2. 数据的预处理.
# 2.1 抽取 特征 和 标签.
x = df[["Pclass", "Sex", "Age"]].copy()  # 船舱等级, 性别, 年龄
y = df["Survived"]

# 2.2 空值处理, 用Age列的平均值 填充Age列的空值
x["Age"] = x["Age"].fillna(x["Age"].mean())

# 2.3 热编码处理.
x = pd.get_dummies(x)

# 2.4 划分训练集 和 测试集.
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=23
)


# 3. 特征工程. 此处略.

# 4. 模型训练, 预测, 评估.
# 场景1: 单一决策树.
# 4.1 创建 决策树对象, 演示: 单一的决策树效果.
estimator1 = DecisionTreeClassifier()
# 4.2 模型训练.
estimator1.fit(x_train, y_train)
# 4.3 预测.
# y_pred = estimator1.predict(x_test)
# print(f"预测值为: {y_pred}")
# 4.4 评估.
print(
    f"决策树模型的准确率为: {estimator1.score(x_test, y_test)}"
)  #  0.8044692737430168
print("-" * 23)

# 场景2: 随机森林算法 -> 采用默认参数.
# 4.1 创建 随机森林对象, 演示: 多个的决策树(Bagging思想)效果.
estimator2 = RandomForestClassifier()  # n_estimators=100, max_depth=None
# 4.2 模型训练.
estimator2.fit(x_train, y_train)
# 4.3 预测.
# y_pred2 = estimator2.predict(x_test)
# print(f"预测值为: {y_pred2}")
# 4.4 评估
print(f"随机森林模型的准确率为: {estimator2.score(x_test, y_test)}")
print("-" * 23)


# 场景3: 随机森林算法 -> 采用网格搜索.
# 4.1 创建 随机森林对象, 演示: 多个的决策树(Bagging思想)效果.
estimator3 = RandomForestClassifier(n_estimators=50, max_depth=5)
estimator3.fit(x_train, y_train)
# 4.2 参数准备.
params = {"n_estimators": [30, 50, 60, 90, 110], "max_depth": [2, 3, 5, 7]}
# 4.3 创建网格搜索对象 结合 交叉验证.
gs_estimator = GridSearchCV(estimator3, param_grid=params, cv=2)
# 4.4 模型训练.
gs_estimator.fit(x_train, y_train)
# 4.5 预测.
# y_pred3 = gs_estimator.predict(x_test)
# print(f"预测值为: {y_pred3}")
# 4.6 评估
print(f"随机森林模型的准确率为: {gs_estimator.score(x_test, y_test)}")
# 4.7 获取 最佳参数.
print(f"最佳参数为: {gs_estimator.best_params_}")
