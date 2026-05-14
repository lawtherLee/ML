"""
案例:
    基于用户的 年收入 和 消费指数, 根据用户的 相似性 进行聚类
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score
import matplotlib.pyplot as plt


# 1. 找聚类的质心数(k值)
def find_k():
    # 1. 加载数据集
    df = pd.read_csv("./data/customers.csv")
    df.info()
    print(df.head())
    # 2. 定义sse_list, sc_list, 记录: 不同k值的 评估效果
    sse_list = []
    sc_list = []
    # 抽取特征
    x = df.iloc[:, 3:]
    # print(x)
    # 3. 循环训练, 测试不同的k值 评估效果
    for k in range(2, 20):
        # 4. 创建KMeans对象
        estimator = KMeans(n_clusters=k, max_iter=100, random_state=42)
        # 5. 模型训练 + 预测
        y_pred = estimator.fit_predict(x)
        # 6. 分别把评分添加到对应的列表中
        sse_list.append(estimator.inertia_)
        sc_list.append(silhouette_score(x, y_pred))

    # 4. 绘制折线图 看看K值那个最好
    plt.figure(figsize=(20, 10))
    plt.plot(range(2, 20), sse_list, label='SSE')
    plt.show()

    plt.figure(figsize=(20, 10))
    plt.plot(range(2, 20), sc_list, label='SC')
    plt.show()

    # 结论 k = 5 效果最好


# 2. 模型训练 预测 评估
def train_predict():
    df = pd.read_csv("./data/customers.csv")
    # 抽取特征
    x = df.iloc[:, 3:]
    # print(x.values)
    estimator = KMeans(n_clusters=5, max_iter=100, random_state=42)
    y_pred = estimator.fit_predict(x)
    # 绘制5个簇的样本点散点图
    plt.scatter(x.iloc[:, 0], x.iloc[:, 1], c=y_pred)
    # 绘制5个簇的质心散点图
    plt.scatter(estimator.cluster_centers_[:, 0], estimator.cluster_centers_[:, 1], c='r')
    plt.show()


if __name__ == "__main__":
    # find_k()
    train_predict()
