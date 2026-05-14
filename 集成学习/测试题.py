# 由于样本数量较少 使用SSE算法评估k值
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei']

data = {
    "客户ID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "平均每月消费金额": [800, 1200, 200, 700, 300, 1000, 400, 800, 1100, 350],
    "平均购买频率": [3, 5, 1, 3, 2, 4, 2, 4, 5, 1],
    "最近一次购买距今时间": [10, 7, 40, 12, 25, 5, 30, 8, 6, 20]
}

df = pd.DataFrame(data)

x = df.iloc[:, 1:]
print(x)

# 特征工程
scaler = StandardScaler()
x = scaler.fit_transform(x)
x_original = df.iloc[:, 1:].values

sse_list = []
for k in range(1, 10):
    estimator = KMeans(n_clusters=k, max_iter=100, random_state=42)
    estimator.fit(x)
    sse_value = estimator.inertia_
    sse_list.append(sse_value)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 10), sse_list, marker='o')
plt.show()

# 最佳k=2

estimator = KMeans(n_clusters=2, max_iter=100, random_state=42)
y_pred = estimator.fit_predict(x)

# print(y_pred)
plt.figure(figsize=(10, 6))
plt.scatter(x_original[:, 0], x_original[:, 1], c=y_pred)
plt.xlabel("平均每月消费金额（元）")
plt.ylabel("平均购买频率（次/月）")
plt.title("客户分群散点图（消费金额 vs 购买频率）")
plt.grid(True)
plt.show()

"""
1. 黄色点（簇 0）
消费金额：200-400 元
购买频率：1-2 次 / 月
对应客户：3、5、7、10
特征：低消费、低频率，属于价格敏感型低价值客户
2. 紫色点（簇 1）
消费金额：500-1200 元
购买频率：3-5 次 / 月
对应客户：1、2、4、6、8、9
特征：中高消费、中高频率，属于平台的核心活跃客户
"""
