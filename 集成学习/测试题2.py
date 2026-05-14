import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei']

data = {
    "收入水平": [5, 8, 3, 10, 4, 9, 2, 6, 7, 4],
    "锻炼频率": [3, 6, 2, 7, 4, 5, 1, 4, 6, 3]
}

df = pd.DataFrame(data)
x = df.values

# print(x)
sse = []
for k in range(1, 10):
    estimator = KMeans(n_clusters=k, max_iter=100, random_state=42)
    estimator.fit(x)
    sse.append(estimator.inertia_)


# 最大曲率法计算最优k值
def find_best_k(sse):
    sse = np.array(sse)
    d1 = np.diff(sse)
    d2 = np.diff(d1)
    curvature = np.abs(d2)
    best_idx = np.argmax(curvature)
    return 2 + best_idx


best_k = find_best_k(sse)
print(f"最佳k值为: {best_k}")

estimator = KMeans(n_clusters=best_k, max_iter=100, random_state=42)
y_pred = estimator.fit_predict(x)
plt.figure(figsize=(10, 6))
plt.scatter(x[:, 0], x[:, 1], c=y_pred)
plt.xlabel("收入水平(万元)")
plt.ylabel("锻炼频率(次/周)")
plt.title("用户分群散点图（收入 vs 锻炼频率）")
plt.grid(True)
plt.show()
