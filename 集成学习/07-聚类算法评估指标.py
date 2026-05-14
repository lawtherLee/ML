"""
案例:
    演示聚类算法的评估指标, SSE+肘部法, SC轮廓系数法, CH轮廓系数法

    聚类算法的评估指标:
        思路1: SSE + 肘部法
             SSE:
                概述:
                     所有簇的 所有样本到该簇质心的 误差的平方和
                特点:
                     随着K值的增加, SSE值会逐渐减少
                目标:
                     SSE值越小, 代表簇内样本越聚集, 内聚程度越高
             肘部法:
                     K值越大, SSE值会随之减小, 下降梯度陡然变缓的时候, 那个K值, 就是最佳值
        思路2: SC轮廓系数
            考虑簇内 -> 聚集程度, 越小越好
            考虑簇外 -> 分离程度, 越大越好
        思路3: CH轮廓系数
            考虑簇内 -> 聚集程度, 越小越好
            考虑簇外 -> 分离程度, 越大越好
            考虑k值 -> k值越小, 代表簇内样本越聚集, 内聚程度越高.

"""

from sklearn.cluster import KMeans  # 算法
from sklearn.metrics import calinski_harabasz_score  # ch系数 评估指标
from sklearn.metrics import silhouette_score  # sc系数
import matplotlib.pyplot as plt  # 绘图
from sklearn.datasets import make_blobs  # 生成数据集
import warnings

warnings.filterwarnings("ignore")


# 1. 演示: SSE + 肘部法
def dm01_sse():
    # 1. 定义sse列表, 记录每个k值的SSE值.
    sse_list = []
    # 2. 生成数据集 参1: 样本数量, 参2: 样本特征数量, 参3: 样本标签数量, 参4: 标准差, 参5: 随机种子.
    x, y = make_blobs(
        n_samples=1000,
        n_features=2,
        centers=[[-1, -1], [0, 0], [1, 1], [2, 2]],
        cluster_std=[0.4, 0.2, 0.3, 0.4],
        random_state=23,
    )
    # 3. 遍历K值. 计算对应的SSE值.
    for k in range(1, 100):
        # 3.1 创建KMeans对象. 指定K值, 迭代次数, 随机种子
        estimator = KMeans(n_clusters=k, max_iter=100, random_state=23)
        # 3.2 训练模型
        estimator.fit(x)
        # 3.3 模型预测
        # 3.4 获取每个簇的SSE值
        sse_value = estimator.inertia_
        # 3.5 将每个k值的SSE值, 添加到列表中.
        sse_list.append(sse_value)

    # 4. 绘制SSE曲线.
    # 4.1 创建画布, 指定宽高
    plt.figure(figsize=(20, 10))
    # 4.2 设置标题.
    plt.title("SSE曲线")
    # 4.3 设置x轴刻度
    plt.xticks(range(0, 100, 3))
    # 4.4 设置xy轴标签 网格
    plt.xlabel("K值")
    plt.ylabel("SSE值")
    plt.grid()
    # 参1: k值, 参2: 对应的SSE值
    plt.plot(range(1, 100), sse_list, marker="o")
    plt.show()


# 2. 演示: SC轮廓系数法
def dm02_sc():
    # 1. 定义sc列表, 记录每个k值的sc值.
    sc_list = []
    # 2. 生成数据集 参1: 样本数量, 参2: 样本特征数量, 参3: 样本标签数量, 参4: 标准差, 参5: 随机种子.
    x, y = make_blobs(
        n_samples=1000,
        n_features=2,
        centers=[[-1, -1], [0, 0], [1, 1], [2, 2]],
        cluster_std=[0.4, 0.2, 0.3, 0.4],
        random_state=23,
    )
    # 3. 遍历K值. 计算对应的sc值.
    for k in range(2, 100):  # 考虑簇外 至少两个簇
        # 3.1 创建KMeans对象. 指定K值, 迭代次数, 随机种子
        estimator = KMeans(n_clusters=k, max_iter=100, random_state=23)
        # 3.2 训练模型
        estimator.fit(x)
        # 3.3 模型预测
        y_pred = estimator.predict(x)
        # 3.4 获取每个簇的sc值
        sc_value = silhouette_score(x, y_pred)
        # 3.5 将每个k值的sc值, 添加到列表中.
        sc_list.append(sc_value)

    # 4. 绘制sc曲线.
    # 4.1 创建画布, 指定宽高
    plt.figure(figsize=(20, 10))
    # 4.2 设置标题.
    plt.title("sc曲线")
    # 4.3 设置x轴刻度
    plt.xticks(range(0, 100, 3))
    # 4.4 设置xy轴标签 网格
    plt.xlabel("K值")
    plt.ylabel("sc值")
    plt.grid()
    # 参1: k值, 参2: 对应的sc值
    plt.plot(range(2, 100), sc_list, marker="o")
    plt.show()


# 3. 演示: CH轮廓系数法
def dm03_ch():
    # 1. 定义ch列表, 记录每个k值的ch值.
    ch_list = []
    # 2. 生成数据集 参1: 样本数量, 参2: 样本特征数量, 参3: 样本标签数量, 参4: 标准差, 参5: 随机种子.
    x, y = make_blobs(
        n_samples=1000,
        n_features=2,
        centers=[[-1, -1], [0, 0], [1, 1], [2, 2]],
        cluster_std=[0.4, 0.2, 0.3, 0.4],
        random_state=23,
    )
    # 3. 遍历K值. 计算对应的ch值.
    for k in range(2, 100):  # 考虑簇外 至少两个簇
        # 3.1 创建KMeans对象. 指定K值, 迭代次数, 随机种子
        estimator = KMeans(n_clusters=k, max_iter=100, random_state=23)
        # 3.2 训练模型
        estimator.fit(x)
        # 3.3 模型预测
        y_pred = estimator.predict(x)
        # 3.4 获取每个簇的ch值
        ch_value = calinski_harabasz_score(x, y_pred)
        # 3.5 将每个k值的ch值, 添加到列表中.
        ch_list.append(ch_value)

    # 4. 绘制ch曲线.
    # 4.1 创建画布, 指定宽高
    plt.figure(figsize=(20, 10))
    # 4.2 设置标题.
    plt.title("ch曲线")
    # 4.3 设置x轴刻度
    plt.xticks(range(0, 100, 3))
    # 4.4 设置xy轴标签 网格
    plt.xlabel("K值")
    plt.ylabel("ch值")
    plt.grid()
    # 参1: k值, 参2: 对应的ch值
    plt.plot(range(2, 100), ch_list, marker="o")
    plt.show()


# 测试
if __name__ == "__main__":
    # dm01_sse()
    # dm02_sc()
    dm03_ch()
