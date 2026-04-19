"""
KNN算法介绍(K Nearest Neighbors), K近邻算法
    原理:
        基于 欧式距离(或者其它距离计算方式)计算 测试集 和 每个训练集之间的距离, 然后根据距离升序排列, 找到最近的K个样本.
        基于K个样本投票, 票数多的就作为最终预测结果 -> 分类问题.
        基于K个样本计算平均值, 作为最终预测结果 -> 回归问题.
    实现思路:
        1. 分类问题
            适用于: 有特征, 有标签, 且标签是不连续的(离散的)
        2. 回归问题.
            适用于: 有特征, 有标签, 且标签是连续的.
    KNN算法, 分类问题思路如下:
        1. 计算测试集和每个训练的样本之间的 距离.
        2. 基于距离进行升序排列.
        3. 找到最近的K个样本.
        4. K个样本进行投票.
        5. 票数多的结果, 作为最终的预测结果.
    代码实现思路:
        1. 导包.
        2. 准备数据集(测试集 和 训练集)
        3. 创建(KNN 分类模型)模型对象.
        4. 模型训练.
        5. 模型预测.
"""

# 1. 导包.
from sklearn.neighbors import KNeighborsClassifier  # 分类
# from sklearn.neighbors import KNeighborsRegressor   # 回归

# 2. 准备数据集(测试集 和 训练集)
# train: 训练集
# test: 测试集
# neighbors: 最近邻的邻居数
x_train = [[0], [1], [2], [3]]      # 训练集的特征数据, 因为特征可以有多个特征, 所以是一个二维数组
y_train = [0, 0, 1, 1]              # 训练集的标签数据, 因为标签是离散的, 所以是一个一维数组
x_test = [[5]]                      # 测试集的特征数据

# 3. 创建(KNN 分类模型)模型对象.
# estimator: 估计器, 模型对象, 也可以用变量名 model做接收.
estimator = KNeighborsClassifier(n_neighbors=3)

# 4. 模型训练
# 传入: 训练集的特征数据, 训练集的标签数据
estimator.fit(x_train, y_train)

# 5. 模型预测.
# 传入: 测试集的特征数据, 获取到: 预测结果(测试集的标签, y_test)
y_pre = estimator.predict(x_test)

# 6. 打印预测结果.
print(f'预测值为: {y_pre}')
