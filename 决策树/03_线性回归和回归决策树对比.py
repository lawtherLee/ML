"""
案例:
    演示 线性回归 和 回归决策树(CART)对比.

细节:
    CART分类回归决策树, 既可以做分类, 也可以做回归, 一般做: 分类.
    做分类是采用 基尼值, 做回归时采用 平方损失(类似于 最小二乘)
"""

# 导包
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor      # 回归决策树
from sklearn.linear_model import LinearRegression   # 线性回归


# 1. 准备数据.
# 训练集的特征数据
x_train = np.array(list(range(1, 11))).reshape(-1, 1)
# 训练集的标签数据
y_train = np.array([5.56, 5.7, 5.91, 6.4, 6.8, 7.05, 8.9, 8.7, 9, 9.05])
# print(x_train)
# print(y_train)

# 2. 数据预处理, 该案例不需要.

# 3. 特征工程, 该案例不需要.

# 4. 模型训练.
# 4.1 分别创建 线性回归 和 回归决策树模型对象.
estimator1 = LinearRegression()                     # 线性回归
estimator2 = DecisionTreeRegressor(max_depth=1)     # 回归决策树, 最大树深度为1
estimator3 = DecisionTreeRegressor(max_depth=3)     # 回归决策树, 最大树深度为3

# 4.2 模型训练.
estimator1.fit(x_train, y_train)
estimator2.fit(x_train, y_train)
estimator3.fit(x_train, y_train)

# 5. 模型预测.
# 5.1 准备测试集的 特征数据
# x_test = np.array(list(range(0.0, 10.0, 0.1))).reshape(-1, 1)   # 报错, Python自带的range()函数, 不支持小数.
x_test = np.arange(0, 10, 0.1).reshape(-1, 1)
# print(x_test)

# # 5.2 具体的预测动作.
y_pred1 = estimator1.predict(x_test)
y_pred2 = estimator2.predict(x_test)
y_pred3 = estimator3.predict(x_test)

# 5.3 打印预测结果.
# print(f'预测结果1(线性回归): {y_pred1}')
# print(f'预测结果2(回归决策树, 最大树深度为1): {y_pred2}')
# print(f'预测结果3(回归决策树, 最大树深度为3): {y_pred3}')

# 6. 模型评估, 此处略.

# 7. 绘图.
# 7.1 以真实值(训练集) 绘制 散点图.
plt.scatter(x_train, y_train, c='gray')
# 7.2 以预测值(线性回归, 回归决策树) 绘制 折线图.
plt.plot(x_test, y_pred1, c='red', label='linear regression')
plt.plot(x_test, y_pred2, c='blue', label='max depth=1')
plt.plot(x_test, y_pred3, c='green', label='max depth=3')
# 7.3 显示图例.
plt.legend()
# 7.4 设置x轴, y轴, 标题.
plt.xlabel('data')
plt.ylabel('target')
plt.title('Decision Tree Regression')
plt.show()
