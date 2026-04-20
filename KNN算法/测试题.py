"""
KNN回归思路
"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

data = pd.DataFrame(
    {
        "temperature": [25, 27, 30, 22, 28, 29, 26, 24, 31, 21],
        "humidity": [80, 65, 90, 40, 70, 85, 75, 60, 95, 50],
        "wind_speed": [10, 15, 5, 12, 8, 11, 9, 6, 14, 4],
        "rain_probability": [10, 20, 0, 5, 30, 15, 50, 25, 5, 10],
        "y": [28, 26, 32, 20, 30, 31, 27, 23, 33, 19],
    }
)

# 划分特征和目标向量
x = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 分割数据集
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# 数据标准化
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)


# 训练&预测
estimator = KNeighborsRegressor()
# estimator.fit(x_train, y_train)

# 交叉验证&网格搜索
param_grid = {"n_neighbors": [i for i in range(1, 5)]}


estimator = GridSearchCV(estimator, param_grid, cv=3)  # 3折交叉验证
estimator.fit(x_train, y_train)
print(f"最优评分: {estimator.best_score_}")
print(f"最优超参组合: {estimator.best_params_}")
print(f"最优的估计器对象: {estimator.best_estimator_}")

estimator = estimator.best_estimator_
estimator.fit(x_train, y_train)

y_pre = estimator.predict(x_test)
print(f"预测值为: {y_pre}")

print(f"R² 决定系数: {estimator.score(x_test, y_test):.2f}")  # 回归得分
print(f"均方误差 MSE: {mean_squared_error(y_test, y_pre):.2f}")
