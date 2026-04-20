import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ----------------------
# 1. 准备数据（你的天气数据）
# ----------------------
data = pd.DataFrame(
    {
        "temperature": [25, 27, 30, 22, 28, 29, 26, 24, 31, 21],
        "humidity": [80, 65, 90, 40, 70, 85, 75, 60, 95, 50],
        "wind_speed": [10, 15, 5, 12, 8, 11, 9, 6, 14, 4],
        "rain_probability": [10, 20, 0, 5, 30, 15, 50, 25, 5, 10],
        "y": [28, 26, 32, 20, 30, 31, 27, 23, 33, 19],
    }
)

# 划分特征和目标变量（自动适配任意列数）
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# ----------------------
# 2. 先分割训练集/测试集（防止数据泄露）
# ----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------
# 3. 标准化（只在训练集上fit）
# ----------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------
# 4. 网格搜索 + 交叉验证（解决小数据报错）
# ----------------------
# 定义要搜索的超参数网格
param_grid = {
    "n_neighbors": [1, 2, 3, 4, 5],  # K值范围
    "weights": ["uniform", "distance"],  # 权重方式：均匀/距离加权
    "metric": ["euclidean", "manhattan"],  # 距离度量方式
}

# 关键修复：用留一法交叉验证（适配小数据，不会报错）
cv_strategy = LeaveOneOut()  # 每次用n-1个样本训练，1个验证，适合小数据

# 初始化网格搜索
grid = GridSearchCV(
    estimator=KNeighborsRegressor(),
    param_grid=param_grid,
    cv=cv_strategy,  # 用留一法交叉验证
    scoring="neg_mean_squared_error",  # 回归任务用负MSE作为评分
    refit=True,  # 搜索结束后用最佳参数重新训练整个训练集
    error_score=np.nan,  # 遇到无法拟合的情况返回NaN，避免报错中断
)

# 在训练集上执行网格搜索
grid.fit(X_train_scaled, y_train)

# ----------------------
# 5. 输出最佳超参数和结果
# ----------------------
print("=" * 50)
print("✅ 网格搜索结果：")
print(f"最佳超参数组合: {grid.best_params_}")
print(f"最佳交叉验证负MSE: {grid.best_score_:.2f}")
print(f"最佳交叉验证MSE: {-grid.best_score_:.2f}")
print("=" * 50)

# ----------------------
# 6. 用最佳模型在测试集上评估
# ----------------------
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test_scaled)

print("\n📊 测试集评估结果：")
print(f"真实值: {y_test.values}")
print(f"预测值: {y_pred.round(1)}")
print(f"R² 决定系数: {r2_score(y_test, y_pred):.2f}")
print(f"均方误差 MSE: {mean_squared_error(y_test, y_pred):.2f}")
