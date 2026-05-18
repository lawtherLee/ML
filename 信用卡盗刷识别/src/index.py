import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# 1. 加载数据
data = pd.read_csv("../行用卡数据集/creditcard.csv")
# print(data["Label"].value_counts())
data = data.dropna(subset=["Label"])
# 2. 特征/标签分离
x = data.drop(["ID", "Label"], axis=1)
y = data["Label"]

# 3. 划分训练集/测试集（stratify保证比例一致）
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

# 4. 标准化（只fit训练集，防止数据泄露）
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

# 5. SMOTE过采样（只对训练集做，测试集保持真实分布）
smt = SMOTE(random_state=42)
x_train, y_train = smt.fit_resample(x_train, y_train)
print("过采样后训练集分布：", pd.Series(y_train).value_counts())

# 6. 训练模型
estimator = LogisticRegression(max_iter=1000)
estimator.fit(x_train, y_train)

# 7. 评估
y_pred = estimator.predict(x_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))