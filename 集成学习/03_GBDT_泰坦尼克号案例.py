"""
案例:
    演示 Boosting思想之  GBDT(Gradient Boosting Decision Tree, 梯度提升树) 处理 泰坦尼克号数据集.

GBDT 梯度提升树解释:
    概述:
        通过拟合 负梯度 来获取一个强学习器
    流程:
        1. 采用所有目标值的均值 作为第1个弱学习器的 预测值.
        2. 目标值 - 预测值 = 负梯度(残差), 该(列)值作为 第2个弱学习器的 目标值.
        3. 针对于第1个弱学习器, 依次计算每个分割点的 最小平方和, 找到最佳 分割点, 至此: 第1个弱学习器搭建完毕.
        4. 把上述的分割点带入第2个弱学习器, 计算它的预测值 = 以此分割点为界, 目标值的均值, 即为该部分数据的 预测值.
        5. 计算第2个弱学习器的 负梯度, 最佳分割点, 至此: 第2个弱学习器搭建完毕.
        6. 以此类推, 直至程序结束.
"""

# 导入库
import pandas as pd
from sklearn.model_selection import train_test_split        # 切割训练集 和 测试集
from sklearn.tree import DecisionTreeClassifier             # 决策树分类器
from sklearn.ensemble import GradientBoostingClassifier     # 梯度提升树分类器
from sklearn.metrics import classification_report, accuracy_score  # 模型评估
from sklearn.model_selection import GridSearchCV            # 网格搜索


# 1. 读取数据.
df = pd.read_csv('./data/train.csv')
# df.info()

# 2. 数据的预处理.
# 2.1 提取 特征 和 标签.
x = df[['Pclass', 'Sex', 'Age']].copy()
y = df['Survived'].copy()

# 2.2 处理Age列的缺失值, 用该列的均值填充.
# x['Age'].fillna(x['Age'].mean(), inplace=True)
x['Age'] = x['Age'].fillna(x['Age'].mean())

# 2.3 热编码处理字符串类型.
x = pd.get_dummies(x)

# 2.4 切割训练集 和 测试集.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=23)

# 3. 特征工程, 这里略.

# 4. 模型训练, 预测, 评估.
# 场景1: 单个决策树对象(CART).
# 4.1 创建模型对象.
estimator = DecisionTreeClassifier()
# 4.2 模型训练.
estimator.fit(x_train, y_train)
# 4.3 模型预测.
y_predict = estimator.predict(x_test)
print(f'单个决策树对象的预测结果: {y_predict}')
# 4.4 模型评估.
# print(f'单个决策树对象的分类评估报告: \n{classification_report(y_test, y_predict)}')
print(f'单个决策树对象的准确率: {accuracy_score(y_test, y_predict)}')      # 0.8044692737430168
print('-' * 23)

# 场景2: 梯度提升树对象(GBDT).
# 4.1 创建模型对象.
estimator2 = GradientBoostingClassifier()        # 默认值
# 4.2 模型训练.
estimator2.fit(x_train, y_train)
# 4.3 模型预测.
y_predict = estimator2.predict(x_test)
print(f'梯度提升树对象的预测结果: {y_predict}')
# 4.4 模型评估.
print(f'梯度提升树对象的准确率: {accuracy_score(y_test, y_predict)}')      # 0.8100558659217877


# 场景3: 针对于GBDT(梯度提升树)模型, 进行参数调优.
# 4.1 定义模型可选参数
param_dict = {
    'n_estimators': [50, 60],              # 弱学习器的数量
    'learning_rate': [0.3, 0.5],                       # 学习率
    'max_depth': [3, 5]                              # 树最大深度
}
# 4.2 创建 梯度提升树 模型对象.
estimator3 = GradientBoostingClassifier()
# estimator3.fit(x_train, y_train)

# 4.3 创建网格搜索对象.
estimator4 = GridSearchCV(estimator3, param_dict, cv=2)
# 4.4 模型训练.
estimator4.fit(x_train, y_train)
# 4.5 模型评估
print(f'网格搜索后的模型准确率: {estimator4.best_score_}')
print(f'网格搜索后的模型: {estimator4.best_estimator_}')