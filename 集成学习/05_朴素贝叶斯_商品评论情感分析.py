"""
案例:
    演示通过 朴素贝叶斯算法 实现  商品评论情感分析, 即: 好评, 差评...

朴素贝叶斯介绍:
    概述:
        贝叶斯: 仅仅依赖 概率 就可以进行分类的 一种机器学习算法.
        朴素:   不考虑特征之间的关联性, 即: 特征间都是相互独立的.
            原始:  P(AB) = P(A) * P(B|A) = P(B) * P(A|B)
            加入朴素后: P(AB) = P(A) * P(B)
    细节:
        因为我们分词要用到 jieba分词器, 记得先装一下, 例如: pip install jieba
"""

# 导包
import numpy as np  # 数学计算包
import pandas as pd  # 数据处理包
import matplotlib.pyplot as plt  # 画图包
import jieba  # 分词包
from sklearn.feature_extraction.text import (
    CountVectorizer,
)  # 词频统计包, 把评论内容 转成 词频矩阵.
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB  # 朴素贝叶斯对象


# 1. 读取文件, 获取到原始数据.
df = pd.read_csv("./data/书籍评价.csv", encoding="gbk")
# df.info()

# 2. 数据预处理.
# 2.1 添加labels列, 充当: 标签列.  好评 -> 1, 差评 -> 0
df["labels"] = np.where(df["评价"] == "好评", 1, 0)
# df.info()
# print(df)

# 2.2 抽取 labels列, 作为: 标签.
y = df["labels"]
# 2.3 演示 jieba 分词
# print(
#     jieba.lcut(
#         "好好学习, 天天向上! 我爱你你爱我, 蜜雪冰城甜蜜蜜! 小明骑车, 一把把把把住了."
#     )
# )

# 2.4 对用户的评论信息, 做切词.
# 数据格式: [[第1条评论切词1, 切词2, 切词3...], [第2条评论切词1, 切词2, 切词3...], ...]
comment_list = [",".join(jieba.lcut(line)) for line in df["内容"]]
# 数据格式: ['第1条评论切词1, 切词2, 切词3...', '第2条评论切词1, 切词2, 切词3...', ...]
# print(comment_list)

# 2.5 加载 停用词列表, 即: 里边记录的词, 不需要参与模型训练, 预测, 要被删除的词, 例如: 的, 啊, 哈, 从, 都...
with open("./data/stopwords.txt", encoding="utf-8") as src_f:
    # 2.5.1 一次读取所有的行
    stopwords_list = src_f.readlines()
    # 2.5.2 删除最后的 '\n'
    stopwords_list = [line.strip() for line in stopwords_list]
    # 2.5.3 对 停用词列表去重.
    stopwords_list = list(set(stopwords_list))

# 2.6 创建向量化对象, 从 评论切词列表(comment_list) 中 删除 停用词, 并且统计词频(单词矩阵).
transfer = CountVectorizer(stop_words=stopwords_list)  # 参数: 停用词列表.
# 2.7 统计词频矩阵, 先训练, 后转换, 在转数组.
transfer.fit(comment_list)
# x的格式: [[第1条评论的切词分布, 有就是1, 没有就是0], [第2条评论的切词分布, 有就是1, 没有就是0], ...]
x = transfer.transform(comment_list).toarray()
print(x)

# 2.8 看一下 我们13条评论, 切词, 且删除 停用词后, 一共剩下多少个词了.
print(transfer.get_feature_names_out())
print(
    len(transfer.get_feature_names_out())
)  # 37个词, 即: 13条评论, 切词, 且删除 停用词后, 一共剩下多少个词了.

# 2.9 因为就 13条数据, 我们把前10条当训练集, 后三条当测试集.
x_train = x[:10]
y_train = y[:10]
x_test = x[10:]
y_test = y[10:]

# 3. 特征工程, 此处略.

# 4. 模型训练.
estimator = MultinomialNB()  # 创建朴素贝叶斯模型对象.
estimator.fit(x_train, y_train)

# 5. 模型预测.
y_pred = estimator.predict(x_test)
print(f"模型预测结果: {y_pred}")

# 6. 模型评估.
print(f"准确率: {accuracy_score(y_test, y_pred)}")
