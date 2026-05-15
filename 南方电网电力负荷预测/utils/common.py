import pandas as pd
import numpy as np


# 对数据预处理 -> 时间格式化, 按照时间升序排列, 且对数据去重
def data_preprocessing(file_path):
    # 1. 加载数据集
    data = pd.read_csv(file_path)
    # data.info()
    # 2. 时间格式化: '%Y%m%d%H%M%S'
    data["time"] = pd.to_datetime(data["time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    # 3. 按照时间升序排列
    data.sort_values(by="time", inplace=True)
    # 4. 去重
    data.drop_duplicates(inplace=True)
    return data


if __name__ == "__main__":
    data_preprocessing("../data/train.csv")
