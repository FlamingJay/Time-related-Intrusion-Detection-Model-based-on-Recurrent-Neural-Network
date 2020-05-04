import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

def load_data():
    # Default values.
    train_set = 'data/UNSW_NB15_training-set.csv'
    test_set = 'data/UNSW_NB15_testing-set.csv'
    train = pd.read_csv(train_set, index_col='id') //指定“id”这一列数据作为行索引
    test = pd.read_csv(test_set, index_col='id') //指定“id”这一列数据作为行索引

    # 二分类数据
    training_label = train['label'].values //将train的“label”这一列的值单独取出来
    testing_label = test['label'].values //将test的“label”这一列的值单独取出来
    temp_train = training_label
    temp_test = testing_label


    # Creates new dummy columns from each unique string in a particular feature 创建新的虚拟列
    unsw = pd.concat([train, test]) //将train和test拼接在一起
    unsw = pd.get_dummies(data=unsw, columns=['proto', 'service', 'state'])//将'proto', 'service', 'state'这三列使用one-hot-encoder转变
    # Normalising all numerical features:
    unsw.drop(['label', 'attack_cat'], axis=1, inplace=True)//删除'label', 'attack_cat'这两列，其中(inplace=True)是直接对原dataFrame进行操作
    unsw_value = unsw.values

    scaler = MinMaxScaler(feature_range=(0, 1)) //初始化MinMaxScaler
    unsw_value = scaler.fit_transform(unsw_value) //将待处理数据矩阵进行归一化
    train_set = unsw_value[:len(train), :] //分离出train集
    test_set = unsw_value[len(train):, :] //分离出test集

    # return train_set, training_label, test_set, testing_label
    return train_set, temp_train, test_set, temp_test
