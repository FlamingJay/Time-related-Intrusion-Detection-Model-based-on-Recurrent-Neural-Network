import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

def load_data(if_binary=True):
    # Default values.
    train_set = 'F:/GlobalCOM/20190314/UNSW_NB15_training-set.csv'
    test_set = 'F:/GlobalCOM/20190314/UNSW_NB15_testing-set.csv'
    train = pd.read_csv(train_set, index_col='id')
    test = pd.read_csv(test_set, index_col='id')
    if if_binary:
        # 二分类数据
        training_label = train['label'].values
        testing_label = test['label'].values
        temp_train = training_label
        temp_test = testing_label
    else:
        # Encoding the category names into numbers so that they can be one hot encoded later.
        # 按照顺序，将类别进行整数编码
        attack_dict = {
            'Normal': 0, 'Fuzzers': 1, 'Analysis': 2, 'Backdoor': 3,
            'DoS': 4, 'Exploits': 5, 'Generic': 6, 'Reconnaissance': 7,
            'Shellcode': 8, 'Worms': 9
        }
        train['attack_cat'] = train['attack_cat'].apply(lambda x: attack_dict[x] if x in attack_dict.keys() else x)
        training_label_multi = train['attack_cat'].values
        test['attack_cat'] = test['attack_cat'].apply(lambda x: attack_dict[x] if x in attack_dict.keys() else x)
        testing_label_multi = test['attack_cat'].values

        temp_train = np.zeros((len(training_label_multi), 10))
        for i in range(len(training_label_multi)):
            temp_train[i, training_label_multi[i]] = 1
        temp_test = np.zeros((len(testing_label_multi), 10))
        for j in range(len(testing_label_multi)):
            temp_test[j, testing_label_multi[j]] = 1

    # Creates new dummy columns from each unique string in a particular feature
    unsw = pd.concat([train, test])
    unsw = pd.get_dummies(data=unsw, columns=['proto', 'service', 'state'])
    # Normalising all numerical features:
    unsw.drop(['label', 'attack_cat'], axis=1, inplace=True)
    unsw_value = unsw.values

    scaler = MinMaxScaler(feature_range=(0, 1))
    unsw_value = scaler.fit_transform(unsw_value)
    train_set = unsw_value[:len(train), :]
    test_set = unsw_value[len(train):, :]

    # return train_set, training_label, test_set, testing_label
    return train_set, temp_train, test_set, temp_test
