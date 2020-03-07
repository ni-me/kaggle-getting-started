import os
import csv
import datetime
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


def open_csv():
    path = os.path.abspath('.')
    data = pd.read_csv(os.path.join(path, 'data/train.csv'))
    data1 = pd.read_csv(os.path.join(path, 'data/test.csv'))

    train_data = data.values[0:, 1:]
    train_label = data.values[0:, 0]
    test_data = data1.values[0:, 0:]

    return train_data, train_label, test_data


def dRPCA(train_data, test_data, COMPONENT_NUM):
    print('dimensionality reduction...')
    trainData = np.array(train_data)
    testData = np.array(test_data)

    pca = PCA(n_components=COMPONENT_NUM, whiten=False)
    pca.fit(trainData)
    pcaTrainData = pca.transform(trainData)
    pcaTestData = pca.transform(testData)

    print('特征数量: %s' % pca.n_components_)
    print('总方差占比: %s' % sum(pca.explained_variance_ratio_))

    return pcaTrainData, pcaTestData


def save_csv(result, filename):
    with open(filename, 'w', newline='') as f:
        my_writer = csv.writer(f, dialect='excel')
        my_writer.writerow(['ImageId', 'Label'])
        index = 0
        for i in result:
            tmp = []
            index += 1
            tmp.append(index)
            tmp.append(int(i))
            my_writer.writerow(tmp)



train_data, train_label, test_data = open_csv()

train_data_pca, test_data_pca = dRPCA(train_data, test_data, 0.90)

'''
param_grid = {'C' : [1, 10, 100],
              'gamma' : [0.1, 1, 10]}

grid_search = GridSearchCV(SVC(), param_grid=param_grid, cv=2, n_jobs=-1)
grid_search.fit(train_data_pca, train_label)
predict_label = grid_search.predict(test_data_pca)

print(predict_label[1:20])
print('best parameters: %s' % grid_search.best_params_)
print('Best score: %s' % grid_search.best_score_)
'''
svc = SVC(C=10)
svc.fit(train_data_pca, train_label)
predict_label = svc.predict(test_data_pca)
save_csv(predict_label, os.path.join(os.path.abspath('.'), 'data/result.csv'))