#  kaggle入门竞赛-数字识别

## 比赛说明

> MNIST ("Modified National Institute of Standards and Technology") is the de facto “hello world” dataset of computer vision. Since its release in 1999, this classic dataset of handwritten images has served as the basis for benchmarking classification algorithms. As new machine learning techniques emerge, MNIST remains a reliable resource for researchers and learners alike.
> 
> In this competition, your goal is to correctly identify digits from a dataset of tens of thousands of handwritten images. We’ve curated a set of tutorial-style kernels which cover everything from regression to neural networks. We encourage you to experiment with different algorithms to learn first-hand what works well and how techniques compare.

## 一、数据分析

### 数据下载
- [数据下载](https://www.kaggle.com/c/digit-recognizer/data)

- 读取数据

```python
def open_csv():
    path = os.path.abspath('.')
    data = pd.read_csv(os.path.join(path, 'data/train.csv'))
    data1 = pd.read_csv(os.path.join(path, 'data/test.csv'))

    train_data = data.values[0:, 1:]
    train_label = data.values[0:, 0]
    test_data = data1.values[0:, 0:]

    return train_data, train_label, test_data

```
## 二、降维处理

为了加快模型的训练，我们可以用PCA对数据进行降维处理：

```python
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
```
## 三、模型选择

根据目标函数确定学习类型：监督学习、分类问题。

- 分类问题：0~9数字
- 常用算法：：knn、决策树、朴素贝叶斯、Logistic回归、SVM、集成方法（随机森林和 AdaBoost）

这里我们使用SVM进行训练：


```python
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

```

