from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
import os
from KNN_KDTree import KDTree_like_sklearn


def get_score(y_predict, y_test):
    ans_eql = 0
    for i in range(y_test.shape[0]):
        if(y_predict[i] == y_test[i]):
            ans_eql += 1
    return ans_eql * 1.0 / y_test.shape[0]

def evaluate_classifier(clf, testX, testY):
    # print('dsafdsafasdf')
    y_pre = clf.predict(testX)
    if isinstance(clf, KDTree_like_sklearn):
        print('use: ', str(clf.dist_kind), str(clf.way))
    else:
        print('standard')
    print(classification_report(testY, y_pre, digits=5))
    return get_score(y_pre, testY)

def evaluate_classifier_parameters(clf, data, target, split_ratio):
    trainX, testX, trainY, testY = train_test_split(data, target, train_size=split_ratio, random_state=0)
    clf.fit(trainX, trainY)
    y_pre = clf.predict(testX)
    return get_score(y_pre, testY)

def load_train_data():
    trainX = list()
    trainY = list()
    dir_path = './digits/trainingDigits'
    for filename in os.listdir(dir_path):
        # print(filename)
        trainY.append(int(filename[0]))
        t = np.loadtxt(os.path.join(dir_path, filename), str)
        tt = np.zeros(32 * 32)
        for i in range(32):
            for j in range(32):
                if t[i][j] == '1':
                    tt[i * 32 + j] = 1
                else:
                    tt[i * 32 + j] = 0
        trainX.append(tt)
    trainX = np.array(trainX)
    trainY = np.array(trainY)
    return trainX, trainY
    # print(trainX[0:2])


def load_test_data():
    testX = list()
    testY = list()
    dir_path = './digits/testDigits'
    for filename in os.listdir(dir_path):
        # print(filename)
        testY.append(int(filename[0]))
        t = np.loadtxt(os.path.join(dir_path, filename), str)
        tt = np.zeros(32 * 32)
        for i in range(32):
            for j in range(32):
                if t[i][j] == '1':
                    tt[i * 32 + j] = 1
        testX.append(tt)
    testX = np.array(testX)
    testY = np.array(testY)
    return testX, testY

load_train_data()

trainX, trainY = load_train_data()
testX, testY = load_test_data()

pca = PCA(n_components=32)
pca.fit(trainX)
trainX = pca.transform(trainX)
testX = pca.transform(testX)

trainX_T = trainX.T
D = np.cov(trainX_T)
print(trainX_T[0][0:10])
invD = np.linalg.pinv(D)

def get_hyperparameter_clf():
    '''
    q = np.array([1, 3, 5, 9, 17])
    ans_para = np.zeros((5, 5))
    for i in range(5):
        print(i)
        clf0 = KNeighborsClassifier(n_neighbors=q[i])
        score_t = evaluate_classifier_parameters(clf0, trainX, trainY, 0.8)
        ans_para[0][i] = score_t
        clf1 = KDTree_like_sklearn(dist_kind='simple', k=q[i])
        score_t = evaluate_classifier_parameters(clf1, trainX, trainY, 0.8)
        ans_para[1][i] = score_t
        clf2 = KDTree_like_sklearn(dist_kind='Mahalanobis', k=q[i], mat=invD)
        score_t = evaluate_classifier_parameters(clf2, trainX, trainY, 0.8)
        ans_para[2][i] = score_t
        clf3 = KDTree_like_sklearn(dist_kind='simple', k=q[i], mat=invD, way='wknn')
        score_t = evaluate_classifier_parameters(clf3, trainX, trainY, 0.8)
        ans_para[3][i] = score_t
        clf4 = KDTree_like_sklearn(dist_kind='Mahalanobis', k=q[i], mat=invD, way='wknn')
        score_t = evaluate_classifier_parameters(clf4, trainX, trainY, 0.8)
        ans_para[4][i] = score_t
    print(ans_para)
    ans_max = np.zeros(5, dtype=int)
    for i in range(5):
        ans_max[i] = q[np.argmax(ans_para[i])]
    print(ans_max)
    '''
    ans_max = np.array([3, 3, 1, 9, 5])
    clf0 = KNeighborsClassifier(n_neighbors=ans_max[0])
    clf0.fit(trainX, trainY)
    clf1 = KDTree_like_sklearn(dist_kind='simple', k=ans_max[1])
    clf1.fit(trainX, trainY)
    clf2 = KDTree_like_sklearn(dist_kind='Mahalanobis', k=ans_max[2], mat=invD)
    clf2.fit(trainX, trainY)
    clf3 = KDTree_like_sklearn(dist_kind='simple', k=ans_max[3], mat=invD, way='wknn')
    clf3.fit(trainX, trainY)
    clf4 = KDTree_like_sklearn(dist_kind='Mahalanobis', k=ans_max[4], mat=invD, way='wknn')
    clf4.fit(trainX, trainY)
    return (clf0, clf1, clf2, clf3, clf4)


clf0, clf1, clf2, clf3, clf4 = get_hyperparameter_clf()
#
#
print('start!!!')
score0 = evaluate_classifier(clf0, testX, testY)
print(score0)
print('adfasd')
score1 = evaluate_classifier(clf1, testX, testY)
print(score1)
score2 = evaluate_classifier(clf2, testX, testY)
print(score2)
score3 = evaluate_classifier(clf3, testX, testY)
print(score3)
score4 = evaluate_classifier(clf4, testX, testY)
print(score4)



names = ['sklearn-std', 'simple', 'Mahalanobis', 'simple+w-knn', 'Mahalanobis+w-knn']
x = [0, 1, 2, 3, 4]
y = [score0, score1, score2, score3, score4]
plt.plot(x, y, 'ro-')
plt.xticks(x, names, rotation=20)
plt.margins(0.08)
plt.subplots_adjust(bottom=0.15)
plt.show()






