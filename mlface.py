import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split, GridSearchCV


from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import time, random


def load_dataset(min_faces_per_person=2):
    data = fetch_lfw_people(min_faces_per_person=min_faces_per_person)#筛选出大于70张照片的人
    return data


def train(X_train, y_train, param, model_type='svc'):
    if model_type == 'knn':
        clf = GridSearchCV(KNeighborsClassifier(), param, n_jobs=-1, cv=5)
    elif model_type == 'svc':
        clf = GridSearchCV(SVC(class_weight='balanced'), param, n_jobs=-1, cv=5)
    else:
        raise Exception('Unkown model type {}'.format(model_type))  
    clf = clf.fit(X_train, y_train)

    print("Best estimator found by grid search:")
    print(clf.best_estimator_)

    return clf

def test(clf, x_test, data):
    print("Predicting people's names on the test set")
    start = time.time()

    y_pred = clf.predict(x_test)
    print('test time {}'.format(time.time() - start))
    print('test time {}'.format(time.strftime("%H:%M:%S", time.localtime(time.time() - start))))


    # print(classification_report(y_test, y_pred, target_names=[str(n) for n in np.unique(y_test)]))
    # print(confusion_matrix(y_test, y_pred, labels=range(len(np.unique(y_test)))))
    print(classification_report(y_test, y_pred, target_names=data.target_names))
    print(confusion_matrix(y_test, y_pred, labels=data.target_names))


def get_split_data(all_data, test_rate=0.3, shuffle=False):
        
        # 按照类别对数据进行划分
        images = all_data.data
        target = all_data.target  # 取出标签数据
        target_names = data.target_names
        assert len(target_names) == len(set(target))
        n_classes = len(target_names) # 40

        x_train, y_train, x_test, y_test = [], [], [], []
        for i in range(n_classes):
            idx = target == i
            class_data = images[idx]
            test_size = int(len(class_data) * test_rate) 


            x_tr = class_data[:-test_size]
            y_tr = target[idx][:-test_size]
            
            x_te = class_data[-test_size:]
            y_te = target[idx][-test_size:]
            

            x_train.append(x_tr)
            y_train.append(y_tr)
            x_test.append(x_te)
            y_test.append(y_te)


        return np.concatenate(x_train, axis=0), np.concatenate(x_test, axis=0), np.concatenate(y_train, axis=0), np.concatenate(y_test, axis=0)


if __name__ == "__main__":

    random_state = 42

    # 数据集设定
    min_faces_per_person = 5
    test_size = 0.2

    # PCA
    n_components = 150

    # svm
    model_type = 'svc' # knn
    # model_type = 'knn' # knn
    svm_param = [{'kernel':['linear'], 'C':[1, 10, 100, 500, 1000]},
                {'kernel':['poly'], 'C':[1, 5, 10, 15, 20], 'degree':[2, 3, 4, 5]}, 
                {'kernel':['rbf'], 'C':[1, 10, 100, 500, 1000], 'gamma':[1, 0.1, 0.01, 0.001]}]

    # KNN
    # knn = False
    knn_param = [{'weights':['uniform'], 'n_neighbors':[i for i in range(5, 50)]},
                {'weights':['distance'], 'n_neighbors':[i for i in range(5, 50)], 'p':[i for i in range(1, 6)]}]
    
    param = svm_param if model_type == 'svc' else knn_param

    data = load_dataset(min_faces_per_person=min_faces_per_person)

    print('Total image shape', data.images.shape)
    print('Total target shape', data.target.shape)
    print('Total target_names shape', data.target_names.shape)

    print(data.target)
    print(data.target_names)

    print('***'*20)


    # 划分数据集
    #按照8：2划分训练集与测试集
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = get_split_data(data, test_rate=test_size)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    print('data.target_names', data.target_names, len(data.target_names))
    print('np.unique(y_test)', np.unique(y_test), len(np.unique(y_test)))

    print('---'*20)


    print(y_test)
    print(len(y_test))
    print(np.unique(y_test))

    print('---'*20)

    print('train size', X_train.shape)
    print('test size', X_test.shape)


    # PCA
    print("Extracting the top %d eigenfaces from %d faces" % (n_components, X_train.shape[0]))
    pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X_train)
    X_train_pca = pca.transform(X_train) 
    X_test_pca = pca.transform(X_test)


    # model
    print("Fitting the classifier to the training set")
    start = time.time()
    clf = train(X_train_pca, y_train, param, model_type=model_type)
    print('train time {}'.format(time.time() - start))
    print('train time {}'.format(time.strftime("%H:%M:%S", time.localtime(time.time() - start))))
    print('-----'*20)




    test(clf, X_test_pca, data)









