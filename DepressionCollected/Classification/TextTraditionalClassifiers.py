from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import os
import pickle
import random
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


prefix = os.path.abspath(os.path.join(os.getcwd(), "."))
text_features = np.load(os.path.join(prefix, 'Features/TextWhole/whole_samples_clf_avg.npz'))['arr_0']
text_targets = np.load(os.path.join(prefix, 'Features/TextWhole/whole_labels_clf_avg.npz'))['arr_0']
text_dep_idxs_tmp = np.where(text_targets == 1)[0]
text_non_idxs = np.where(text_targets == 0)[0]

def model_performance(y_test, y_test_pred_proba):
    """
    Evaluation metrics for network performance.
    """
#     y_test_pred = y_test_pred_proba.data.max(1, keepdim=True)[1]
    y_test_pred = y_test_pred_proba

    # Computing confusion matrix for test dataset
    conf_matrix = standard_confusion_matrix(y_test, y_test_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    return y_test_pred, conf_matrix

def standard_confusion_matrix(y_test, y_test_pred):
    [[tn, fp], [fn, tp]] = confusion_matrix(y_test, y_test_pred)
    return np.array([[tp, fp], [fn, tn]])

train_idxs_tmps = [np.load(os.path.join(prefix, 'Features/TextWhole/train_idxs_0.63_1.npy'), allow_pickle=True),
np.load(os.path.join(prefix, 'Features/TextWhole/train_idxs_0.65_2.npy'), allow_pickle=True),
np.load(os.path.join(prefix, 'Features/TextWhole/train_idxs_0.60_3.npy'), allow_pickle=True)]
precs, recs, f1s = [], [], []

for idx_idx, train_idxs_tmp in enumerate(train_idxs_tmps):
    test_idxs_tmp = list(set(list(text_dep_idxs_tmp)+list(text_non_idxs)) - set(train_idxs_tmp))
    train_idxs, test_idxs = [], []

    # depression data augmentation
    for idx in train_idxs_tmp:
        if idx in text_dep_idxs_tmp:
            feat = text_features[idx]
            count = 0
            resample_idxs = [0,1,2,3,4,5]
            for i in itertools.permutations(feat, feat.shape[0]):
                if count in resample_idxs:
                    text_features = np.vstack((text_features, np.expand_dims(list(i), 0)))
                    text_targets = np.hstack((text_targets, 1))
                    train_idxs.append(len(text_features)-1)
                count += 1
        else:
            train_idxs.append(idx)

    for idx in test_idxs_tmp:
        if idx in text_dep_idxs_tmp:
            feat = text_features[idx]
            count = 0
            # resample_idxs = random.sample(range(6), 4)
            resample_idxs = [0,1,4,5]
            for i in itertools.permutations(feat, feat.shape[0]):
                if count in resample_idxs:
                    text_features = np.vstack((text_features, np.expand_dims(list(i), 0)))
                    text_targets = np.hstack((text_targets, 1))
                    test_idxs.append(len(text_features)-1)
                count += 1
        else:
            test_idxs.append(idx)
    # train_idxs = train_idxs_tmp
    # test_idxs = test_idxs_tmp

    X_train = text_features[train_idxs]
    Y_train = text_targets[train_idxs]
    X_test = text_features[test_idxs]
    Y_test = text_targets[test_idxs]

    # Decision Tree
    from sklearn import tree
    clf = tree.DecisionTreeClassifier(max_depth=20)

    # svm
    # from sklearn.svm import SVC
    # clf = SVC(kernel='rbf', gamma='auto')

    # rf
    # from sklearn.ensemble import RandomForestClassifier
    # clf = RandomForestClassifier(n_estimators=10, max_depth=20)

    # lr
    # from sklearn.linear_model import LogisticRegression
    # clf = LogisticRegression()

    clf.fit([f.flatten() for f in X_train], Y_train)
    pred = clf.predict([f.flatten() for f in X_test])
    # clf.fit([f.sum(axis=0) for f in X_train], Y_train)
    # pred = clf.predict([f.sum(axis=0) for f in X_test])

    y_test_pred, conf_matrix = model_performance(Y_test, pred)

    # custom evaluation metrics
    print('Calculating additional test metrics...')
    accuracy = float(conf_matrix[0][0] + conf_matrix[1][1]) / np.sum(conf_matrix)
    precision = float(conf_matrix[0][0]) / (conf_matrix[0][0] + conf_matrix[0][1])
    recall = float(conf_matrix[0][0]) / (conf_matrix[0][0] + conf_matrix[1][0])
    f1_score = 2 * (precision * recall) / (precision + recall)
    print("Accuracy: {}".format(accuracy))
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1-Score: {}\n".format(f1_score))
    print('='*89)
    # precs.append(0 if np.isnan(precision) else precision)
    # recs.append(0 if np.isnan(recall) else recall)
    # f1s.append(0 if np.isnan(f1_score) else f1_score)
    precs.append(precision)
    recs.append(recall)
    f1s.append(f1_score)
print(np.mean(precs), np.mean(recs), np.mean(f1s))