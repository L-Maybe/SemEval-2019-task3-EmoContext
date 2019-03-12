#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/1/6 22:08
# @Author  : David
# @email   : mingren4792@126.com
# @File    : xgboost_semeval.py

import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import warnings
import io

from keras.utils import to_categorical
from sklearn.grid_search import GridSearchCV

warnings.filterwarnings("ignore")

x, d, t, y = pickle.load(open('./pickle/stacking_7565.pickle', 'rb'))
label2emotion = {0: "others", 1: "happy", 2: "sad", 3: "angry"}
emotion2label = {"others": 0, "happy": 1, "sad": 2, "angry": 3}

golden_label = pd.read_table('./data/dev.txt', sep='\t')
golden_label = golden_label['label'].replace(emotion2label)
golden_label = to_categorical(golden_label)


def getMetrics(predictions, ground):
    """Given predicted labels and the respective ground truth labels, display some metrics
    Input: shape [# of samples, NUM_CLASSES]
        predictions : Model output. Every row has 4 decimal values, with the highest belonging to the predicted class
        ground : Ground truth labels, converted to one-hot encodings. A sample belonging to Happy class will be [0, 1, 0, 0]
    Output:
        accuracy : Average accuracy
        microPrecision : Precision calculated on a micro level. Ref - https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin/16001
        microRecall : Recall calculated on a micro level
        microF1 : Harmonic mean of microPrecision and microRecall. Higher value implies better classification
    """
    # [0.1, 0.3 , 0.2, 0.1] -> [0, 1, 0, 0]
    discretePredictions = to_categorical(predictions)

    truePositives = np.sum(discretePredictions * ground, axis=0)
    falsePositives = np.sum(np.clip(discretePredictions - ground, 0, 1), axis=0)
    falseNegatives = np.sum(np.clip(ground - discretePredictions, 0, 1), axis=0)

    # print("True Positives per class : ", truePositives)
    # print("False Positives per class : ", falsePositives)
    # print("False Negatives per class : ", falseNegatives)

    # ------------- Macro level calculation ---------------
    macroPrecision = 0
    macroRecall = 0
    # We ignore the "Others" class during the calculation of Precision, Recall and F1
    for c in range(1, 4):
        precision = truePositives[c] / (truePositives[c] + falsePositives[c])
        macroPrecision += precision
        recall = truePositives[c] / (truePositives[c] + falseNegatives[c])
        macroRecall += recall
        f1 = (2 * recall * precision) / (precision + recall) if (precision + recall) > 0 else 0
        # print("Class %s : Precision : %.3f, Recall : %.3f, F1 : %.3f" % (label2emotion[c], precision, recall, f1))

    macroPrecision /= 3
    macroRecall /= 3
    macroF1 = (2 * macroRecall * macroPrecision) / (macroPrecision + macroRecall) if (macroPrecision + macroRecall) > 0 else 0
    # print("Ignoring the Others class, Macro Precision : %.4f, Macro Recall : %.4f, Macro F1 : %.4f" % (
    # macroPrecision, macroRecall, macroF1))

    # ------------- Micro level calculation ---------------
    truePositives = truePositives[1:].sum()
    falsePositives = falsePositives[1:].sum()
    falseNegatives = falseNegatives[1:].sum()

    # print("Ignoring the Others class, Micro TP : %d, FP : %d, FN : %d" % (truePositives, falsePositives, falseNegatives))

    microPrecision = truePositives / (truePositives + falsePositives)
    microRecall = truePositives / (truePositives + falseNegatives)

    microF1 = (2 * microRecall * microPrecision) / (microPrecision + microRecall) if ( microPrecision + microRecall) > 0 else 0
    # -----------------------------------------------------

    predictions = predictions
    ground = ground.argmax(axis=1)
    accuracy = np.mean(predictions == ground)

    # print("Accuracy : %.4f, Micro Precision : %.4f, Micro Recall : %.4f, Micro F1 : %.4f" % (
    # accuracy, microPrecision, microRecall, microF1))
    return accuracy, microPrecision, microRecall, microF1


def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(x, label=t.argmax(axis=1))
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds, show_progress=False)
        alg.set_params(n_estimators=cvresult.shape[0])


        # Fit the algorithm on the data
        alg.fit(x, t.argmax(axis=1), eval_metric='auc')

        # Predict training set:
        dtrain_predictions = alg.predict(t)
        # dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]
        print(dtrain_predictions)



def trainandTest(X_train, y_train, X_test):
    # 'learning_rate': 0.011, 'n_estimators': 10, 'max_depth': 4, 'min_child_weight': 1, 'seed': 0,
    # 'subsample': 0.7, 'colsample_bytree': 0.7, 'gamma': 1.7, 'reg_alpha': 1e-05, 'reg_lambda': 1,
    # 'scale_pos_weight': 1
    # XGBoost训练过程，下面的参数就是刚才调试出来的最佳参数组合i
    max = 0
    step = 0
    for i in np.arange(0,30, 1):
        print('-'*50)
        print(i)
        model = xgb.XGBClassifier(learning_rate= 0.02, n_estimators= 16, max_depth=3, imin_child_weight= 0, seed= 14,
                    subsample= 0.029, colsample_bytree=0.469, gamma=0.4, reg_alpha=0.015, reg_lambda=0.031,
                    scale_pos_weight= 1)
        model.fit(X_train, y_train)

            # 对测试集进行预
        ans = model.predict(X_test)

        accuracy, microPrecision, microRecall, microF1 = getMetrics(ans, golden_label)
        print(microF1)
        if max < microF1:
            max = microF1
            step = i
    print('*'*50)
    print(max, step)
    # with io.open('./test.txt', "w", encoding="utf8") as fout:
    #     fout.write('\t'.join(["id", "turn1", "turn2", "turn3", "label"]) + '\n')
    #     with io.open('./data/testwithoutlabels.txt', dencoding="utf8") as fin:
    #         fin.readline()
    #         for lineNum, line in enumerate(fin):
    #             fout.write('\t'.join(line.strip().split('\t')[:4]) + '\t')
    #             fout.write(label2emotion[ans[lineNum]] + '\n')

if __name__ == '__main__':
    trainandTest(x, y.argmax(axis=1),d)
    # optimization_function()



