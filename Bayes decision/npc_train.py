# -*- coding: utf-8 -*-
# Example of Neyman-Pearson (NP) Classification Algorithms
# Richard Zhao, Yang Feng, Jingyi Jessica Li and Xin Tong

import numpy as np
import os
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from untitled0 import data, extract, extract_select
from nproc import npc


train_path = 'TwoLeadECG/TwoLeadECG_TRAIN.tsv'
test_path = 'TwoLeadECG/TwoLeadECG_TEST.tsv'
train_data = pd.read_csv(train_path, sep='\t', header = None).values
test_data = pd.read_csv(test_path, sep='\t', header = None).values
y_train, x_train = train_data[:, 0], train_data[:, 1:]
y_test, x_test = test_data[:, 0], test_data[:, 1:]
y_train[y_train != 1] = 0
y_test[y_test != 1] = 0


if __name__ == '__main__':
    test = npc()
    
    np.random.seed()
    X_train, X_test= extract_select(x_test, x_train)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    # Call the npc function to construct Neyman-Pearson classifiers.
    # The default type I error rate upper bound is alpha=0.05.
                    
    fit = test.npc(X_train, y_train, 'logistic', alpha = 0.45, delta = 0.3, n_cores=os.cpu_count())
      
    # Calculate the overall accuracy of the classifier as well as the realized 
    # type I error rate on test data.
    # Strictly speaking, to demonstrate the effectiveness of the fit classifier 
    # under the NP paradigm, we should repeat this experiment many times, and 
    # show that in 1 - delta of these repetitions, type I error rate is smaller than alpha.
    
    fitted_score = test.predict(fit,X_train)
    print("Accuracy on training set:", accuracy_score(fitted_score[0], y_train))
    pred_score = test.predict(fit,X_test)
    print("Accuracy on test set:", accuracy_score(pred_score[0], y_test))
    
    cm = confusion_matrix(y_test, pred_score[0])
    print("Confusion matrix:")
    print(cm)
    tn, fp, fn, tp = cm.ravel()
    print("Type I error rate: {:.5f}".format(fp/(fp+tn)))
    print("Type II error rate: {:.5f}".format(fn/(fn+tp)))
