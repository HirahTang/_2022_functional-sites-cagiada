import sys, os

import pandas as pd
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import mdtraj as md
import random
import seaborn as sns
import re

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score


from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier, Pool,cv
from catboost.datasets import titanic
from graphviz import Digraph

def MCC_test_subset(prediction,test):
    TP=0
    FP=0
    total=0
    for i in range(len(prediction)):
        if np.isnan(prediction[i])!= True and np.isnan(test[i])!=True:
            if prediction[i]==1:
                if int(prediction[i])==int(test[i]):
                    TP+=1
                else:
                    FP+=1
            if int(test[i])==1:
                total+=1
    precision=TP/(TP+FP)
    recall=TP/total
    mcc=np.sqrt(abs(precision*recall))
    F1 = 2 * (precision*recall)/(precision+recall)
    
    return precision,recall,mcc  ,F1               

def MCC_test_multi(prediction,test):
    mcc=[]
    temp=[]
    for v in range(3):
        TP=0
        FP=0
        total=0
        mcc=[]
        for i in range(len(prediction)):
            if np.isnan(prediction[i])!= True and np.isnan(test[i])!=True:
                if prediction[i]==v:
                    if int(prediction[i])==int(test[i]):
                        TP+=1
                    else:
                        FP+=1
                if int(test[i])==v:
                    total+=1
        precision=TP/(TP+FP)
        recall=TP/total
        temp.append(np.sqrt(abs(precision*recall)))
    mcc=[i for i in temp]
    
    return mcc

def model_define():
    cat=CatBoostClassifier(iterations=2500, random_strength= 1, depth= 7, l2_leaf_reg= 9, bagging_temperature= 2,verbose=0,class_weights={0:1,1:2,2:1,3:0.1})
    kfold=RepeatedStratifiedKFold(n_splits=5,n_repeats=2,random_state=1)
    return cat,kfold
def main():
    
    # Loading the dataset for CatBoost formatting
    
    mutation_data = pd.read_csv("/home/qcx679/hantang/_2022_functional-sites-cagiada/output.csv")
    print("Using all data split to train/test set")
    X = mutation_data[["0", "1", "2", "3", "4", "5", "6", "7"]]
    X = X.to_numpy()
    Y = mutation_data['target']
    Y = Y.to_numpy()
    cat, kfold = model_define()
    MCC=[0,0,0,0]
    MCC_multi=[]
    count=0
    for train_index, test_index in kfold.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        
        cat.fit(X_train,Y_train)
        prediction=cat.predict(X_test)

        MCC[0]+=MCC_test_subset(prediction,Y_test)[0]
        MCC[1]+=MCC_test_subset(prediction,Y_test)[1]
        MCC[2]+=MCC_test_subset(prediction,Y_test)[2] 
        MCC[3]+=MCC_test_subset(prediction,Y_test)[3]
        MCC_multi.append(MCC_test_multi(prediction,Y_test))
        
        count+=1
    
    print("MCC: ",MCC[0]/count,MCC[1]/count,MCC[2]/count,MCC[3]/count)

    
    print(X.shape, Y.shape)

if __name__ == "__main__":
    main()    
    
