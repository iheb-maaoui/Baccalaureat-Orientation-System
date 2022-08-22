##Features Selection Function (chi-squared)
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import pandas as pd
from sklearn import metrics
import numpy as np
from sklearn.model_selection import KFold 
from sklearn.model_selection import RepeatedKFold

def select_features(Xtrain, ytrain, Xtest,i):
    fs = SelectKBest(score_func=chi2, k=i)
    fs.fit(Xtrain, ytrain)
    Xtrain_fs = fs.transform(Xtrain)
    Xtest_fs = fs.transform(Xtest)
    return Xtrain_fs, Xtest_fs, fs


def SVM(Xtrain_fs,ytrain,Xtest_fs):
    from sklearn import svm
    clf = svm.SVC(kernel='linear') # Linear Kernel
    clf.fit(Xtrain_fs, ytrain)
    ypred = clf.predict(Xtest_fs)
    return ypred


def MLP(Xtrain_fs,ytrain,Xtest_fs):
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(50,), random_state=1)
    clf.fit(Xtrain_fs, ytrain)
    ypred = clf.predict(Xtest_fs)
    return ypred

def LR(Xtrain_fs,ytrain,Xtest_fs):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import KFold
    lr = LogisticRegression(random_state=1,max_iter=1000)
    lr.fit(Xtrain_fs, ytrain)
    ypred = lr.predict(Xtest_fs)
    return ypred

def final_model(Xtrain_fs, ytrain, Xtest_fs):
    y1 = np.array(LR(Xtrain_fs,ytrain,Xtest_fs))
    y2 = np.array(SVM(Xtrain_fs,ytrain,Xtest_fs))
    y3 = np.array(MLP(Xtrain_fs,ytrain,Xtest_fs))
    
    ypred = np.zeros(y1.shape)
    ypred = np.where(y1==y2,y1,y3)
    ypred = np.where(ypred==y2,ypred,y3)
    ypred = np.where(ypred==y1,ypred,y3)
    
    return ypred


X = np.load('X_cleaned.npy')
y = np.load('X_cleaned.npy')
rkf = RepeatedKFold(n_splits=3, n_repeats=10, random_state=17)
m1,m2,m3,m4 = 0,0,0,0
i=0

score = pd.DataFrame(columns=['Fusion','MLP','SVM','LR'])

for train_index, test_index in rkf.split(X):
    Xtrain, Xtest = X[train_index], X[test_index] 
    ytrain, ytest = y[train_index], y[test_index]
    Xtrain_fs, Xtest_fs, fs = select_features(Xtrain, ytrain, Xtest,'all')
    
    ypredfusion=final_model(Xtrain_fs, ytrain, Xtest_fs)
    m1+=metrics.accuracy_score(ypredfusion, ytest)
    
    ypredknn=MLP(Xtrain_fs, ytrain, Xtest_fs)
    m2+=metrics.accuracy_score(ypredknn, ytest)
    
    ypredsvm=SVM(Xtrain_fs, ytrain, Xtest_fs)
    m3+=metrics.accuracy_score(ypredsvm, ytest)
    
    ypredlr=LR(Xtrain_fs, ytrain, Xtest_fs)
    m4+=metrics.accuracy_score(ypredlr, ytest)
    
    i+=1
    score.loc[i-1] = [metrics.accuracy_score(ypredfusion, ytest),metrics.accuracy_score(ypredknn, ytest),metrics.accuracy_score(ypredsvm, ytest),metrics.accuracy_score(ypredlr, ytest)]


print(X)