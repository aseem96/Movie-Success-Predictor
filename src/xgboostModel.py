import xgboost
from sklearn.metrics import accuracy_score
import numpy as np
from split_dataset import split_train_test
from sklearn.utils import shuffle

def run_xgboost_cornell():
    print("\n\n----------------------XGBoost on Cornell dataset----------------------\n\n")
    x = np.loadtxt("../data/cornell/X_train.txt")
    y = np.loadtxt("../data/cornell/y_train.txt", dtype = int)

    x_new, y_new = shuffle(x, y)
    x_train = x_new[:1000]
    y_train = y_new[:1000]
    x_test = x_new[1000:]
    y_test = y_new[1000:]

    model = xgboost.XGBClassifier()
    model.fit(x_train, y_train)

    pred = model.predict(x_train)
    accuracy = accuracy_score(y_train, pred)
    print("Training accuracy: %.2f%%" % (accuracy * 100.0))

    pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, pred)
    print("Testing accuracy: %.2f%%" % (accuracy * 100.0))

def run_xgboost_imdb(df_knn):
    print("\n\n----------------------XGBoost on IMDB dataset----------------------\n\n")
    x = np.array(df_knn.ix[:, 0:])
    y = np.array(df_knn['class'])

    x_train, x_test, y_train, y_test = split_train_test(x,y)
    x_train = np.delete(x_train,[0,1,2,3,4,5,9,44],axis=1)
    x_test = np.delete(x_test,[0,1,2,3,4,5,9,44],axis=1)

    model = xgboost.XGBClassifier()
    model.fit(x_train, y_train)

    pred = model.predict(x_train)
    accuracy = accuracy_score(y_train, pred)
    print("Training accuracy: %.2f%%" % (accuracy * 100.0))

    pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, pred)
    print("Testing accuracy: %.2f%%" % (accuracy * 100.0))
