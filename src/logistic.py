from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
import numpy as np

def run_logistic_regression():
    print("\n\n----------------------Logistic Regression----------------------\n\n")
    x = np.loadtxt("../data/cornell/X_train.txt")
    y = np.loadtxt("../data/cornell/y_train.txt", dtype = int)

    x_new, y_new = shuffle(x, y)
    x_train = x_new[:1000]
    y_train = y_new[:1000]
    x_test = x_new[1000:]
    y_test = y_new[1000:]

    logistic = LogisticRegression()
    logistic.fit(x_train,y_train)

    pred_train = logistic.predict(x_train)
    print ("Training accuracy: ", (logistic.score(x_train, y_train) * 100))

    pred_test = logistic.predict(x_test)
    print ("Testing accuracy: ", (logistic.score(x_test, y_test) * 100))
