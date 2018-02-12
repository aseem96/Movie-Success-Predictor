from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from split_dataset import split_train_test
import numpy as np
import matplotlib.pyplot as plt

def plot_data(neighbors, MSE):
    plt.plot(neighbors, MSE)
    plt.xlabel('Number of Neighbors K')
    plt.ylabel('Misclassification Error')
    plt.show()

def run_knn(df_knn):
    print("\n\n----------------------K Nearest Neighbors----------------------\n\n")

    x = np.array(df_knn.ix[:, 0:])
    y = np.array(df_knn['class'])

    x_train, x_test, y_train, y_test = split_train_test(x,y)

    neighbors = [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49]
    cv_scores = []
    for k in neighbors:
        knn = KNeighborsClassifier(n_neighbors = k)
        scores = cross_val_score(knn, x_train, y_train, cv=10, scoring='accuracy')
        cv_scores.append(scores.mean())

    MSE = [1 - x for x in cv_scores]
    optimal_k = neighbors[MSE.index(min(MSE))]
    print ("The optimal number of neighbors is %d" % optimal_k)

    plot_data(neighbors, MSE)
