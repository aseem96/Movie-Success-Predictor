from sklearn.cross_validation import train_test_split
import numpy as np

def split_train_test(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    x_train = np.delete(x_train,[0,1,2,3,4,5,9,44],axis=1)
    x_test = np.delete(x_test,[0,1,2,3,4,5,9,44],axis=1)
    return(x_train, x_test, y_train, y_test)
