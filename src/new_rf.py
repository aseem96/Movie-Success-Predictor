import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier

def remove_string_cols(df):
    strlist = []
    for colname, colvalue in df.iteritems():
        if type(colvalue[1]) == str:
            strlist.append(colname)
    num_list = df.columns.difference(strlist)
    df = df[num_list]
    return df

def classify(row):
    if row['imdb_score'] >= 0 and row['imdb_score'] < 4:
        return 0
    elif row['imdb_score'] >= 4 and row['imdb_score'] < 6:
        return 1
    elif row['imdb_score'] >= 6 and row['imdb_score'] < 7:
        return 2
    elif row['imdb_score'] >= 7 and row['imdb_score'] < 8:
        return 3
    elif row['imdb_score'] >= 8 and row['imdb_score'] <= 10:
        return 4


def run_random_forest(df):
    x = np.array(df.ix[:, 0:])
    y = np.array(df['class'])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(random_state=1, n_estimators=250, min_samples_split=8, min_samples_leaf=4)
    rf.fit(x_train,y_train)
    pred = rf.predict(x_test)
    print("Accuracy: ",accuracy_score(y_test, pred))

if __name__ == '__main__':
    df = pd.read_csv('../data/movie_metadata.csv')
    df = remove_string_cols(df)
    df = df.fillna(value=0,axis=1)
    df["class"] = df.apply(classify, axis=1)
    df = df.drop('imdb_score', 1)
    df = df.drop('facenumber_in_poster', 1)
    df = df.drop('title_year', 1)
    df = df.drop('aspect_ratio', 1)
    df = df.drop('duration', 1)
    run_random_forest(df)
