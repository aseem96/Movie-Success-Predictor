import pandas as pd
from sklearn.preprocessing import StandardScaler
from pca import run_pca
from knn import run_knn
from random_forest import run_random_forest
from logistic import run_logistic_regression
from xgboostModel import run_xgboost_cornell
from xgboostModel import run_xgboost_imdb

def fill_nan(df_movie, col):
    df_movie[col] = df_movie[col].fillna(df_movie[col].median())

def data_prepocessing():
    df = pd.read_csv('../data/imdb.csv', error_bad_lines=False)
    df = df[df['year'] > 2000]
    df_movie = df[df['type'] != 'video.episode']

    cols = list(df_movie.columns)
    fill_nan(df_movie,cols)

    col = list(df_movie.columns)
    col.remove('type')
    col = col[5:15]

    sc = StandardScaler()
    temp = sc.fit_transform(df_movie[col])
    # df_movie[col] = temp

    df_standard = df_movie[list(df_movie.describe().columns)]
    return (df_movie, df_standard)

def classify(row):
    if row['imdbRating'] >= 0 and row['imdbRating'] < 4:
        return 0
    elif row['imdbRating'] >= 4 and row['imdbRating'] < 7:
        return 1
    elif row['imdbRating'] >= 7 and row['imdbRating'] <= 10:
        return 2

if __name__ == '__main__':
    df_movie, df_standard = data_prepocessing()
    run_pca(df_standard, df_movie)

    df_knn = df_movie
    df_knn["class"] = df_knn.apply(classify, axis=1)
    run_knn(df_knn)

    run_logistic_regression()

    run_xgboost_cornell()
    run_xgboost_imdb(df_knn)
