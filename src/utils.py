from sklearn.cross_validation import train_test_split
import numpy as np
import unicodedata

def split_train_test(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    return(x_train, x_test, y_train, y_test)

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

def remove_string_cols(df):
    strlist = []
    for colname, colvalue in df.iteritems():
        if type(colvalue[1]) == str:
            strlist.append(colname)
    num_list = df.columns.difference(strlist)
    df = df[num_list]
    return df

def fill_nan(df, col):
    df[col] = df[col].fillna(df[col].median())
    return df

def clean_backward_title(col):
    string = col.rstrip()[:-2]
    ''.join((c for c in unicodedata.normalize('NFD', string) if unicodedata.category(c) != 'Mn'))
