import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def run_pca(df_standard, df_movie):
    print("\n\n----------------------Principal Component Analysis----------------------\n\n")
    pca = PCA(n_components=3)
    pca.fit(df_standard)
    df_pca = pca.transform(df_standard)
    print("\nExplained Variance: ",pca.explained_variance_ratio_)
    print("\n")

    df_standard['pca_one'] = df_pca[:, 0]
    df_standard['pca_two'] = df_pca[:, 1]
    df_standard['pca_three'] = df_pca[:, 2]

    plt.figure(figsize=(15,15))
    plt.scatter(df_standard['pca_one'][-50:], df_standard['pca_two'][-50:], color=['orange', 'cyan', 'brown'])
    for m, p1, p2 in zip(df_movie['wordsInTitle'], df_standard['pca_one'][-50:], df_standard['pca_two'][-50:]):
        plt.text(p1, p2, s=m, color=np.random.rand(3)*0.7)
    plt.show()
