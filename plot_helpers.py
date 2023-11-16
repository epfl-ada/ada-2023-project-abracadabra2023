import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

def plot_most_frequent_articles(data: pd.Series, type: str, pct: float=10, main_article: str="United_Kingdom"):
    """ 
    plot most frequent articles that precede/follow main_article
    input:
        data: percentage of apparition of articles
        type: precede/follow
        main_article: main_article for selection of articles
        pct: percentage of most frequent articles to display
    """
    most_frequent = data[data >= np.percentile(data, 100 - pct)]

    plt.figure(figsize=(12,5))
    sns.barplot(
        x=most_frequent.index, y=most_frequent.values, palette="coolwarm"
    )
    plt.yscale("log")
    plt.xlabel("most frequent articles")
    plt.ylabel("%")
    plt.title(f"Articles that {type} '{main_article}'")
    plt.rc('xtick', labelsize=8)
    plt.xticks(rotation=90)  # Rotate x-axis labels for readability
