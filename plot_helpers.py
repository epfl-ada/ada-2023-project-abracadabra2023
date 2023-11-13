import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

def plot_most_frequent_articles(data: pd.Series, type: str, pct: float=10, key_word: str="United_Kingdom"):
    """ 
    plot most frequent articles that precede/follow keyword
    input:
        data: percentage of apparition of articles
        type: precede/follow
        key_word: key_word for selection of articles
        pct: percentage of most frequent articles to display
    """
    most_frequent = data[data >= np.percentile(data, 100 - pct)]

    fig, ax = plt.subplots(figsize=(12,5))
    sns.barplot(
        x=most_frequent.index, y=most_frequent.values
    )
    ax.set_title(f"Articles that {type} '{key_word}'")
    ax.set_xlabel("most frequent articles")
    ax.set_ylabel("%")

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    return fig, ax
