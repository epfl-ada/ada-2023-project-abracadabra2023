import pandas as pd
from urllib.parse import unquote
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
from collections import Counter

__all__ = [
    "graph_data_prefix_path",
    "articles_plain_text_data_prefix_path",
    "articles_html_data_prefix_path",
    "articles_path",
    "categories_path",
    "links_path",
    "paths_finished_path",
    "paths_unfinished_path",
    "shortest_paths_path",
    "import_and_clean_data",
    "get_shortest_length",
]

# path to TSV file
graph_data_prefix_path = "data/graph"
articles_plain_text_data_prefix_path = "data/articles_plain_text"
articles_html_data_prefix_path = "data/articles_html"

articles_path = graph_data_prefix_path + "/articles.tsv"
categories_path = graph_data_prefix_path + "/categories.tsv"
links_path = graph_data_prefix_path + "/links.tsv"
paths_finished_path = graph_data_prefix_path + "/paths_finished.tsv"
paths_unfinished_path = graph_data_prefix_path + "/paths_unfinished.tsv"
shortest_paths_path = graph_data_prefix_path + "/shortest-path-distance-matrix.txt"


def import_and_clean_data():
    # import TSV file into DataFrame
    # articles = pd.read_csv(
    #     articles_path, delimiter="\t", comment="#", encoding="utf-8", header=None
    # )
    articles = np.genfromtxt(
        articles_path,
        comments="#",
        dtype="str",
        autostrip=True,
    )
    categories = pd.read_csv(
        categories_path, delimiter="\t", comment="#", encoding="utf-8", header=None
    )
    links = pd.read_csv(
        links_path, delimiter="\t", comment="#", encoding="utf-8", header=None
    )
    paths_finished = pd.read_csv(
        paths_finished_path, delimiter="\t", comment="#", encoding="utf-8", header=None
    )
    paths_unfinished = pd.read_csv(
        paths_unfinished_path,
        delimiter="\t",
        comment="#",
        encoding="utf-8",
        header=None,
    )
    # import theoretical shortest paths
    shortest_paths_matrix = np.genfromtxt(
        shortest_paths_path,
        comments="#",
        dtype="str",
    )

    # Decode URL-encoded names
    categories[0] = categories[0].apply(unquote)
    links[0] = links[0].apply(unquote)
    links[1] = links[1].apply(unquote)

    # Rename some columns for convenience
    categories = categories.rename(columns={0: "article"})
    categories = categories.rename(columns={1: "categories"})
    links.columns = ["linkSource", "linkTarget"]
    paths_finished.columns = [
        "hashedIpAddress",
        "timestamp",
        "durationInSec",
        "path",
        "rating",
    ]
    paths_unfinished.columns = [
        "hashedIpAddress",
        "timestamp",
        "durationInSec",
        "path",
        "target",
        "type",
    ]

    # Split 'Categories' columns into separate columns
    categories[["subject", "category1", "category2", "category3"]] = categories[
        "categories"
    ].str.split(".", expand=True)

    # Transform paths into list of article names and "<"
    paths_finished["path"] = paths_finished["path"].str.split(";")
    paths_unfinished["path"] = paths_unfinished["path"].str.split(";")

    # We remove the NaN's in rating by replacing them with 0
    paths_finished["rating"] = np.where(
        paths_finished["rating"].isnull(), 0, paths_finished["rating"]
    )

    ## add different miscalleneous columns -------------------------------------------------
    # adding path length column
    paths_finished["path_length"] = (
        paths_finished["path"].apply(lambda row: len(row) - 1).astype(int)
    )

    # keeping only finished paths with lengths below the 99th percentile (to exclude outliers)
    bound = 0.99
    bound_path_lengths = np.percentile(paths_finished["path_length"], bound * 100.0)
    paths_finished = paths_finished[paths_finished["path_length"] <= bound_path_lengths]

    # adding theoretical shortest path column
    paths_finished["shortest_path"] = (
        paths_finished["path"]
        .apply(lambda path: get_shortest_length(articles, shortest_paths_matrix, path))
        .astype(int)
    )

    # compute difference between shortest paths and taken path
    paths_finished["diff_length"] = (
        paths_finished["path_length"] - paths_finished["shortest_path"]
    )

    # format articles, paths (spaces, accents etc.)
    articles = pd.DataFrame(list(map(unquote, articles)))
    articles = articles.rename(columns={0: "article"})
    paths_finished["path"] = paths_finished["path"].apply(
        lambda path: unquote(";".join(path)).split(";")
    )

    return (
        articles,
        categories,
        links,
        paths_finished,
        paths_unfinished,
    )


def get_shortest_length(
    articles_names: pd.DataFrame, path_matrix: np.ndarray, path: list[str]
):
    """
    get shortest path length for path
    input:
        articles: all articles (give order for path_matrix)
        path_matrix: matrix with element length of shortest paths between articles
        path: list of articles from starting article to target article
    """
    # get index
    index_starting_article = np.where(path[0] == articles_names)[0][0]
    index_target_article = np.where(path[-1] == articles_names)[0][0]

    # store length
    try:
        length = path_matrix[index_starting_article][index_target_article]
    except Exception as e:
        print(f"Exception '{e}'")
        print(f"(row, col) = ({index_starting_article}, {index_target_article})")

    # if < 0, no paths
    return length if length != "_" else "-1"
