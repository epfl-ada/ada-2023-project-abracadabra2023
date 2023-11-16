import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "get_df_main_article",
    "get_category_main_article",
    "get_index_main_article_in",
]


def get_df_main_article(paths_finished: pd.DataFrame, main_article="United_Kingdom"):
    """
    given paths_finished, get all rows that have 'keyword' in their path.
    """
    # get all paths with main_article in it (not matter its place in the path)
    df_with_main_article = paths_finished.iloc[
        np.where(
            np.char.rfind(
                np.array(
                    paths_finished["path"].apply(lambda path: ";".join(path)),
                    dtype="str",
                ),
                main_article,
            )
            >= 0
        )[0]
    ]
    return df_with_main_article


def get_category_main_article(main_article, path, categories):
    """
    return category of element (in path) with main_article in it
    """
    article = path[np.where(np.char.rfind(path, main_article) >= 0)[0][0]]
    cat = np.array(categories[categories["article"] == article]["category1"])
    return cat


def get_index_main_article_in(main_article, path):
    """
    get index of main_article in path
    input:
        main_article: str
        path: list of str
    """

    # find index that contains main_article
    index = np.where(np.char.rfind(path, main_article) >= 0)[0][0]

    if index == 0:
        # return just after
        return path[0:2]
    elif index != len(path) - 1:
        # return just before and just after

        # if there was a return, check next article
        i = 1
        while path[index + i] == "<":
            i += 1
        return path[index - 1 : index + i + 1]
    else:
        return path[index - 1 :]


def test_difference_path_length_cliche(
    df: pd.DataFrame,
    rating: int,
    cliche: str = "London",
    main_article: str = "United_Kingdom",
    verbose: bool = True,
    return_df: bool = False,
):
    """
    test if there is a siginificant difference between the length of the path
    for a given main article, cliche and rating. More precisely, select paths with main_article.
    Keep paths with given rating. Compare difference of length (shortest path vs actual path)
    for path with cliche and without cliche.
    input:
        df: whole data
        rating: rating to select from
        cliche: cliche of key_word
        main_article:
        verbose: display results
        return_df: return created dataframe or not
    """
    rating_main_article = get_df_main_article(df[df["rating"] == rating], main_article=main_article)

    def print_results(data):
        print(f"Size data: {data.shape}")
        print(
            f"Mean difference path length: {data['diff_length'].mean()} ({np.std(data['diff_length'])})"
        )
        print(f"Median difference path length: {data['diff_length'].median()}")

    # get all paths with cliche London
    index_with_cliche = []
    for idx in rating_main_article.index:
        index_key_word = np.where(
            np.char.rfind(rating_main_article["path"].loc[idx], cliche) > 0
        )[0]
        if len(index_key_word) > 0:
            index_with_cliche.append(idx)
    rating_main_article_cliche = rating_main_article.loc[np.array(index_with_cliche).flatten()]

    # get all paths without cliche
    index_wo_cliche = []
    for idx in rating_main_article.index:
        index_key_word = np.where(
            np.char.rfind(rating_main_article["path"].loc[idx], cliche) > 0
        )[0]
        if len(index_key_word) == 0:
            index_wo_cliche.append(idx)
    rating_main_article_nocliche = rating_main_article.loc[np.array(index_wo_cliche).flatten()]

    # print results
    if verbose:
        print_results(rating_main_article_cliche)
        print_results(rating_main_article_nocliche)

    # ttest
    stat, p = stats.ttest_ind(
        rating_main_article_cliche["diff_length"],
        rating_main_article_nocliche["diff_length"],
        equal_var=False,
        alternative="less",
    )

    if verbose:
        print(f"stat={stat}, pvalue={p}")
        print(
            "Result is significant at 0.05" if p < 0.05 else "Result is not significant"
        )

    if return_df:
        return (stat, p, rating_main_article_cliche, rating_main_article_nocliche)
    else:
        return (
            stat,
            p,
            rating_main_article_cliche["diff_length"].mean(),
            rating_main_article_nocliche["diff_length"].mean(),
            rating_main_article_cliche.shape,
            rating_main_article_nocliche.shape,
        )
