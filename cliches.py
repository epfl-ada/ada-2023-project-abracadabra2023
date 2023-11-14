import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "get_df_keyword",
    "get_category_keyword",
    "get_index_keyword_in",
]


def get_df_keyword(paths_finished: pd.DataFrame, key_word="United_Kingdom"):
    """
    given paths_finished, get all rows that have 'keyword' in their path.
    """
    # get all paths with key_word in it (not matter its place in the path)
    df_with_key_word = paths_finished.iloc[
        np.where(
            np.char.rfind(
                np.array(
                    paths_finished["path"].apply(lambda path: ";".join(path)),
                    dtype="str",
                ),
                key_word,
            )
            >= 0
        )[0]
    ]
    return df_with_key_word


def get_category_keyword(key_word, path, categories):
    """
    return category of element (in path) with keyword in it
    """
    article = path[np.where(np.char.rfind(path, key_word) >= 0)[0][0]]
    cat = np.array(categories[categories["article"] == article]["category1"])
    return cat


def get_index_keyword_in(key_word, path):
    """
    get index of key_word in path
    input:
        key_word: str
        path: list of str
    """

    # find index that contains key_word
    index = np.where(np.char.rfind(path, key_word) >= 0)[0][0]

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
    key_word: str = "United_Kingdom",
    verbose: bool = True,
    return_df: bool = False,
):
    """
    test if there is a siginificant difference between the length of the path
    for a given key word, cliche and rating. More precisely, select paths with key word.
    Keep paths with given rating. Compare difference of length (shortest path vs actual path)
    for path with cliche and without cliche.
    input:
        df: whole data
        rating: rating to select from
        cliche: cliche of key_word
        key_word:
        verbose: display results
        return_df: return created dataframe or not
    """
    rating_uk = get_df_keyword(df[df["rating"] == rating], key_word=key_word)

    def print_results(data):
        print(f"Size data: {data.shape}")
        print(
            f"Mean difference path length: {data['diff_length'].mean()} ({np.std(data['diff_length'])})"
        )
        print(f"Median difference path length: {data['diff_length'].median()}")

    # get all paths with cliche London
    index_with_cliche = []
    for idx in rating_uk.index:
        index_key_word = np.where(
            np.char.rfind(rating_uk["path"].loc[idx], cliche) > 0
        )[0]
        if len(index_key_word) > 0:
            index_with_cliche.append(idx)
    rating_uk_cliche = rating_uk.loc[np.array(index_with_cliche).flatten()]

    # get all paths without cliche
    index_wo_cliche = []
    for idx in rating_uk.index:
        index_key_word = np.where(
            np.char.rfind(rating_uk["path"].loc[idx], cliche) > 0
        )[0]
        if len(index_key_word) == 0:
            index_wo_cliche.append(idx)
    rating_uk_nocliche = rating_uk.loc[np.array(index_wo_cliche).flatten()]

    # print results
    if verbose:
        print_results(rating_uk_cliche)
        print_results(rating_uk_nocliche)

    # ttest
    stat, p = stats.ttest_ind(
        rating_uk_cliche["diff_length"],
        rating_uk_nocliche["diff_length"],
        equal_var=False,
        alternative="less",
    )

    if verbose:
        print(f"stat={stat}, pvalue={p}")
        print(
            "Result is significant at 0.05" if p < 0.05 else "Result is not significant"
        )

    if return_df:
        return (stat, p, rating_uk_cliche, rating_uk_nocliche)
    else:
        return (
            stat,
            p,
            rating_uk_cliche["diff_length"].mean(),
            rating_uk_nocliche["diff_length"].mean(),
            rating_uk_cliche.shape,
            rating_uk_nocliche.shape,
        )
