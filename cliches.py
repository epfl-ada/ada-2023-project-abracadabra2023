import numpy as np
import pandas as pd

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
