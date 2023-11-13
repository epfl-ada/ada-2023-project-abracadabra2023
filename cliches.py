import numpy as np


def get_df_keyword(df, key_word="United_Kingdom"):
    """
    given df, get all rows that have 'keyword' in their path. 
    """
    # get all paths with key_word in it (not matter its place in the path)
    df_with_key_word = df.iloc[
        np.where(np.char.rfind(np.array(df["path"], dtype="str"), key_word) >= 0)[0]
    ]
    return df_with_key_word