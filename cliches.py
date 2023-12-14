import numpy as np
import pandas as pd
from scipy import stats
import spacy
from empath import Empath
import gensim
from gensim import corpora
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

__all__ = [
    "get_df_main_article",
    "get_category_main_article",
    "get_index_main_article_in",
    "test_difference_path_length_cliche",
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
    rating_main_article = get_df_main_article(
        df[df["rating"] == rating], main_article=main_article
    )

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
    rating_main_article_cliche = rating_main_article.loc[
        np.array(index_with_cliche).flatten()
    ]

    # get all paths without cliche
    index_wo_cliche = []
    for idx in rating_main_article.index:
        index_key_word = np.where(
            np.char.rfind(rating_main_article["path"].loc[idx], cliche) > 0
        )[0]
        if len(index_key_word) == 0:
            index_wo_cliche.append(idx)
    rating_main_article_nocliche = rating_main_article.loc[
        np.array(index_wo_cliche).flatten()
    ]

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


def top_k_categories(attributes, k=10, verbose=False):
    """
    get the k top topics of attributes
    input:
        attributes: string with text or words
    """
    lexicon = Empath()

    # topic detection
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(attributes)
    empath_features = lexicon.analyze(doc.text, normalize=True)

    # sort dict by strength of association
    sorted_categories = sorted(
        empath_features.items(), key=lambda x: x[1], reverse=True
    )

    if verbose:
        # get max length of topic for nice alignment printing
        max_length = max(len(category) for category, _ in sorted_categories[:10])

        for category, value in sorted_categories[:k]:
            print(f"{category.ljust(max_length)} {value:.3%}")

    return sorted_categories


def get_topics(article_name, k=5, l=10, data_path="data/articles_plain_text/"):
    """
    get topics of a given article.

    input:
        article_name: str
        k: int number of words to keep for each found topic
        l: int number of topics to keep for each found topic (empath)
        data_path: str
    """
    # get text
    with open(data_path + article_name, "r", encoding="utf-8") as file:
        text = file.read()

    # tokenize text (separates at each space bar)
    tokens = word_tokenize(text)

    # remove stop words; keep only alphatic words, eg not #, % etc.
    stop_words = set(stopwords.words("english"))
    tokens = [
        word.lower()
        for word in tokens
        if word.isalpha() and word.lower() not in stop_words
    ]

    # lemmatization (returns "base" form, eg drunk -> drink)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # create a dictionary and corpus
    dictionary = corpora.Dictionary([tokens])
    corpus = [dictionary.doc2bow(tokens)]

    # build the LDA model
    lda_model = gensim.models.LdaModel(corpus, num_topics=3, id2word=dictionary)

    # get topics (no name)
    topics = lda_model.print_topics(num_words=k)

    def get_words_topic(topic):
        """
        obtain only the word associated to the topic
        """
        return np.array(
            [
                topic.split("+")[i].split("*")[1].replace('"', "").strip()
                for i in range(len(topic.split("+")))
            ]
        )

    # only first topic (for now)
    return top_k_categories(" ".join(get_words_topic(topics[0][1])), l)
