import csv
import os
from ast import literal_eval

import gensim
import nltk
import numpy as np
import pandas as pd
import spacy
from empath import Empath
from gensim import corpora
from nltk.corpus import PlaintextCorpusReader, stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# from cliches import get_topics, top_k_categories

# download if first time running library
download = True

if download:
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("wordnet")


__all__ = [
    "data_path",
    "articles_plain_text_data_prefix_path",
    "articles_plain_text_path",
    "article_topics_file_path",
    "seegull_file_path",
    "write_to_csv",
    "top_k_categories",
    "get_topics",
    "generate_seegull_topics",
    "generate_articles_topics",
    "load_articles_topics",
    "generate_token_pos_tags",
    "load_token_pos",
    "generate_links_commmon",
    "load_links_common",
]

data_path = "data"
articles_plain_text_data_prefix_path = "data/articles_plain_text"
articles_plain_text_path = os.path.join(data_path, "articles_plain_text")
article_topics_file_path = os.path.join(data_path, "topics_articles.csv")
seegull_file_path = os.path.join(data_path, "seegull.csv")


def write_to_csv(
    data_path_article: str,
    df: pd.DataFrame,
    columns: list[str],
    header: bool = True,
    index: bool = False,
):
    if os.path.isfile(data_path_article):
        with open(data_path_article, "a") as csvfile:
            csvwriter = csv.writer(csvfile)

            csvwriter.writerows(df.to_numpy())
    else:
        df.to_csv(
            data_path_article,
            sep=",",
            encoding="utf-8",
            columns=columns,
            header=header,
            index=index,
        )


def top_k_categories(attributes: str, k=10, verbose=False):
    """
    get the k top topics of attributes
    :param attributes: string with text or words
    :param k: number of topics to extract (the top k)
    :param verbose: print the topics and their associated percentage
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
        max_length = max(len(category) for category, _ in sorted_categories[:k])

        for category, value in sorted_categories[:k]:
            print(f"{category.ljust(max_length)} {value:.3%}")

    return sorted_categories[:k]


def get_topics(
    article_name: str,
    k: int = 5,
    l: int = 10,
    data_path: str = "data/articles_plain_text/",
):
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


def generate_seegull_topics(countries: list[str] = ["British", "English"], k: int = 10):
    """
    Extracts the attributes of the specified countries from the SeeGull dataset and
    generates a list of topics of these attributes.
    :param countries: list of countries to extract attributes from
    :param k: number of topics to extract
    """
    # import SeeGull dataset
    df_seegull = pd.read_csv(seegull_file_path)

    # extract attributes related to the specified countries
    attributes_uk = (
        df_seegull[np.isin(df_seegull["identity"], countries)]["attribute"]
        .to_numpy()
        .tolist()
    )

    # remove all stopwords (at, in, with, the, ...)
    new = []
    english_stopwords = stopwords.words("english")
    for a in attributes_uk:
        new.extend([word for word in a.split(" ") if word not in english_stopwords])
    joined_attributes = " ".join(new)
    print("attributes of UK:\n ", joined_attributes)

    # find main topics
    print("\nmain topics:")
    top_k = top_k_categories(joined_attributes, k, verbose=True)
    return [topic for topic, percentage in top_k]


def generate_articles_topics():
    """
    Extract the topics from each article in the Wikispeedia dataset and store them in a
    csv file.
    """

    # get all articles (corpus)
    # only analyzing lim articles, do for all at term
    all_articles = PlaintextCorpusReader(
        articles_plain_text_path, ".*.txt", encoding="utf8"
    ).fileids()

    # get their topics and store in dataframe
    df_article_topics = pd.DataFrame(columns=["article_name", "topics", "confidence"])
    number_topic_to_keep = 10
    for article_name in all_articles:
        topics_value = pd.DataFrame(
            get_topics(article_name, data_path=articles_plain_text_path),
            columns=["topics", "value"],
        )

        new_row = {
            "article_name": article_name.removesuffix(".txt"),
            "topics": topics_value["topics"].to_numpy()[:number_topic_to_keep],
            "confidence": topics_value["value"].to_numpy()[:number_topic_to_keep],
        }

        df_article_topics.loc[df_article_topics.shape[0]] = new_row

    df_article_topics.to_csv(article_topics_file_path, index=False)


def load_articles_topics():
    return pd.read_csv(
        article_topics_file_path,
        usecols=["article_name", "topics"],
        converters={
            "topics": (
                lambda s: literal_eval("[" + (",".join(s[1:-1].split(" "))) + "]")
            )
        },
    )


def generate_token_pos_tags():
    # Load spaCy model and NLTK stop words
    nlp = spacy.load("en_core_web_sm")
    stop_words = set(stopwords.words("english"))

    def preprocess_text(text):
        # Process the text using spaCy
        doc = nlp(text)

        # Extract tokens and part-of-speech tags
        tokens_pos = [(token.text, token.pos_) for token in doc]

        # Remove stop words and non-alphabetic words
        filtered_tokens_pos = [
            (word, "NOUN" if word.istitle() and pos != "PROPN" else pos)
            for word, pos in tokens_pos
            if word.isalpha() and word.lower() not in stop_words
        ]

        return filtered_tokens_pos

    # List to store preprocessed documents
    docs = []
    all_articles = PlaintextCorpusReader(
        articles_plain_text_path, ".*.txt", encoding="utf8"
    ).fileids()
    for article_name in all_articles:
        with open(
            os.path.join(article_topics_file_path, article_name), "r", encoding="utf-8"
        ) as file:
            text = file.read()

        # Preprocess the text
        preprocessed_tokens_pos = preprocess_text(text)

        # Append preprocessed tokens to the docs list
        docs.append(preprocessed_tokens_pos)

    # Define the CSV file path
    csv_file_path = os.path.join(data_path, "article_token_pos.csv")

    # Write the preprocessed data to the CSV file
    with open(csv_file_path, mode="w", encoding="utf-8", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)

        # Write header
        csv_writer.writerow(["article_name", "tokens_pos"])

        # Write data
        for i, article_tokens_pos in enumerate(docs):
            csv_writer.writerow([all_articles[i], article_tokens_pos])


def load_token_pos():
    return pd.read_csv(
        os.path.join(data_path, "article_token_pos.csv"),
        converters={"tokens_pos": lambda s: literal_eval(s)},
    )


def generate_links_commmon(
    links: pd.DataFrame, reference_article: str = "United_Kingdom"
):
    """
    create dataframe that contains the links in common between the reference article and an article
    :param reference_article:
    """
    df_links_common = pd.DataFrame(
        columns=["reference_article", "article", "common_articles"]
    )

    def create_set(link: str, df: pd.DataFrame):
        return set(df[df["linkSource"] == link]["linkTarget"].to_numpy())

    reference_set = create_set(reference_article, links)

    for article_name in links["linkSource"].unique():
        if article_name == reference_article:
            continue
        comparison_set = create_set(article_name, links)
        new_row = {
            "reference_article": reference_article,
            "article": article_name,
            "common_articles": list(reference_set & comparison_set),
        }
        df_links_common.loc[df_links_common.shape[0]] = new_row

    df_links_common.to_csv(
        os.path.join(data_path, "links_common.csv"), encoding="utf-8"
    )


def load_links_common():
    return pd.read_csv(
        os.path.join(data_path, "links_common.csv"),
        converters={"common_links": literal_eval},
    )
