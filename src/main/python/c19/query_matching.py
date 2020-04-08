#!/usr/bin/env python3

import json
import multiprocessing as mp
import os
import sqlite3
import time
from operator import itemgetter
from typing import Any, List, Tuple
from datetime import datetime
import numpy as np
import tqdm
from sklearn.metrics.pairwise import cosine_similarity

from c19.database_utilities import get_sentences
from c19.embedding import Embedding
from c19.text_preprocessing import preprocess_text

from dateutil import parser


class Sentence:
    """
    Proper object to hold data about top-k related sentences.
    """
    def __init__(self, sentence: List[Any], article: List[Any]) -> None:
        # DOI of the article containing this sentence
        self.doi: str = sentence[0]
        # Section of the article  containing this sentence
        self.section: str = sentence[1]
        # The raw sentence as a str
        self.sentence: str = sentence[2]
        # The pre-processed sentence as a list of tokens
        self.preprocessed_sentence: List[str] = sentence[3]
        # The corresponding vector
        self.vector: List[float] = sentence[4]
        # Distance between this sentence and the query
        self.distance: float = sentence[5]
        # Publish date
        self.date: datetime = parser.parse(article[1])

        # Do we need it?

        # Body of the article containing this sentence
        # self.body: str = article[2]
        # Abstract of the article containing this sentence
        # self.abstract: str = article[3]
        # Tiel  of the article containing this sentence
        self.title: str = article[4]
        # SHA signature of the JSON file
        self.sha: str = article[5]
        # Source of the publication
        self.licence = article[6]


def vectorize_query(embedding_model: Embedding, query: str) -> List[float]:
    """
    Pre-process and then vectorize the user query.

    Args:
        embedding_model (Embedding): The embedding to be used to vectorise the query.
        query (str): The query.

    Returns:
        List[float]: The vector representing the query.
    """
    pp_query, query_raw = preprocess_text(query,
                                          stem_words=False,
                                          remove_num=True)
    del query_raw
    query_vector = embedding_model.compute_sentence_vector(pp_query[0])
    return query_vector


def get_article(db_path: str, paper_doi: str) -> List[Any]:
    """
    Retrieve an article by its paper_doi.

    Args:
        db_path (str): Path to the SQLite file.
        paper_doi ([type]): The DOI of the paper to be retrieved.

    Returns:
        List[Any]: The columns of the matching article.
    """
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    command = "SELECT * FROM articles WHERE paper_doi='%s'" % paper_doi
    cursor.execute(command)
    data = cursor.fetchone()
    cursor.close()
    connection.close()

    return data


def get_sentences_data(
        db_path: str = "articles_database.sqlite") -> List[List[float]]:
    """
    Get sentences data to be matched with queries (stay in RAM, thus computed once).

    Args:
        db_path (str, optional): Path to the DB. Defaults to "articles_database.sqlite".

    Returns:
        [List[List[float]]]: All found sentences data.
    """
    tic = time.time()
    loaded_sentences = []
    sentences = get_sentences(db_path)
    for sentence in sentences:
        if sentence[4] is not None:  # If vector
            sentence = list(sentence)
            sentence[4] = [float(x) for x in json.loads(sentence[4])]
            if np.nansum(
                    sentence[4]) != 0:  # If at least one word has been embeded
                loaded_sentences.append(sentence)

    toc = time.time()
    print(
        f"Queries will be match versus {len(sentences)} sentences ({round((toc-tic) / 60, 2)} minutes to load)."
    )
    return loaded_sentences


def get_k_closest_sentences(query: str,
                            db_path: str,
                            sentences: List[Any],
                            embedding_model: Embedding,
                            k: int = 10) -> Any:
    """
    Compute the cosine distance between the query and all sentences found in the DB.

    Args:
        query (str): The query as a sentence.
        sentences ([List[Any]): Pre-processed sentences data from the DB.
        embedding_model (Embedding): The embedding model to be used to vectorise the query.

    Returns:
        Any: A list of tuples [(score, target_vector_1), (score, target_vector_2), ...]
    """
    tic = time.time()

    # Vectorize it and format as arguments to be mapped by mp.Pool
    query_vector = list(vectorize_query(embedding_model, query))
    # Cosine sim of 1 means same vector, -1 opposite
    distances = cosine_similarity([query_vector],
                                  [sentence[4] for sentence in sentences])[0]
    for sentence, distance in zip(sentences, distances):
        sentence.append(distance)
    del distances
    # Now, index 5 contains distance and instance Sentence objects
    sentences = sorted(sentences, key=itemgetter(5), reverse=True)[0:k]
    # May be long
    sentences = [
        Sentence(sentence, get_article(db_path, sentence[0]))
        for sentence in sentences
    ]

    toc = time.time()
    print(f"Took {round((toc-tic) / 60, 2)} minutes to process the query.")

    return sentences
