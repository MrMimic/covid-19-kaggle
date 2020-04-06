#!/usr/bin/env python3
import json
import multiprocessing as mp
import os
import sqlite3
import time
from typing import Any, List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from c19.database_utilities import get_sentences
from c19.embedding import Embedding
from c19.text_preprocessing import preprocess_text


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
                                          remove_num=False)
    del query_raw
    query_vector = embedding_model.compute_sentence_vector(pp_query[0])
    return query_vector


def get_article(db_path: str, paper_doi) -> List[Any]:
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
    data = cursor.fetchall()
    cursor.close()
    connection.close()

    return data


def compute_cosine_distance(args: Any) -> float:
    """
    Compute cosine distance between two embeded sentences (query and article sentences eg).

    Args:
        args (Any): A tuple (vector_1, vector_2)

    Returns:
        float: The cosine distance and the sentence vector.
    """
    sentence_vector = args[1]
    query_vector = args[0]
    distance = 1 - cosine_similarity([query_vector], [sentence_vector])[0][0]

    return (distance, sentence_vector)


def query_db_for_sentence(db_path: str, vector: str) -> Any:
    """
    Get a full sentence (not pre-processed) from a vector for a better display.

    Args:
        db_path (str): Path to the SQLite DB.
        vector (str): The vector to be retrieved.

    Returns:
        [Any]: The columns values of the matching sentence.
    """
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    command = "SELECT * FROM sentences WHERE vector='%s'" % vector
    cursor.execute(command)
    data = cursor.fetchall()
    cursor.close()
    connection.close()

    data = list(set(data))

    if len(data) > 1:
        print(f"ERROR: two sentences with vector {vector} have been found.")
        data = None

    return data


def get_db_sentences_vectors(
        db_path: str = "articles_database.sqlite") -> List[List[float]]:
    """
    Get sentences vectors to be matched with queries (stay in RAM, thus computed once).

    Args:
        db_path (str, optional): Path to the DB. Defaults to "articles_database.sqlite".

    Returns:
        [List[List[float]]]: All found sentences vectors.
    """
    tic = time.time()
    sentences = get_sentences(db_path)
    sentences_vectors = [[float(x) for x in json.loads(sentence_vector[4])]
                         for sentence_vector in sentences]
    sentences_vectors = [
        vector for vector in sentences_vectors if np.nansum(vector) != 0
    ]
    toc = time.time()

    print(
        f"Queries will be matched versus {len(sentences_vectors)} vectors ({round((toc-tic) / 60, 2)} min to load)."
    )
    return sentences_vectors


def get_query_distances_and_vectors(
        query: str, sentences_vectors: List[List[float]],
        embedding_model: Embedding) -> List[Tuple[float, List[float]]]:
    """
    Compute the cosine distance between the query and all sentences found in the DB.

    Args:
        query (str): The query as a sentence.
        sentences_vectors ([List[List[float]]]): Pre-processed sentences vectors from the DB.
        embedding_model (Embedding): The embedding model to be used to vectorise the query.

    Returns:
        List[Tuple[float, List[float]]]: A list of tuples [(score, target_vector_1), (score, target_vector_2), ...]
    """
    tic = time.time()

    # Vectorize it and format as arguments to be mapped by mp.Pool
    query_vector = list(vectorize_query(embedding_model, query))
    mapping_arguments = [(query_vector, sentence_vector)
                         for sentence_vector in sentences_vectors]

    # Execute
    with mp.Pool(os.cpu_count()) as pool:
        distances_and_vectors = pool.map(compute_cosine_distance,
                                         mapping_arguments)
    del mapping_arguments

    toc = time.time()
    print(f"Took {round((toc-tic) / 60, 2)} min to process the query.")

    return distances_and_vectors


def get_k_closest_sentences(distances_and_vectors: List[Tuple[float,
                                                              List[float]]],
                            db_path: str = "articles_database.sqlite",
                            k: int = 10) -> List[Any]:
    """
    Sort the results from get_query_distances_and_vectors() to extract the top k sentences.

    Args:
        distances_and_vectors (List[Tuple[float,List[float]]]): The list of tuples
        [(score, target_vector_1), (score, target_vector_2), ...]
        k (int, optional): Number of top sentences to extract. Defaults to 10.

    Returns:
        List[Any]: [description]
    """
    # Get results
    distances = [item[0] for item in distances_and_vectors]
    vectors = [item[1] for item in distances_and_vectors]

    # Find  k closest
    closest_sentence_indexes = np.argpartition(np.array(distances), k)[:k]
    closest_vectors = [vectors[idx] for idx in closest_sentence_indexes]
    closest_vectors_str = [
        json.dumps([str(x) for x in vec]) for vec in closest_vectors
    ]

    # Retrieve closest sentences
    closest_sentences = [
        query_db_for_sentence(vector=vec_str, db_path=db_path)
        for vec_str in closest_vectors_str
    ]

    return closest_sentences
