#!/usr/bin/env python3

import json
import time
import pandas as pd
from operator import itemgetter
from typing import Any, List

from dateutil import parser

import numpy as np
import tqdm
from c19.database_utilities import get_sentences
from c19.embedding import Embedding
from c19.text_preprocessing import preprocess_text
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.cluster import KMeans


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
                            all_sentences: List[Any],
                            embedding_model: Embedding,
                            number_threshold: int = 10,
                            distance_threshold: float = 0.8,
                            filtering_method: str = "distance") -> pd.DataFrame:
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
    sentences_vectors = [sentence[4] for sentence in all_sentences]

    # Cosine sim of 1 means same vector, -1 opposite
    distances = cosine_similarity([query_vector], sentences_vectors)[0]

    # Group sentences and vectors
    for sentence, distance in zip(all_sentences, distances):
        sentence.append(distance)
    del distances

    # Now, index 5 contains distance and instance Sentence objects
    k_sentences = sorted(all_sentences, key=itemgetter(5), reverse=True)

    # Now, create a DF output
    k_sentences_df = pd.DataFrame(k_sentences, columns = ["paper_doi", "section", "raw_sentence", "sentence", "vector", "distance"])
    del k_sentences

    # Filter sentences
    if filtering_method == "distance":
        k_sentences_df = k_sentences_df[
            k_sentences_df["distance"] > distance_threshold]
    elif filtering_method == "number":
        k_sentences_df = k_sentences_df.sort_values(
            by="distance", ascending=False)[0:number_threshold]
    else:
        raise Exception(f"Unknown filtering method: {filtering_method}")

    toc = time.time()
    print(f"Took {round((toc-tic) / 60, 2)} minutes to process the query.")

    return k_sentences_df


def clusterise_sentences(k_closest_sentences_df: pd.DataFrame,
                         number_of_clusters: int) -> pd.DataFrame:
    """
    Add a columns "cluster" and "is_closest" to the sentence dataframe.

    Args:
        k_closest_sentences_df (pd.DataFrame): The DF to be updated.
        number_of_clusters (int): Number of K in the Kmean.

    Returns:
        pd.DataFrame: Updated DF.
    """
    tic = time.time()

    # Clusterise vectors
    vectors = k_closest_sentences_df["vector"].tolist()
    kmean_model = KMeans(n_clusters=number_of_clusters).fit(vectors)
    # Label clusters
    k_closest_sentences_df["cluster"] = kmean_model.labels_
    # Compute closest from barycentres and store in a boolean column
    closest, _ = pairwise_distances_argmin_min(kmean_model.cluster_centers_,
                                               vectors)
    k_closest_sentences_df["is_closest"] = [
        True if vectors.index(vector) in closest else False
        for vector in vectors
    ]

    toc = time.time()
    print(f"Took {round((toc-tic), 2)} seconds to clusterise closest sentences.")

    return k_closest_sentences_df
