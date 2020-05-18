#!/usr/bin/env python3

import json
import time
from copy import deepcopy
from operator import itemgetter
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
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
                                          remove_num=True)
    del query_raw
    try:
        query_vector = embedding_model.compute_sentence_vector(pp_query[0])
    except IndexError:
        raise Exception(
            f"Please reformule the query '{query}', the pre-processed version is: {str(pp_query)} (not enough words)."
        )
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
                            minimal_number_of_sentences: int = 100,
                            similarity_threshold: float = 0.8,
                            return_logs_and_query_vector: bool = False) -> Union[pd.DataFrame, Optional[List[str]]]:
    """
    Compute the cosine distance between the query and all sentences found in the DB.

    Args:
        query (str): The query as a sentence.
        all_sentences ([List[Any]): Pre-processed sentences data from the DB.
        embedding_model (Embedding): The embedding model to be used to vectorise the query.
        minimal_number_of_sentences (int): The minimal number of sentence to keep when filtering by distance.
        The similarity_threshold will be decreased gently if needed until the
        sufficient amount of sentence is reached.
        similarity_threshold (float): The minimal cosine similarity between sentence and query to be kept.

    Returns:
        pd.DataFrame: Updated DF with new columns distance.
        Optional[List[str]]: Logs from the query.
    """
    tic = time.time()
    logs = []

    # Vectorize it and format as arguments to be mapped by mp.Pool
    query_vector = list(vectorize_query(embedding_model, query))
    sentences_vectors = [sentence[4] for sentence in all_sentences]

    # Cosine sim of 1 means same vector, -1 opposite
    similarities = cosine_similarity([query_vector], sentences_vectors)[0]

    # Now, estimate the threshold to get enough sentences to keep.
    base_similarity_threshold = similarity_threshold
    while True:
        similarities_to_keep = [
            sim for sim in similarities if sim >= similarity_threshold
        ]
        if len(similarities_to_keep) >= minimal_number_of_sentences:
            break
        else:
            similarity_threshold -= 0.01
    if base_similarity_threshold != similarity_threshold:
        log = f"Similarity threshold lowered from {base_similarity_threshold} to {round(similarity_threshold, 2)} due to minimal number of sentence constraint."
        logs.append(log)
        print(log)

    # Then, create a list of sentences to keep (deepcopy only after filtering)
    # So the original list of sentence remains untouched for further queries.
    # Let's add for each sentence the cosine similarity with the query.
    k_sentences = []
    for sentence, similarity in zip(all_sentences, similarities):
        if similarity >= similarity_threshold:
            sentence_copied = deepcopy(sentence)
            sentence_copied.append(similarity)
            k_sentences.append(sentence_copied)

    # Now, index 5 contains distance and instance Sentence objects.
    k_sentences = sorted(k_sentences, key=itemgetter(5), reverse=True)

    # Now, create a DF output.
    k_sentences = pd.DataFrame(k_sentences,
                               columns=[
                                   "paper_doi", "section", "raw_sentence",
                                   "sentence", "vector", "distance"
                               ])

    toc = time.time()

    log = f"Took {round((toc-tic) / 60, 2)} minutes to process the query ({k_sentences.shape[0]} sentences kept by distance filtering)."
    logs.append(log)
    print(log)
    if return_logs_and_query_vector is True:
        return k_sentences, logs, query_vector
    else:
        return k_sentences
