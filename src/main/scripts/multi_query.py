#!/usr/bin/env python3
"""
This script is an example on how to loop on multi queries to generate a MD report.
"""
import json
from copy import deepcopy
from time import time

from c19 import (clusterise_sentences, display_output, embedding, parameters,
                 query_matching)

queries_path = "resources/queries.json"
with open(queries_path) as json_file:
    queries_data = json.load(json_file)

params = parameters.Parameters(
    database=parameters.Database(
        local_path="local_exec/articles_database_v12_16042020.sqlite",
        kaggle_data_path="local_exec/kaggle_data"),
    embedding=parameters.Embedding(
        local_path="resources/global_df_w2v_tfidf.parquet"))

# Load pre-trained word vectors
embedding_model = embedding.Embedding(
    parquet_embedding_path=params.embedding.local_path,
    embeddings_dimension=params.embedding.dimension,
    sentence_embedding_method=params.embedding.word_aggregation_method,
    weight_vectors=params.embedding.weight_with_tfidf)

# Load sentences from SQLite
all_db_sentences_original = query_matching.get_sentences_data(
    db_path=params.database.local_path)

# Loop over re-formulated queries
for task, subtasks in queries_data.items():
    for subtask, queries in subtasks.items():
        for query in queries:
            if query != "":
                print(query)
                # Get K closest for each
                closest_sentences_df = query_matching.get_k_closest_sentences(
                    query=query,
                    all_sentences=all_db_sentences_original,
                    embedding_model=embedding_model,
                    minimal_number_of_sentences=params.query.
                    minimum_sentences_kept,
                    similarity_threshold=params.query.
                    cosine_similarity_threshold)
                # Clusterise them
                closest_sentences_df = clusterise_sentences.perform_kmean(
                    k_closest_sentences_df=closest_sentences_df,
                    number_of_clusters=params.query.number_of_clusters,
                    k_min=params.query.k_min,
                    k_max=params.query.k_max,
                    min_feature_per_cluster=params.query.
                    min_feature_per_cluster)
                # Add the result to the markdown report
                display_output.create_md_report(
                    query=query,
                    closest_sentences_df=closest_sentences_df,
                    top_x=3,
                    task=task,
                    subtask=subtask,
                    output_report_path="resources/executed_queries.md")
