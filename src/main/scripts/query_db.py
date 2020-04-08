#!/usr/bin/env python3
"""
This will give an example to query the DB.
One query is processed in +/- 1min on 8 vCPUs.
It return a list of Sentence() object
"""

import sys
sys.path.append("src/main/python/c19")
import warnings


# Load c19 custom library
from c19 import embedding, query_matching, parameters

# Some dependencies warnings are ugly
warnings.filterwarnings("ignore")


def main(query):

    # Get parameters
    params = parameters.Parameters(database=parameters.Database(
        local_path="local_exec/articles_database_v8_07042020.sqlite",
        kaggle_data_path="local_exec/kaggle_data"))

    # Load pre-trained word vectors
    embedding_model = embedding.Embedding(
        parquet_embedding_path=params.embedding.local_path,
        embeddings_dimension=params.embedding.dimensions,
        sentence_embedding_method=params.embedding.word_aggregation_method,
        weight_vectors=params.embedding.weight_with_tfidf)

    # Get sentence data (including vector) from sentence table
    sentences = query_matching.get_sentences_data(
        db_path=params.database.local_path)

    # Find the K closest sentence to the query
    closest_sentences = query_matching.get_k_closest_sentences(
        db_path=params.database.local_path,
        query=query,
        sentences=sentences,
        embedding_model=embedding_model,
        k=params.query.top_k_sentences)


if __name__ == "__main__":
    query = "What do we know about Chloroquine to treat covid-19 induced by coronavirus?"
    main(query)
