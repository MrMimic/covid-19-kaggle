#!/usr/bin/env python3
"""
The idea is to create an SQLite DB containing both articles and pre-processed sentences.
It will contain two tables 'articles' and 'sentences'.
"""

import sys
sys.path.append("src/main/python/c19")
import warnings

import nltk

# Load c19 custom library
from c19 import (binary_classification, database_utilities, embedding,
                 parameters, text_preprocessing)


# Some dependencies warnings are ugly
warnings.filterwarnings("ignore")

# We need stopwords and tokeniser
nltk.download('stopwords')
nltk.download('punkt')


def main():

    # Get parameters
    params = parameters.Parameters(first_launch=True)

    # Load all articles (title, abstract and body) into the 'article' table.
    database_utilities.create_db_and_load_articles(
        db_path=params.database.local_path,
        kaggle_data_path=params.database.kaggle_data_path,
        first_launch=params.first_launch,
        load_body=params.preprocessing.load_text_body)

    # Load pre-trained word vectors
    embedding_model = embedding.Embedding(
        parquet_embedding_path=params.embedding.local_path,
        embeddings_dimension=params.embedding.dimensions,
        sentence_embedding_method=params.embedding.word_aggregation_method,
        weight_vectors=params.embedding.weight_with_tfidf)

    # Pre-process and vectorise all sentences
    text_preprocessing.pre_process_and_vectorize_texts(
        embedding_model=embedding_model,
        db_path=params.database.local_path,
        first_launch=params.first_launch,
        stem_words=params.preprocessing.stem_words,
        remove_num=params.preprocessing.remove_numeric)


if __name__ == "__main__":
    main()
