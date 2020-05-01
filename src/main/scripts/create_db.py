#!/usr/bin/env python3
"""
The idea is to create an SQLite DB containing both articles and pre-processed sentences.
It will contain two tables 'articles' and 'sentences'.
"""


import sys
sys.path.append("/home/dynomante/projects/covid-19-kaggle/src/main/python/c19")

# Load c19 custom library
from c19 import (database_utilities, embedding,
                 parameters, text_preprocessing)


def main():

    params = parameters.Parameters(
        first_launch=True,
        database=parameters.Database(
            local_path=
            "/home/dynomante/projects/covid-19-kaggle/local_exec/articles_database_v13_01052020.sqlite",
            kaggle_data_path=
            "/home/dynomante/projects/covid-19-kaggle/local_exec/kaggle_data",
            only_newest=False,
            only_covid=False),
        preprocessing=parameters.PreProcessing(max_body_sentences=0, stem_words=False),
        # embedding=parameters.Embedding(
        #     local_path="resources/global_df_w2v_tfidf.parquet")
        )

    # Load all articles (title, abstract and body) into the 'article' table.
    database_utilities.create_db_and_load_articles(
        db_path=params.database.local_path,
        kaggle_data_path=params.database.kaggle_data_path,
        first_launch=params.first_launch,
        only_newest=params.database.only_newest,
        only_covid=params.database.only_covid,
        enable_data_cleaner=params.database.enable_data_cleaner)

    # Load pre-trained word vectors
    # embedding_model = embedding.Embedding(
    #     parquet_embedding_path=params.embedding.local_path,
    #     embeddings_dimension=params.embedding.dimension,
    #     sentence_embedding_method=params.embedding.word_aggregation_method,
    #     weight_vectors=params.embedding.weight_with_tfidf)
    embedding_model = None

    # Pre-process and vectorise all sentences
    text_preprocessing.pre_process_and_vectorize_texts(
        embedding_model=embedding_model,
        db_path=params.database.local_path,
        first_launch=params.first_launch,
        stem_words=params.preprocessing.stem_words,
        remove_num=params.preprocessing.remove_numeric,
        batch_size=params.preprocessing.batch_size,
        max_body_sentences=params.preprocessing.max_body_sentences)


if __name__ == "__main__":
    main()
