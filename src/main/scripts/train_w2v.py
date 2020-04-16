#!/usr/bin/env python3

from c19 import word2vec_utilities, parameters, database_utilities, text_preprocessing
import os
"""
The trick is to train a DB of articles without sentence embedding.
It will insert Null instead of the vector.
Then, all pre-processed sentences are used to train W2V.
"""


def main():

    params = parameters.Parameters(database=parameters.Database(
        local_path="articles_database_v8_07042020.sqlite",
        kaggle_data_path="kaggle_data"))

    # Load all articles (title, abstract and body) into the 'article' table.
    database_utilities.create_db_and_load_articles(
        db_path=params.database.local_path,
        kaggle_data_path=params.database.kaggle_data_path,
        first_launch=params.first_launch)

    # Pre-process all sentences (no embedding)
    text_preprocessing.pre_process_and_vectorize_texts(
        embedding_model=None,
        db_path=params.database.local_path,
        first_launch=params.first_launch,
        stem_words=params.preprocessing.stem_words,
        remove_num=params.preprocessing.remove_numeric)

    # Param have been set up with: https://www.aclweb.org/anthology/W16-2922.pdf
    w2v_params = {
        "sg": 1,
        "hs": 1,
        "sample": 1e-5,
        "negative": 10,
        "min_count": 20,
        "size": 100,
        "window": 7,
        "seed": 42,
        "workers": os.cpu_count(),
        "iter": 10
    }

    # Train and save W2V and TFIDF as a parquet file DF.parquet
    word2vec = word2vec_utilities.W2V(params.database.local_path,
                                      tfidf_path="TFIDF.pkl",
                                      w2v_path="W2V.bin",
                                      w2v_params=w2v_params,
                                      parquet_output_path="DF.parquet")
    word2vec.train()


if __name__ == "__main__":
    main()
