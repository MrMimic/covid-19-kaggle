#!/usr/bin/env python3
"""
This will give an example to query the DB.
Set chain_query to True will allow to query the model
in less than 0.1min but need 20Go of RAM at least.
It returns a list of Sentence() object.
"""

# Load c19 custom library
from c19 import embedding, query_matching, parameters, clusterise_sentences


def prepare_data(params):

    # Load pre-trained word vectors
    embedding_model = embedding.Embedding(
        parquet_embedding_path=params.embedding.local_path,
        embeddings_dimension=params.embedding.dimension,
        sentence_embedding_method=params.embedding.word_aggregation_method,
        weight_vectors=params.embedding.weight_with_tfidf)

    # Get sentence data (including vector) from sentence table
    all_db_sentences = query_matching.get_sentences_data(
        db_path=params.database.local_path)

    return embedding_model, all_db_sentences


if __name__ == "__main__":

    params = parameters.Parameters(
        database=parameters.Database(
            local_path="/home/dynomante/projects/covid-19-kaggle/local_exec/articles_database_v14_02052020_test.sqlite",
            kaggle_data_path="/home/dynomante/projects/covid-19-kaggle/local_exec/kaggle_data"),
        embedding=parameters.Embedding(
            local_path="/home/dynomante/projects/covid-19-kaggle/w2v_parquet_file_new_version.parquet"))

    embedding_model, all_db_sentences = prepare_data(params)

    query = "What do we know about Chloroquine to treat covid-19 induced by coronavirus?"

    closest_sentences_df = query_matching.get_k_closest_sentences(
        query=query,
        all_sentences=all_db_sentences,
        embedding_model=embedding_model,
        minimal_number_of_sentences=params.query.minimum_sentences_kept,
        similarity_threshold=params.query.cosine_similarity_threshold)

    closest_sentences_df = clusterise_sentences.perform_kmean(
        k_closest_sentences_df=closest_sentences_df,
        number_of_clusters=params.query.number_of_clusters,
        k_min=params.query.k_min,
        k_max=params.query.k_max,
        min_feature_per_cluster=params.query.min_feature_per_cluster)

    closest_sentences_df.to_csv("local_exec/output.csv")
