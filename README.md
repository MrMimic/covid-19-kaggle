# Kaggle: COVID-19 Challenge

**This library provides tools aiming to find different opinions in the scientific litterature regarding the user query.**

The Kaggle notebook can be find [here](https://www.kaggle.com/mrmimic/risk-factors-analysis-by-opinion-finding).

**Birielfy**:

- It loads all articles into an SQLite DB.
- Sentences are pre-processed.
- Word2vec and TF-IDF are trained.
- Sentences are vectorised.
- The query is pre-processed and vectorised.
- The distance between query and sentences is computed.
- The top-k sentences are kept.
- A clustering is applied on these sentences.
- A ranking regarding its proximity to the centroid and authors of the papers.

## Installation

Simply use:

    pip install -q git+https://github.com/MrMimic/covid-19-kaggle

An then the library can be imported with:

    from c19 import parameters, database_utilities, text_preprocessing, embedding, query_matching, clusterise_sentences, plot_clusters, display_output

## Usage

### Create the database

Please use [this script](https://github.com/MrMimic/covid-19-kaggle/blob/master/src/main/scripts/create_db.py) to create the local database.

### Query the DB

Please use [this one](https://github.com/MrMimic/covid-19-kaggle/blob/master/src/main/scripts/query_db.py) to query the trained DB.

### Re-train the W2V and TF-IDF

[This script](https://github.com/MrMimic/covid-19-kaggle/blob/master/src/main/scripts/train_w2v.py) allows to re-train the W2V and TF-IDF to re-generate the parquet file.

## Usage

All queries from the Kaggle challenge have been reformulated [here](https://github.com/MrMimic/covid-19-kaggle/blob/master/resources/queries.json). They have then been processed with the tool presented [here](https://github.com/MrMimic/covid-19-kaggle/blob/master/resources/executed_queries.md).

Results are visible [on Kaggle](https://www.kaggle.com/mrmimic/opinions-extraction-tool-chloroquine-case-study).
