#!/usr/bin/env python3

import multiprocessing as mp
import os
import sqlite3
import time
import tqdm
from pathlib import Path
from typing import Any, List, Tuple

import pandas as pd
from dateutil import parser

from c19.file_processing import get_body, read_file
from retry import retry


def instanciate_sql_db(db_path: str = "articles_database.sqlite") -> None:
    """
    Create the needed SQLite database.

    Args:
        db_path (str, optional): Path to the DB to be created. Defaults to "articles_database.sqlite".
    """
    if os.path.isfile(db_path):
        os.remove(db_path)
    database = sqlite3.connect(db_path)

    # Storing articles
    articles_table = {
        "paper_doi": "TEXT PRIMARY KEY",
        "date": "DATETIME",
        "body": "TEXT",
        "abstract": "TEXT",
        "title": "TEXT",
        "sha": "TEXT",
        "folder": "TEXT"
    }
    columns = [
        "{0} {1}".format(name, col_type)
        for name, col_type in articles_table.items()
    ]
    command = "CREATE TABLE IF NOT EXISTS articles ({});".format(
        ", ".join(columns))
    database.execute(command)

    # Storing sentences
    sentences_table = {
        "paper_doi": "TEXT",
        "section": "TEXT",
        "raw_sentence": "TEXT",
        "sentence": "TEXT",
        "vector": "TEXT"
    }
    columns = [
        "{0} {1}".format(name, col_type)
        for name, col_type in sentences_table.items()
    ]
    command = "CREATE TABLE IF NOT EXISTS sentences ({});".format(
        ", ".join(columns))
    database.execute(command)
    database.close()


def get_articles_to_insert(articles_df: pd.DataFrame) -> List[Any]:
    """
    Create a list of articles to be inserted.

    TODO: Add language_detection function here.

    Args:
        articles_df (pd.DataFrame): The metadata dataframe (sliced or not).

    Returns:
        List[Any]: A list of tuples (index, article).
    """
    articles = []
    for index, data in articles_df.iterrows():
        articles.append((index, data))
    return articles


@retry(sqlite3.OperationalError, tries=5, delay=2)
def insert_row(list_to_insert: List[Any],
               table_name: str = "articles",
               db_path: str = "articles_database.sqlite") -> None:
    """
    Insert row into the SQLite database. Retry 5 times if database is locked
    by concurring accesses.

    Args:
        list_to_insert (List[Any]): List of data matching either "articles" or "sentences" table columns.
        table_name (str, optional): The name of the table (articles of sentences). Defaults to "articles".
        db_path (str, optional): Path to the SQLite DB. Defaults to "articles_database.sqlite".

    Raises:
        Exception: Unknown table.
    """
    if table_name == "articles":
        command = "INSERT INTO articles(paper_doi, title, body, abstract, date, sha, folder) VALUES (?, ?, ?, ?, ?, ?, ?)"
    elif table_name == "sentences":
        command = "INSERT INTO sentences(paper_doi, section, raw_sentence, sentence, vector) VALUES (?, ?, ?, ?, ?)"
    else:
        raise Exception(f"Unknown table {table_name}")

    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    cursor.execute(command,
                   list_to_insert)  # This line will be retried if fails
    cursor.close()
    connection.commit()
    connection.close()


def insert_article(args: List[Tuple[int, pd.Series, str, str]]) -> None:
    """
    Parse and insert a single article into the SQLite DB. Parallelised method.
    args = [(index, df_line), db_path, data_path]

    Args:
        args (List[Tuple[int, pd.Series, str]]): Index, article and path to the DB.
    """
    data = args[0][1]
    db_path = args[1]
    kaggle_data_path = args[2]
    load_body = args[3]
    # Get body
    if data.has_pdf_parse is True and load_body is True:

        json_file = [
            file_path for file_path in Path(
                os.path.join(kaggle_data_path, data.full_text_file)).glob('**/*.json')
            if data.sha in str(file_path)
        ]

        try:
            json_data = read_file(json_file[0])
            body = get_body(json_data=json_data)
            folder = data.full_text_file
        except (FileNotFoundError, KeyError, IndexError):
            body = None
            folder = None
    else:
        body = None
        folder = None
    # Get date
    try:
        date = parser.parse(data.publish_time)
    except Exception:  # Better to get no date than a string of whatever
        date = None
    # Insert
    raw_data = [data.doi, data.title, body, data.abstract, date, data.sha, folder]
    insert_row(list_to_insert=raw_data, db_path=db_path)


def get_all_articles_doi(
        db_path: str = "articles_database.sqlite") -> List[str]:
    """
    Return all articles DOIs stored in the article table.

    Args:
        db_path (str, optional): Path to the SQLite DB. Defaults to "articles_database.sqlite".

    Returns:
        List[str]: List of found DOIs.
    """
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    cursor.execute("SELECT paper_doi FROM articles")
    ids = cursor.fetchall()
    cursor.close()
    connection.close()
    ids_cleaneds = [id_[0] for id_ in ids if len(id_) == 1 and id_[0] is not None]

    return ids_cleaneds


def create_db_and_load_articles(db_path: str = "articles_database.sqlite",
                                metadata_df_path: str = os.path.join(
                                    os.sep, "kaggle", "input",
                                    "CORD-19-research-challenge",
                                    "metadata.csv"),
                                kaggle_data_path: str = os.path.join(
                                    os.sep, "kaggle", "input",
                                    "CORD-19-research-challenge"),
                                first_launch: bool = False,
                                load_body: bool = False) -> None:
    """
    Main function to create the DB at first launch.
    Load metadata.csv, try to get body texts and insert everything without pre-processing.

    Args:
        db_path (str, optional): Path to the SQLite file. Defaults to "articles_database.sqlite".
        metadata_df_path (str, optional): Path to metadata DF. Defaults to os.path.join(os.sep, "content", "kaggle_data","metadata.csv").
        kaggle_data_path (str, optional): Path to the folder containing Kaggle JSON files.
        load_file (bool, optional): Debug option to prevent to create a new file. Defaults to True.
    """

    if first_launch is False:
        assert os.path.isfile(db_path)
        print(f"DB {db_path} will be used instead.")

    else:
        tic = time.time()

        # The metadata.csv file will be used to fetch available files
        metadata_df = pd.read_csv(metadata_df_path, low_memory=False)
        # The DOI isn't unique, then let's keep the last version of a duplicated paper
        metadata_df.drop_duplicates(subset=["doi"], keep="last", inplace=True)
        # Load usefull information to be stored: id, title, body, abstract, date, sha, folder
        articles_to_be_inserted = [
            (article, db_path, kaggle_data_path, load_body)
            for article in get_articles_to_insert(metadata_df)
        ]
        # Create a new SQLite DB file
        instanciate_sql_db(db_path=db_path)
        # Parallelize articles insertion
        with mp.Pool(os.cpu_count()) as pool:
            with tqdm.tqdm(total=len(articles_to_be_inserted)) as pbar:
                for i, _ in enumerate(
                        pool.imap_unordered(insert_article, articles_to_be_inserted)):
                    pbar.update()

        toc = time.time()
        print(
            f"Took {round((toc-tic) / 60, 2)} min to insert {len(articles_to_be_inserted)} articles (SQLite DB: {db_path})."
        )
        del articles_to_be_inserted


def get_sentences(db_path: str) -> List[Any]:
    """
    Retrieve all sentences from the DB.

    Args:
        db_path (str): Path to the DB.

    Returns:
        List[Any]: List of sentences data.
    """
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    command = "SELECT * FROM sentences"
    cursor.execute(command)
    data = cursor.fetchall()
    cursor.close()
    connection.close()

    return data
