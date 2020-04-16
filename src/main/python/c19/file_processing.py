#!/usr/bin/env python3

import json
from collections import OrderedDict
from typing import Any, Dict
import pandas as pd


def read_file(file_path: str) -> Dict[str, Any]:
    """
    Open JSON file and return a parsable dict() data.

    Args:
        file_path (str): Path to the file to read.

    Returns:
        Dict[str, Any]: Loaded JSON data
    """
    with open(file_path, "r") as handler:
        json_data = json.loads(handler.read(), object_pairs_hook=OrderedDict)
    return json_data


def read_parquet(file_path: str) -> pd.DataFrame:
    """
    Read a parquet file

    Args:
        file_path (str): Path to the file.

    Returns:
        pd.DataFrame: The loaded dataframe.
    """
    return pd.read_parquet(file_path)


def get_body(json_data: Dict[str, Any]) -> str:
    """
    Return only body keys from an article loaded as JSON data.

    Args:
        json_data (Dict[str, Any]): Loaded JSON data.

    Returns:
        str: The body text as a string.
    """
    return " ".join([
        json_data["body_text"][index]["text"].strip()
        for index in range(len(json_data["body_text"]))
    ])
