#!/usr/bin/env python3

import random

import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from typing import Any, Dict


def add_br_every(sentence: str, n: int) -> str:
    """
    Add <BR \> tag after X words in a sentence.

    Args:
        sentence (str): The sentence as a str
        n (int): The number of words between two "<BR \>".

    Returns:
        The sentence with BR added.
    """
    i = n
    sentence_split = sentence.split()
    while i < len(sentence_split):
        sentence_split.insert(i, "<br />")
        i += (n + 1)
    return "<b>" + " ".join(sentence_split) + "</b>"


def load_trace_names(fig: Any) -> Dict[str, int]:
    """
    Small function to extract default traces names.
    It allows to rename the legends entries from colors to real clusters.

    Args:
        fig (Any): The plotly figure.

    Returns:
        [type]: The dict {cluster_old_name: cluster_id}
    """
    cluster_index = 0
    traces = {}
    for trace in fig.data:
        trace_old_name = trace.name.split(",")[0]
        if trace_old_name not in traces.keys():
            traces[trace_old_name] = cluster_index
            cluster_index += 1
    return traces


def scatter_plot(closest_sentences_df: pd.DataFrame, query: str) -> None:
    """
    Plot the closest sentences as an interactive scatter.

    Args:
        closest_sentences_df (pd.DataFrame): The closest sentences DF, output from clusterise_sentences()
        query (str): The user query for title.
    """
    # Reduce vectors dimensions
    pca = PCA(n_components=2)
    vector_reduced = pca.fit_transform(closest_sentences_df.vector.to_list())
    closest_sentences_df["x"] = [vector[0] for vector in vector_reduced]
    closest_sentences_df["y"] = [vector[1] for vector in vector_reduced]
    # Get a random color per cluster
    colors = {
        index: "%06x" % random.randint(0, 0xFFFFFF)
        for index in closest_sentences_df.cluster.unique()
    }
    # Split raw sentences and join them back with <BR /> for prettier output
    closest_sentences_df["sentence_split"] = [
        add_br_every(s, 7)
        for s in closest_sentences_df.raw_sentence.to_list()
    ]
    # Start drawing scatter
    fig = px.scatter(
        data_frame=closest_sentences_df,
        x="x",
        y="y",
        color=[colors[x] for x in closest_sentences_df.cluster.to_list()],
        size=[
            7 if is_closest is True else 1
            for is_closest in closest_sentences_df.is_closest.to_list()
        ],
        hover_name=[
            f"DOI: {doi}" for doi in closest_sentences_df.paper_doi.to_list()
        ],
        hover_data=["sentence_split"],
        labels={
            "sentence_split": "<b>Sentence</b>",
            "x": "X reduced by PCA",
            "y": "Y reduced by PCA",
            "color": "<b>Cluster</b>",
            "section": "<b>Section</b>"
        },
        title=f"Closest sentences clustering for query: <i>{query.lower()}</i>",
        symbol="section")
    # Generate dict to update legends entries
    traces = load_trace_names(fig)
    # Update both legend and hovers
    for trace in fig.data:
        section = trace.name.split(",")[1]
        trace_old_value = trace.name.split(",")[0]
        trace.name = f"C{traces[trace_old_value]}, {section}"
        trace.hovertemplate = "<b>%{hovertext}</b><br><br><b>Section</b>: "+section+"<br><br><b>Sentence</b>: %{customdata[0]}<extra></extra>"
    # Final tweaks
    fig.update_traces(showlegend=True)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.show()
