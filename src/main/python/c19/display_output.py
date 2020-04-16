#!/usr/bin/env python3

import pandas as pd
from IPython.core.display import HTML, display


def create_html_report(query: str,
                       closest_sentences_df: pd.DataFrame,
                       top_x: int = 3) -> None:
    """
    Print an HTML output in a jupyter kernel.

    Args:
        query (str): The user query.
        closest_sentences_df (pd.DataFrame): The output DF.
        top_x (int, optional): The number of sentences to display for each cluster.
        Defaults to 3.
    """
    number_of_kept_sentences = closest_sentences_df.shape[0]
    number_of_unique_papers = closest_sentences_df.paper_doi.unique().size
    number_of_clusters = closest_sentences_df.cluster.unique().size

    display(HTML(f"<h3>Generalities</h3>"))
    display(HTML(f"<p><b>User query:</b></p>&emsp;{query}"))
    display(
        HTML(
            f"<p><b>Number of closest sentences regarding the distance with the query:</b></p>&emsp;{number_of_kept_sentences}"
        ))
    display(
        HTML(
            f"<p><b>Unique papers found in the top-close sentences:</b></p>&emsp;{number_of_unique_papers}"
        ))
    display(
        HTML(
            f"<p><b>Clusters designed with the top-close sentences:</b></p>&emsp;{number_of_clusters}"
        ))

    for cluster in sorted(closest_sentences_df.cluster.unique().tolist()):

        sub_df = closest_sentences_df[closest_sentences_df["cluster"] ==
                                      cluster].sort_values(by="distance",
                                                           ascending=False)
        display(
            HTML(
                f"<h3>Cluster {cluster} top {top_x} sentences ({sub_df.shape[0]} total):</h3>"
            ))

        for index, row in sub_df.head(top_x).iterrows():
            display(
                HTML(
                    f"&emsp;{row.raw_sentence} (<a href=https://www.doi.org/{row.paper_doi} target='_blank'>{row.paper_doi}</a>)"
                ))


def create_md_report(query: str,
                     closest_sentences_df: pd.DataFrame,
                     output_report_path: str,
                     top_x: int = 3) -> None:
    """
    Generates a markdown report.

    Args:
        query (str): The user query.
        closest_sentences_df (pd.DataFrame): Output DF of the clustering step.
        output_report_path (str): Markdown file path.
        top_x (int, optional): Top X sentences per cluster. Defaults to 3.
    """
    number_of_kept_sentences = closest_sentences_df.shape[0]
    number_of_unique_papers = closest_sentences_df.paper_doi.unique().size
    number_of_clusters = closest_sentences_df.cluster.unique().size

    with open(output_report_path, "a") as handler:
        handler.write(f"Query: {query}\n")
        handler.write(
            f"**Number of sentences kept by distance filtering**: {number_of_kept_sentences}\n"
        )
        handler.write(
            f"**Number of unique papers found among these sentences**: {number_of_unique_papers}\n"
        )
        handler.write(
            f"**Number of clusters automatically designed**: {number_of_clusters}\n\n"
        )

        for cluster in sorted(closest_sentences_df.cluster.unique().tolist()):
            handler.write(f"**Cluster {cluster}:**\n")
            sub_df = closest_sentences_df[closest_sentences_df["cluster"] ==
                                          cluster].sort_values(by="distance",
                                                               ascending=False)
            for index, row in sub_df.head(top_x).iterrows():
                handler.write(f"\t- {row.raw_sentence}\n")
            handler.write("\n")
        handler.write(f"{'-' * 100}\n\n")
