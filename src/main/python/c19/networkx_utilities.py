import urllib.request

import networkx as nx
import pandas as pd


def get_citations_graph(urls=None) -> nx.DiGraph:
    """
    Return the dataset citations graph.
    """
    print("Creating citation graph to generate pagerank... ")
    file_path = "title_citation_df"
    all_files = []
    if urls is None:
        urls = [
            "https://github.com/MrMimic/covid-19-kaggle/raw/master/resources/title_citation_part1.zip",
            "https://github.com/MrMimic/covid-19-kaggle/raw/master/resources/title_citation_part2.zip"
        ]
    for i, url in enumerate(urls):
        filename = "title_citation_df_" + str(i)
        urllib.request.urlretrieve(url, filename)
        all_files.append(filename)
    dataframe = pd.concat((pd.read_csv(f, compression='zip')[['title', 'citation']] for f in all_files))
    # dataframe = pd.read_csv(file_path, compression='zip')[['title', 'citation']]
    G = nx.from_pandas_edgelist(dataframe,source='title',target='citation',create_using=nx.DiGraph )
    print(f"Full citation graph loaded is having {len(list(G.nodes))} nodes and {len(list(G.edges))} edges")
    return G

def get_paper_cited_K_times_graph(G , M = 500) -> nx.DiGraph:
    """
    Return a network of paper cited at least M times
    """
    Gs = nx.DiGraph()
    for node in G.nodes():
        if G.in_degree[node] > M:
            # We look for adjacent nodes
            for adj_node in G.in_edges(
                    node):  # create link for each paper point to current paper
                Gs.add_node(adj_node)
                Gs.add_node(node)
                Gs.add_edge(adj_node,node)
    return Gs

def add_pagerank_to_metadata_df(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Return the dataframe with an extra column containing the pagerank of all papers in the dataframe subjecent network 
    """
    assert('title' in dataframe.columns)
    # copy original title before processing title lowercase
    dataframe['original_title'] = dataframe['title'].copy()
    dataframe['title'] = dataframe['title'].apply(lambda x : str(x).lower())
    # create a subgraph on the nodes given in the dataframe (keeping themselves and their neighbors)
    G = get_citations_graph()
    node_sublist = list(dataframe['title'].apply(lambda x : str(x).lower()))
    to_keep_nodes = []
    for title in node_sublist:
        to_keep_nodes.append(title)
        if title in G:
            for n in G.neighbors(title):
                to_keep_nodes.append(n)
    Gsub = G.subgraph(to_keep_nodes)
    print(f"Reduce citation graph loaded is having {len(list(Gsub.nodes))} nodes and {len(list(Gsub.edges))} edges")
    # compute pagerank on network
    pr = nx.pagerank(Gsub)
    pagerank = pd.DataFrame(pr.items(), columns=["title", "pagerank"]).sort_values(by="pagerank", ascending=False)
    merged_df = pd.merge(dataframe, pagerank, how='left', on=['title'])
    assert(len(merged_df), len(dataframe))
    # get the original title
    merged_df = merged_df.drop(['title'],axis=1)
    merged_df.rename(columns={"original_title": "title"}, inplace=True)
    print(f'Found {len(merged_df[closest_sentences_df["pagerank"].notna()])} pagerank for {len(merged_df)} articles')
    return merged_df
