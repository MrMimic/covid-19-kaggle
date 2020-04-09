import networkx as nx
import pandas as pd
import urllib.request

def get_citations_graph(file_path: str =None , url = None) -> nx.DiGraph:
    """
    Return the dataset citations graph 

    Args:
        file_path (str): Path to local file to be written.
    """
    print("loading citation graph... ")
    if url is None: url = "https://github.com/MrMimic/covid-19-kaggle/raw/master/resources/title_citation.zip"
    if file_path is None: file_path = "title_citation_df"
    urllib.request.urlretrieve(url, file_path)
    dataframe = pd.read_csv(file_path, compression='zip')[['title', 'citation']]
    G = nx.from_pandas_edgelist(dataframe,source='title',target='citation',create_using=nx.DiGraph )
    print(f"Graph loaded is having {len(list(G.nodes))} nodes and {len(list(G.edges))} edges")
    return G

def get_paper_cited_K_times_graph(G , M = 500) -> nx.DiGraph:
    # Create network of "is cited" papers only 
    Gs = nx.DiGraph()
    for node in G.nodes():
        if G.in_degree[node] > M : 
            print(node)
            # We look for adjacent nodes
            for adj_node in G.in_edges(node): # create link for each paper point to current paper
                Gs.add_node(adj_node)
                Gs.add_node(node)
                Gs.add_edge(adj_node,node)
    return Gs