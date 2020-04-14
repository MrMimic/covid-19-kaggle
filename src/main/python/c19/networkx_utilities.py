import networkx as nx
import pandas as pd
import urllib.request

def get_citations_graph(urls = None) -> nx.DiGraph:
    """
    Return the dataset citations graph 
    """
    print("loading citation graph... ")
    file_path = "title_citation_df"
    all_files = []
    if urls is None: urls = ["https://github.com/MrMimic/covid-19-kaggle/raw/master/resources/title_citation_part1.zip", "https://github.com/MrMimic/covid-19-kaggle/raw/master/resources/title_citation_part2.zip"]
    for i, url in enumerate(urls):
        filename = "title_citation_df_" + str(i)
        urllib.request.urlretrieve(url, filename)
        all_files.append(filename)
    dataframe = pd.concat((pd.read_csv(f, compression='zip')[['title', 'citation']] for f in all_files))
    # dataframe = pd.read_csv(file_path, compression='zip')[['title', 'citation']]
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