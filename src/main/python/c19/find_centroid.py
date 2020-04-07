#!/usr/bin/env python
# coding: utf-8

# In[ ]:




def nearest_to_centroid(cluster: List[List], k: int)-> List[List]:
    """ Return the k most nearest sentences from the centroid of a cluster
    
    
    Args:
        cluster (List[List]): List of vectors of the cluster sentences.
        K (int): Number of vectors to keep.
        
        
    Returns:
      nearest_vec  : [K nearest vector from the centroid of the cluster]
    
    """
    
    
    
    sum_element=[0]*len(cluster[0])
    for vec in cluster:
        sum_element=[sum(x) for x in zip(sum_element,vec)]
    centroid= [x/len(cluster) for x in sum_element]
    dist={}
    index=0
    for vec in cluster:
        dist[index]= math.sqrt(sum([(a - b) ** 2 for a, b in zip(centroid, vec)]))
        index+=1
    dist = {k: v for k, v in sorted(dist.items(), key=lambda item: item[1])}
    nearest_vec=[cluster1[i] for i in list(dist.keys())[0:k]]    
    return nearest_vec


def average_anwers(clusters: Dict[str, List[List]], k: int)-> Dict[str, List[list]]:
    """ Return for each cluster the the k most nearest sentences from its centroid
    
    
    Args:
        clusters (Dict[str, List[List]]): dict of clusters.
        K (int): Number of vectors to keep.
        
        
    Returns:
        average_sentences: dict with cluster in key and list of the k nearest vector from the centroid as value
    
    """
    average_sentences={}
    for cl in clusters.keys():
        average_sentences[cl]= nearest_to_centroid(clusters[cl],k)
        
    return average_sentences 

