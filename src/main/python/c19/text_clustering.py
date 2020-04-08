#!/usr/bin/env python3
from typing import Any
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE



def clusterise_sentences(sentences: Any, number_of_clusters: int):

    vectors = [sentence.vector for sentence in sentences]

    kmean_model = KMeans(n_clusters=number_of_clusters).fit(vectors)

    clusters = kmean_model.labels_
    barycentres = kmean_model.cluster_centers_



    print(clusters)
    print(len(barycentres))

    tsne = TSNE(random_state=42, perplexity=30, n_components=2)
    vectors_2d = tsne.fit_transform(vectors)


    cmap = plt.get_cmap('tab20')
    plt.figure(figsize=(15, 15))
    plt.scatter(vectors_2d[:, 0],
                vectors_2d[:, 1],
                color=[cmap(x / 15) for x in clusters],
                s=10,
                alpha=0.5)
    plt.scatter(barycentres[:, 0],
                barycentres[:, 1],
                color="red",
                s=10,
                alpha=0.9)
    plt.show()


if __name__ == "__main__":

    import joblib
    from c19.query_matching import Sentence

    closest = joblib.load(
        "/home/dynomante/projects/covid-19-kaggle/local_exec/example_output.pkl"
    )
    # Should be like {cluster_1: [s1, s2, s3]}
    clusters = clusterise_sentences(sentences=closest, number_of_clusters=5)
