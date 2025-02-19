import os

os.environ["LOKY_MAX_CPU_COUNT"] = "4"
import matplotlib.pyplot as plt
import spacy
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

x = [4, 5, 10, 4, 3, 11, 14, 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]
data = list(zip(x, y))


def first_clustering():
    plt.scatter(x, y)
    plt.show()


def second_clustering():
    # Definition: Die Inertia ist ein Maß für die Summe der quadrierten Abstände
    # zwischen den Datenpunkten und den Zentroiden ihrer jeweiligen Cluster.
    # Sie gibt also an, wie nah die Punkte innerhalb eines Clusters zueinander sind.
    inertias = []

    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)

    plt.plot(range(1, 11), inertias, marker='o')
    plt.title('Elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()


def third_clustering():
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(data)

    plt.scatter(x, y, c=kmeans.labels_)
    plt.show()


def fourth_clustering():
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(data)

    # Predictions und Evaluationsmetriken
    labels = kmeans.labels_
    inertia = kmeans.inertia_
    silhouette_avg = silhouette_score(data, labels)
    davies_bouldin = davies_bouldin_score(data, labels)

    print(f'Inertia: {inertia:.2f}')
    print(f'Silhouette Score: {silhouette_avg:.2f}')
    print(f'Davies-Bouldin Index: {davies_bouldin:.2f}')

    # Visualisierung der Clusterung
    plt.scatter(x, y, c=kmeans.labels_, cmap='viridis')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='X',
                label='Centroids')
    plt.title('K-Means Clustering with Evaluation Metrics')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.show()


def fifth_clustering():
    data, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=42)

    # KMeans Clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(data)

    # Predictions und Evaluationsmetriken
    labels = kmeans.labels_
    inertia = kmeans.inertia_
    silhouette_avg = silhouette_score(data, labels)
    davies_bouldin = davies_bouldin_score(data, labels)

    print(f'Inertia: {inertia:.2f}')
    print(f'Silhouette Score: {silhouette_avg:.2f}')
    print(f'Davies-Bouldin Index: {davies_bouldin:.2f}')

    # Visualisierung der Clusterung
    plt.figure(figsize=(10, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k', s=50)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='X',
                label='Centroids')
    plt.title(f'K-Means Clustering with DB Index: {davies_bouldin:.2f}', fontsize=14)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.grid()
    plt.show()


def sixth_clustering():
    # Lade das SpaCy-Modell
    nlp = spacy.load("de_core_news_lg")

    # Liste von Wörtern für das Clustering
    words = ["Hund", "Katze", "Maus", "Köln", "Berlin", "Schule", "Universität", "Haus", "Auto", "Fahrrad", "Fußball",
             "Basketball"]

    # Erhalte die Vektoren der Wörter
    word_vectors = np.array([nlp(word).vector for word in words])

    # Wende KMeans-Clustering an
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(word_vectors)

    # Cluster-Labels
    labels = kmeans.labels_

    # Ausgabe der Cluster-Zuordnungen
    word_clusters = {}
    for word, label in zip(words, labels):
        if label not in word_clusters:
            word_clusters[label] = []
        word_clusters[label].append(word)

    # Ausgabe der Cluster
    for cluster, items in word_clusters.items():
        print(f'Cluster {cluster}: {items}')

    # Optional: Visualisierung (nur bei 2D-Vektoren, hier daher als Beispiel mit PCA)
    from sklearn.decomposition import PCA

    # Reduziere die Dimensionen auf 2 für die Visualisierung
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(word_vectors)

    # Visualisierung der Cluster
    plt.figure(figsize=(10, 6))
    for cluster, items in word_clusters.items():
        cluster_vectors = reduced_vectors[labels == cluster]
        plt.scatter(cluster_vectors[:, 0], cluster_vectors[:, 1], label=f'Cluster {cluster}')
        for i, word in enumerate(items):
            plt.annotate(word, (cluster_vectors[i, 0], cluster_vectors[i, 1]))

    plt.title('Clustering von Wörtern mit spaCy-Vektoren')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.legend()
    plt.grid()
    plt.show()


def seventh_clustering():
    # Erzeuge Beispiel-Daten
    data, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=42)

    # DBSCAN Clustering
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    labels = dbscan.fit_predict(data)

    # Einsichten
    # Einige Punkte können als -1 klassifiziert werden, was Rauschen darstellt
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f'Anzahl der Cluster: {n_clusters}')
    print(f'Cluster-Labels: {set(labels)}')

    # Silhouette Score und Davies-Bouldin Index können nur berechnet werden, wenn mehr als 1 Cluster vorhanden ist
    if n_clusters > 1:
        silhouette_avg = silhouette_score(data, labels)
        davies_bouldin = davies_bouldin_score(data, labels)
        print(f'Silhouette Score: {silhouette_avg:.2f}')
        print(f'Davies-Bouldin Index: {davies_bouldin:.2f}')
    else:
        print("Nicht genügend Cluster für die Berechnung der Evaluationsmetriken.")

    # Visualisierung der Clusterung
    plt.figure(figsize=(10, 6))
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(float(i) / len(unique_labels)) for i in range(len(unique_labels))]

    for k, col in zip(unique_labels, colors):
        if k == -1:  # rauschpunkt
            col = 'k'  # Schwarz für Rauschen

        class_member_mask = (labels == k)
        xy = data[class_member_mask]
        plt.scatter(xy[:, 0], xy[:, 1], color=col, s=50, edgecolor='k', label=f'Cluster {k}')

    plt.title('DBSCAN Clustering')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.grid()
    plt.show()


def eight_clustering():
    # Lade das spaCy-Modell (großes Modell)
    nlp = spacy.load("de_core_news_lg")

    # Liste von Wörtern für das Clustering
    large_word_list = [
        "Hund", "Katze", "Maus", "Köln", "Berlin", "Schule", "Universität",
        "Haus", "Auto", "Fahrrad", "Fußball", "Basketball", "Liebe",
        "Freundschaft", "Natur", "Technologie"  # Füge hier viele weitere Wörter hinzu
    ]

    # Verwende nlp.pipe für die massenhafte Verarbeitung
    word_vectors = np.array([nlp(word).vector for word in large_word_list])

    # DBSCAN Clustering
    dbscan = DBSCAN(eps=0.5, min_samples=3)  # Passe die Parameter nach Bedarf an
    labels = dbscan.fit_predict(word_vectors)

    # Visualisierung der Ergebnisse
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(word_vectors)

    plt.figure(figsize=(10, 6))
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(float(i) / len(unique_labels)) for i in range(len(unique_labels))]

    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = 'k'  # Rauschen schwarz
            label_text = 'Rauschen'
        else:
            label_text = f'Cluster {k}'

        # Punkte im Cluster darstellen
        class_member_mask = (labels == k)
        xy = reduced_vectors[class_member_mask]
        plt.scatter(xy[:, 0], xy[:, 1], color=col, s=50, edgecolor='k', label=label_text)

    # Annotiere die Punkte
    for i, word in enumerate(large_word_list):
        plt.annotate(word, (reduced_vectors[i, 0], reduced_vectors[i, 1]), fontsize=9)

    plt.title('Clustering von Wörtern mit spaCy Vektoren')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()


def ninth_clustering():
    # Lade das spaCy-Modell
    nlp = spacy.load("de_core_news_lg")

    # Liste von Sätzen für das Clustering
    sentences = [
        "Der Hund läuft im Park.",
        "Die Katze schläft auf dem Tisch.",
        "Köln ist eine Stadt in Deutschland.",
        "Berlin hat viele Sehenswürdigkeiten.",
        "Ich liebe die Natur und ihre Schönheit.",
        "Technologie verändert die Welt.",
        "Fahrradfahren ist gesund und umweltfreundlich.",
        "Freundschaft ist wichtig im Leben.",
        "Der Fußballverein gewann das Spiel.",
        "Basketball ist ein spannender Sport.",
    ]

    # Erhalte die Vektoren der Sätze
    sentence_vectors = np.array([nlp(sentence).vector for sentence in sentences])

    # DBSCAN Clustering
    dbscan = DBSCAN(eps=0.5, min_samples=2)
    labels = dbscan.fit_predict(sentence_vectors)

    # Visualisierung der Ergebnisse
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(sentence_vectors)

    plt.figure(figsize=(10, 6))
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(float(i) / len(unique_labels)) for i in range(len(unique_labels))]

    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = 'k'  # Rauschen schwarz
        class_member_mask = (labels == k)
        xy = reduced_vectors[class_member_mask]
        plt.scatter(xy[:, 0], xy[:, 1], color=col, s=50, edgecolor='k', label=f'Cluster {k}')

    for i, sentence in enumerate(sentences):
        plt.annotate(sentence, (reduced_vectors[i, 0], reduced_vectors[i, 1]), fontsize=9)

    plt.title('Clustering von Sätzen mit spaCy Vektoren')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    # first_clustering()
    # second_clustering()
    # third_clustering()
    # fourth_clustering()
    # fifth_clustering()
    # sixth_clustering()
    # seventh_clustering()
    # eight_clustering()
    ninth_clustering()
