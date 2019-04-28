import pickle
import numpy as np
from sklearn.cluster import KMeans
import operator
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import random

if __name__ == "__main__":
    # Load data
    data = []
    print('reading data...')
    with open("feature", 'r') as f:
        content = f.read()
        data = pickle.loads(content)
    data = np.asarray(data)
    data = data
    data = data.reshape((data.shape[0], -1))
    print('finished reading data')

    # KMeans
    #labels = []
    #scores = []
    #l = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 1000]
    #for k in l:
    #    km = KMeans(n_clusters=k, random_state=0)
    #    km.fit(data)
    #    labels.append(km.labels_)
    #    scores.append(km.inertia_)
    #    print(km.inertia_)
    #index, value = min(enumerate(scores), key=operator.itemgetter(1))
    #print(scores)
    #plt.figure()
    #plt.plot(l, scores, '-ok')
    #plt.xlabel('Number of cluster')
    #plt.ylabel('SSE')
    #plt.show()

    km = KMeans(n_clusters=2, random_state=0)
    km.fit(data)
    labels = km.labels_
    pca = PCA(n_components=2)
    newX = pca.fit_transform(data)
    vis_x = newX[:, 0]
    vis_y = newX[:, 1]
    colors = ['r','b']
    plt.scatter(vis_x, vis_y, c=[colors[labels[i]] for i in range(len(labels))], marker=".", linestyle="None")
    plt.title('KMeans Clustering into 2 Clusters')
    plt.show()
