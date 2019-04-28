from sklearn.cluster import AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import pickle

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

    # AgglomerativeClustering
    ag = AgglomerativeClustering(n_clusters=10, affinity='cosine', linkage='average')
    ag.fit(data)

    labels = ag.labels_
    print(labels)
    pca = PCA(n_components=3)
    newX = pca.fit_transform(data)
    vis_x = newX[:, 0]
    vis_y = newX[:, 1]
    vis_z = newX[:, 2]
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    for i in range(len(labels)):
        colors.append(labels[i])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(vis_x, vis_y, vis_z, c=[colors[labels[i]] for i in range(len(labels))], marker='o')
    plt.show()
