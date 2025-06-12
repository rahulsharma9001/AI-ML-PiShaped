from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

def silhouette(X, labels):
    score = silhouette_score(X, labels)
    print(f"Silhouette Score: {score:.2f}")
    return score

def plot_clusters(X, labels):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X)
    sns.scatterplot(x=reduced[:,0], y=reduced[:,1], hue=labels, palette='Set2')
    plt.title('KMeans Clusters (PCA-reduced)')
    plt.show()
