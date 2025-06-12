from data_loader import load_data
from preprocess import select_features, scale_features
from kmeans_model import elbow_method, train_kmeans
from plots import silhouette, plot_clusters

df = load_data("data/Wholesalecustomersdata.csv")
X = select_features(df)
X_scaled = scale_features(X)

elbow_method(X_scaled)

# Suppose optimal k = 5
model, labels, centroids = train_kmeans(X_scaled, 5)
df['Cluster'] = labels

silhouette(X_scaled, labels)
plot_clusters(X_scaled, labels)
