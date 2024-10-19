import pandas as pd

# Load the dataset
file_path = '/mnt/data/customer_segmentation.csv'
data = pd.read_csv(file_path)

# Display basic information and the first few rows to understand the structure
data.info(), data.head()


# Handling missing values by filling them with the median value of the 'Income' column
data['Income'].fillna(data['Income'].median(), inplace=True)

# Convert 'Dt_Customer' to datetime format for potential future use
data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'], format='%d-%m-%Y')

# Basic statistics of numerical columns for EDA
numeric_summary = data.describe()

# Checking unique values in categorical columns
categorical_summary = {
    'Education': data['Education'].unique(),
    'Marital_Status': data['Marital_Status'].unique()
}

numeric_summary, categorical_summary


import matplotlib.pyplot as plt
import seaborn as sns

# Visualizing distributions of a few important numerical features
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

sns.histplot(data['Income'], kde=True, ax=axes[0, 0], color='blue').set_title('Income Distribution')
sns.histplot(data['MntWines'], kde=True, ax=axes[0, 1], color='green').set_title('Wine Expenditure Distribution')
sns.histplot(data['MntMeatProducts'], kde=True, ax=axes[1, 0], color='purple').set_title('Meat Expenditure Distribution')
sns.histplot(data['Recency'], kde=True, ax=axes[1, 1], color='orange').set_title('Recency Distribution')

plt.tight_layout()
plt.show()

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

# Features to be used for clustering (excluding ID, Response, and Z_Revenue/CustContact which are constant)
features = ['Year_Birth', 'Income', 'Kidhome', 'Teenhome', 'Recency', 'MntWines', 
            'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 
            'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 
            'NumStorePurchases', 'NumWebVisitsMonth']

categorical_features = ['Education', 'Marital_Status']

# OneHotEncoder for categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), features),
        ('cat', OneHotEncoder(), categorical_features)])

# Applying transformations
X = preprocessor.fit_transform(data)

# Splitting the data into training and testing sets
X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)

X_train.shape, X_test.shape


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Clustering with KMeans using Euclidean distance (default)
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_train)

# Predicting on test set
y_pred_train = kmeans.predict(X_train)
y_pred_test = kmeans.predict(X_test)

# Plotting inertia (elbow method) to find the optimal number of clusters
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_train)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia)
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Number of Clusters (Euclidean)')
plt.show()


from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics.pairwise import manhattan_distances
import numpy as np

# Custom KMeans implementation using Manhattan distance
class KMeansManhattan:
    def __init__(self, n_clusters=5, max_iter=300, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
    
    def fit(self, X):
        np.random.seed(self.random_state)
        initial_idx = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.centroids = X[initial_idx]

        for _ in range(self.max_iter):
            self.labels_, _ = pairwise_distances_argmin_min(X, self.centroids, metric='manhattan')
            new_centroids = np.array([X[self.labels_ == i].mean(axis=0) for i in range(self.n_clusters)])

            if np.all(self.centroids == new_centroids):
                break
            self.centroids = new_centroids
    
    def predict(self, X):
        labels, _ = pairwise_distances_argmin_min(X, self.centroids, metric='manhattan')
        return labels

# Fit the model
kmeans_manhattan = KMeansManhattan(n_clusters=5, random_state=42)
kmeans_manhattan.fit(X_train)

# Predicting on test set
y_pred_train_manhattan = kmeans_manhattan.predict(X_train)
y_pred_test_manhattan = kmeans_manhattan.predict(X_test)


# Comparing clustering accuracy visually between Euclidean and Manhattan distance
from sklearn.decomposition import PCA

# Reducing data dimensions for 2D plot
pca = PCA(n_components=2)
X_train_2d = pca.fit_transform(X_train)

plt.figure(figsize=(12, 6))

# Euclidean distance plot
plt.subplot(1, 2, 1)
plt.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_pred_train, cmap='viridis')
plt.title('KMeans with Euclidean Distance')

# Manhattan distance plot
plt.subplot(1, 2, 2)
plt.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_pred_train_manhattan, cmap='plasma')
plt.title('KMeans with Manhattan Distance')

plt.tight_layout()
plt.show()



from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

# Applying hierarchical clustering (Ward method for linkage)
Z = linkage(X_train, method='ward')

# Plotting the dendrogram
plt.figure(figsize=(12, 8))
dendrogram(Z, truncate_mode='level', p=5)
plt.title('Dendrogram for Hierarchical Clustering')
plt.xlabel('Data Points')
plt.ylabel('Distance')
plt.show()


# Agglomerative Clustering (Ward linkage method)
agg_cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_agg_train = agg_cluster.fit_predict(X_train)
y_agg_test = agg_cluster.fit_predict(X_test)

# Visualizing clusters using PCA (for 2D plotting)
X_train_2d = pca.fit_transform(X_train)

plt.figure(figsize=(6, 6))
plt.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_agg_train, cmap='rainbow')
plt.title('Agglomerative Clustering (Hierarchical)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()



