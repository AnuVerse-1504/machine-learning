# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Sample dataset
data = {
    'Age': [22, 25, 47, 52, 46, 56, 23, 27],
    'Income': [15000, 29000, 48000, 52000, 50000, 60000, 16000, 30000]
}

df = pd.DataFrame(data)

# K-Means model
kmeans = KMeans(n_clusters=2)

# Fit model
kmeans.fit(df)

# Assign clusters
df['Cluster'] = kmeans.labels_

print(df)

# Plot clusters
plt.scatter(df['Age'], df['Income'], c=df['Cluster'])
plt.xlabel("Age")
plt.ylabel("Income")
plt.title("K-Means Clustering")
plt.show()