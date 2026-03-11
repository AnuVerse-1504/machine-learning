import pandas as pd
from sklearn.cluster import KMeans

data = {
    'Income':[15000,20000,25000,80000,90000,85000],
    'Spending':[2000,3000,2500,10000,12000,11000]
}

df = pd.DataFrame(data)

model = KMeans(n_clusters=2)

model.fit(df)

df['Cluster'] = model.labels_

print(df)