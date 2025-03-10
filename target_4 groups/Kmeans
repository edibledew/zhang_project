#new 4 groups

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

file_path = "target.csv"  
df = pd.read_csv(file_path)

data = df["Tot_Gas_Yield_umol_gcat_h"].values.reshape(-1, 1)

kmeans = KMeans(n_clusters=4, random_state=0, n_init=10)
df["KMeans_Bin"] = kmeans.fit_predict(data)

cluster_means = df.groupby("KMeans_Bin")["Tot_Gas_Yield_umol_gcat_h"].mean().sort_values()
cluster_mapping = {old: new for new, old in enumerate(cluster_means.index)}
df["KMeans_Bin"] = df["KMeans_Bin"].map(cluster_mapping)

plt.figure(figsize=(10, 6))
sns.histplot(df["Tot_Gas_Yield_umol_gcat_h"], bins=20, kde=True, edgecolor='black', alpha=0.6)

for center in kmeans.cluster_centers_:
    plt.axvline(center, color='red', linestyle='dashed', label='Cluster Center' if center[0] == kmeans.cluster_centers_[0] else "")

plt.xlabel("Tot_Gas_Yield_umol_gcat_h")
plt.ylabel("Frequency")
plt.title("K-Means Clustered Binning")
plt.legend()
plt.grid(axis="y", alpha=0.5)

plt.show()

df.to_csv("target_with_kmeans_bins.csv", index=False)

df.head()

df.groupby("KMeans_Bin")["Tot_Gas_Yield_umol_gcat_h"].agg(["min", "max"]).sort_values("min")
