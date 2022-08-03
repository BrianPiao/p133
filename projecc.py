import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv("https://raw.githubusercontent.com/whitehatjr/c-129-project/main/Processed_data/star_with_gravity.csv")
x = df.iloc[:,[3,4]].values
wcss = []
for i in range(1,11):
  k = KMeans(n_clusters = i,init = "k-means++",random_state = 42)
  k.fit(x)
  wcss.append(k.inertia_)
plt.plot(range(1,11),wcss)
plt.title("elbow method")
plt.xlabel("number of clusters")
plt.show()