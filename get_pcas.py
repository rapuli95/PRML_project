import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Read the files for each number and save it as elements to a  list
file_path = "parquets"
df_dict = [];
for file in range(0,10):
  df = pd.read_parquet(f'{file_path}/Num_{file}.parquet')
  grouped = df.groupby(level='sheet')
  df_dict.append(grouped)

# PCA, lets choose number for example 0 
study_df = df_dict[0] # 0 to 10
first_sample = study_df.get_group(25) # 
# apply PCA
X = first_sample.to_numpy()
# fit PCA
X_pca = PCA(n_components=2).fit_transform(X)
# plot the image
plt.figure()
plt.plot(X_pca[:,1], X_pca[:,0])
plt.axis("equal")
plt.show
# try to save every single sample as tuple to list
pcas_dict = {}
for df_num in range(0,10):
  study_df = df_dict[df_num]
  # key = str(df_num)
  pcas_dict[df_num] = {}
  for s in range(1,101):
    sample = study_df.get_group(s)
    X = sample.to_numpy()
    X_pca = PCA(n_components=2).fit_transform(X)
    pcas_dict[df_num][s] = pd.DataFrame({
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1]
    })
  
print(pcas_dict[0][10]) # PCA values for number 0 and sample 10
