# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 11:36:49 2023

@author: samuel veer singh
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import metrics
import seaborn as sns
from REMclust import REM
import pandas as pd


## ------------------------------------------------------------------------- ##
#   Example 2
## ------------------------------------------------------------------------- ##
df = pd.read_csv('Data/penguins_size.csv')

df['species'] = df['species'].map({'Adelie':0, 'Gentoo':1, 'Chinstrap':2})
df['island'] = df['island'].map({'Torgersen':0, 'Biscoe':1, 'Dream':2})
df['sex'] = df['sex'].map({'MALE':0, 'FEMALE':1})

df = df.dropna()

X = df[df.columns[2:-1]]    
X = np.array(X)

scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)
n_samples, n_features = X_scaled.shape


bndwk = int(np.floor(np.min((30, np.square(n_samples)))))
Cluster = REM(data=X_scaled, covariance_type = "full", criteria = "all", bandwidth = bndwk, tol = 1e-3)
Cluster.mode_decision_plot()
Cluster.fit(density_threshold = 1.0, distance_threshold = 1.0)

yp = Cluster.get_labels(mixture_selection='aic')

# plot the clustered dataset
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], s = 5, c = yp, marker = "o", cmap='tab10')
plt.xlabel('culmen length (mm)')
plt.ylabel('culmen depth (mm)')
plt.title('density threshold = 1.0' + '\n' + 'distance threshold = 1.0')
plt.show()


# AMI & ARI
nmi = metrics.normalized_mutual_info_score(df['species'].astype(int), yp.astype(int))
ari = metrics.adjusted_rand_score(df['species'].astype(int), yp.astype(int))
print("Adjusted Rand Score: \t", ari)
print("Normalized Mutual Information Score: \t", nmi)


# sensitivity w.r.t density threshold
n = 5
nmi_x = np.zeros((n))
ari_x = np.zeros((n))
dens_thres = [0.90,1.00,1.10,1.20,1.30]

for i in range(n):
    Cluster.fit(density_threshold = dens_thres[i], 
                distance_threshold = 1.0)
    yp_x = Cluster.get_labels(mixture_selection='aic')
    nmi_x[i] = metrics.normalized_mutual_info_score(df['species'].astype(int), yp_x.astype(int))
    ari_x[i] = metrics.adjusted_rand_score(df['species'].astype(int), yp_x.astype(int))

plt.plot(dens_thres, nmi_x, marker = 'o', label='nmi')
plt.plot(dens_thres, ari_x, marker = 'o', label='ari')
plt.xlabel('density threshold')
plt.ylabel('nmi/ari')
plt.legend()
plt.show()

# sensitivity w.r.t distance threshold
n = 5
nmi_x = np.zeros((n))
ari_x = np.zeros((n))
dist_thres = [0.90,0.95,1.00,1.05,1.10]

for i in range(n):
    Cluster.fit(density_threshold = 1.0, 
                distance_threshold = dist_thres[i])
    yp_x = Cluster.get_labels(mixture_selection='aic')
    nmi_x[i] = metrics.normalized_mutual_info_score(df['species'].astype(int), yp_x.astype(int))
    ari_x[i] = metrics.adjusted_rand_score(df['species'].astype(int), yp_x.astype(int))

plt.plot(dist_thres, nmi_x, marker = 'o', label='nmi')
plt.plot(dist_thres, ari_x, marker = 'o', label='ari')
plt.xlabel('distance threshold')
plt.ylabel('nmi/ari')
plt.legend()
plt.show()
