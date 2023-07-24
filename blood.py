# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 13:42:48 2023

@author: samuel veer singh
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import metrics
import seaborn as sns
from REMclust import REM
import pandas as pd
from scipy.io import arff


# ## ------------------------------------------------------------------------- ##
# #   Example 1
# ## ------------------------------------------------------------------------- ##
# # load Blood transfusion dataset from UCI 
# data = arff.loadarff('Data/blood.arff')
# df = pd.DataFrame(data[0])

# # sns.lineplot(data=df.drop(['Class'], axis=1)[:10])
# # plt.show()

# X = df[df.columns[:-1]]    
# X = np.array(X)

# scaler = preprocessing.StandardScaler().fit(X)
# X_scaled = scaler.transform(X)
# n_samples, n_features = X_scaled.shape

# # sns.lineplot(data=X_scaled[:100])
# # plt.show()

# bndwk = int(np.floor(np.min((30, np.sqrt(n_samples)))))
# Cluster = REM(data=X_scaled, covariance_type = "full", criteria = "all", bandwidth = bndwk, tol = 1e-3)
# Cluster.mode_decision_plot()
# Cluster.fit(density_threshold = 1.0, distance_threshold = 1.5)

# yp = Cluster.get_labels(mixture_selection='aic')

# # plot the clustered dataset
# plt.scatter(X_scaled[:, 2], X_scaled[:, 3], s = 5, c = yp, marker = "o", cmap='tab10')
# plt.xlabel('RI (refractive index)')
# plt.ylabel('Na weight %')
# plt.title('density threshold = 1' + '\n' + 'distance threshold = 1.5')
# plt.show()



## ------------------------------------------------------------------------- ##
#   Example 2
## ------------------------------------------------------------------------- ##
# Pulsar candidate data, used to detect pulsar stars.
df = pd.read_csv('Data/HTRU_2.csv')

X = df[df.columns[:-1]]    
X = np.array(X)

scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)
n_samples, n_features = X_scaled.shape


bndwk = int(np.floor(np.min((30, np.sqrt(n_samples)))))
Cluster = REM(data=X_scaled, covariance_type = "full", criteria = "all", bandwidth = bndwk, tol = 1e-3)
Cluster.mode_decision_plot()
Cluster.fit(density_threshold = 1.0, distance_threshold = 3.5)

yp = Cluster.get_labels(mixture_selection='aic')

# plot the clustered dataset
plt.scatter(X_scaled[:, 1], X_scaled[:, 5], s = 5, c = yp, marker = "o", cmap='tab10')
plt.xlabel('std of integrated radiation profile')
plt.ylabel('std of DM-SNR curve')
plt.title('density threshold = 1.0' + '\n' + 'distance threshold = 3.5')
plt.show()

# AMI & ARI
nmi = metrics.normalized_mutual_info_score(df['0'].astype(int), yp.astype(int))
ari = metrics.adjusted_rand_score(df['0'].astype(int), yp.astype(int))
print("Adjusted Rand Score: \t", ari)
print("Normalized Mutual Information Score: \t", nmi)


# sensitivity w.r.t density threshold
n = 5
nmi_x = np.zeros((n))
ari_x = np.zeros((n))
dens_thres = [0.95,1.00,1.05,1.10,1.15]

for i in range(n):
    Cluster.fit(density_threshold = dens_thres[i], 
                distance_threshold = 3.5)
    yp_x = Cluster.get_labels(mixture_selection='aic')
    nmi_x[i] = metrics.normalized_mutual_info_score(df['0'].astype(int), yp_x.astype(int))
    ari_x[i] = metrics.adjusted_rand_score(df['0'].astype(int), yp_x.astype(int))

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
dist_thres = [0.60,0.65,0.70,0.75,0.80]

for i in range(n):
    Cluster.fit(density_threshold = 1.0, 
                distance_threshold = dist_thres[i])
    yp_x = Cluster.get_labels(mixture_selection='aic')
    nmi_x[i] = metrics.normalized_mutual_info_score(df['Type'].astype(int), yp_x.astype(int))
    ari_x[i] = metrics.adjusted_rand_score(df['Type'].astype(int), yp_x.astype(int))

plt.plot(dens_thres, nmi_x, marker = 'o', label='nmi')
plt.plot(dens_thres, ari_x, marker = 'o', label='ari')
plt.xlabel('density threshold')
plt.ylabel('nmi/ari')
plt.legend()
plt.show()


# ## ------------------------------------------------------------------------- ##
# #   Example 3
# ## ------------------------------------------------------------------------- ##
# # MAGIC Gamma Telescope Dataset
# df = pd.read_csv('Data/zoo.csv')[1:]
# sns.lineplot(data=df.drop(['class_type'], axis=1)[:100])
# plt.show()

# X = df[df.columns[1:-1]]    
# X = np.array(X)

# scaler = preprocessing.StandardScaler().fit(X)
# X_scaled = scaler.transform(X)
# n_samples, n_features = X_scaled.shape


# bndwk = int(np.floor(np.min((30, np.sqrt(n_samples)))))
# Cluster = REM(data=X_scaled, covariance_type = "full", criteria = "all", bandwidth = bndwk, tol = 1e-3)
# Cluster.mode_decision_plot()
# Cluster.fit(density_threshold = 1.0, distance_threshold = 3.5)

# yp = Cluster.get_labels(mixture_selection='aic')

# # plot the clustered dataset
# plt.scatter(X_scaled[:, 1], X_scaled[:, 2], s = 5, c = yp, marker = "o", cmap='tab10')
# plt.xlabel('RI (refractive index)')
# plt.ylabel('Na weight %')
# plt.title('density threshold = 1' + '\n' + 'distance threshold = 1.5')
# plt.show()