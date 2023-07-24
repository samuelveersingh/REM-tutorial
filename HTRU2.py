# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 13:24:37 2023

@author: samuel veer singh
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from REMclust import REM
import pandas as pd

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