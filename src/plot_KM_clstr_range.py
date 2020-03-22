#!/usr/bin/env python
# coding: utf-8
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import preprocess as prp
import seaborn as sns


# ### Create synthetic data using Scikit learn `make_blob` method
# 
# - Number of features: 4
# - Number of clusters: 5
# - Number of samples: 200

# In[2]:


from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs


# n_features = 4
# n_cluster = 5
# cluster_std = 1.2
# n_samples = 200
# data1 = make_blobs(n_samples=n_samples,n_features=n_features,centers=n_cluster,cluster_std=cluster_std)
# d1 = data1[0]
# df1=pd.DataFrame(data=d1,columns=['Feature_'+str(i) for i in range(1,n_features+1)])
# df1.head()
# from itertools import combinations
# lst_vars=list(combinations(df1.columns,2))
# len(lst_vars)
# plt.figure(figsize=(15,8))
# for i in range(1,7):
#     plt.subplot(2,3,i)
#     dim1=lst_vars[i-1][0]
#     dim2=lst_vars[i-1][1]
#     plt.scatter(df1[dim1],df1[dim2],c=data1[1],edgecolor='k',s=150)
#     plt.xlabel(f"{dim1}",fontsize=13)
#     plt.ylabel(f"{dim2}",fontsize=13)
# ### How are the classes separated (boxplots)
# plt.figure(figsize=(16,14))
# for i,c in enumerate(df1.columns):
#     plt.subplot(3,2,i+1)
#     sns.boxplot(y=df1[c],x=data1[1])
#     plt.xticks(fontsize=15)
#     plt.yticks(fontsize=15)
#     plt.xlabel("Class",fontsize=15)
#     plt.ylabel(c,fontsize=15)
#     #plt.show()


# ## k-means clustering



from sklearn.cluster import KMeans
# ### Unlabled data

#pkr_data
pkr_data = prp.pkr_data()
pkr_data.clean()
pkr_data.init_model_data(target =['hand'],features = ['suit1','card1','suit2','card2','suit3','card3','suit4','card4'])

#ab data
ab_data = prp.ab_data()
ab_data.clean()
ab_data.target = 'room_type'
ab_data.features = ab_data.all.columns[ab_data.all.columns != ab_data.target]
ab_data.init_model_data(target=ab_data.target,features=ab_data.features)

#AB Run
X=ab_data.X
y=ab_data.Y
title = "AB Data"


#pkr Run
# X=pkr_data.X
# y=pkr_data.Y
# title = "PKR Data"


# X=df1
# X.head()
# y=data1[1]


# ### Scaling

# In[16]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled=scaler.fit_transform(X)
# ### Metrics
from sklearn.metrics import silhouette_score, davies_bouldin_score,v_measure_score
# ### Running k-means and computing inter-cluster distance score for various *k* values
km_scores= []
km_silhouette = []
vmeasure_score =[]
db_score = []
clst_min = 2
clst_max = 15
for i in range(clst_min,clst_max):
    km = KMeans(n_clusters=i, random_state=0).fit(X_scaled)
    preds = km.predict(X_scaled)
    
    print("Score for number of cluster(s) {}: {}".format(i,km.score(X_scaled)))
    km_scores.append(-km.score(X_scaled))
    
    silhouette = silhouette_score(X_scaled,preds)
    km_silhouette.append(silhouette)
    print("Silhouette score for number of cluster(s) {}: {}".format(i,silhouette))
    
    # db = davies_bouldin_score(X_scaled,preds)
    # db_score.append(db)
    # print("Davies Bouldin score for number of cluster(s) {}: {}".format(i,db))
    
    # v_measure = v_measure_score(y,preds)
    # vmeasure_score.append(v_measure)
    # print("V-measure score for number of cluster(s) {}: {}".format(i,v_measure))
    # print("-"*100)


# In[21]:


plt.figure(figsize=(7,4))
plt.title(title+" K Clusters vs Inertia Score",fontsize=16)
plt.scatter(x=[i for i in range(clst_min,clst_max)],y=km_scores,s=150,edgecolor='k')
plt.grid(True)
plt.xlabel("K Clusters",fontsize=14)
plt.ylabel("K-means score",fontsize=15)
plt.xticks([i for i in range(clst_min,clst_max)],fontsize=14)
plt.yticks(fontsize=15)
plt.tight_layout()
plt.savefig(title+'_kmean_inert_score.png')
# plt.show()
plt.close()

plt.figure(figsize=(7,4))
plt.title(title+" K Clusters vs Silhouette Coefficient",fontsize=16)
plt.scatter(x=[i for i in range(clst_min,clst_max)],y=km_silhouette,s=150,edgecolor='k')
plt.grid(True)
plt.xlabel("K Clusters",fontsize=14)
plt.ylabel("Silhouette score",fontsize=15)
plt.xticks([i for i in range(clst_min,clst_max)],fontsize=14)
plt.yticks(fontsize=15)
plt.tight_layout()
plt.savefig(title+'_kmean_sil_score.png')
# plt.show()
plt.close()

marker = 1
#TODO bring silhouette plot over here

# ## Expectation-maximization (Gaussian Mixture Model)

# from sklearn.mixture import GaussianMixture
# gm_bic= []
# gm_score=[]
# for i in range(2,12):
#     gm = GaussianMixture(n_components=i,n_init=10,tol=1e-3,max_iter=1000).fit(X_scaled)
#     print("BIC for number of cluster(s) {}: {}".format(i,gm.bic(X_scaled)))
#     print("Log-likelihood score for number of cluster(s) {}: {}".format(i,gm.score(X_scaled)))
#     print("-"*100)
#     gm_bic.append(-gm.bic(X_scaled))
#     gm_score.append(gm.score(X_scaled))
# plt.figure(figsize=(7,4))
# plt.title("The Gaussian Mixture model BIC \nfor determining number of clusters\n",fontsize=16)
# plt.scatter(x=[i for i in range(2,12)],y=np.log(gm_bic),s=150,edgecolor='k')
# plt.grid(True)
# plt.xlabel("Number of clusters",fontsize=14)
# plt.ylabel("Log of Gaussian mixture BIC score",fontsize=15)
# plt.xticks([i for i in range(2,12)],fontsize=14)
# plt.yticks(fontsize=15)
# plt.show()
#
#
# # In[28]:
#
#
# plt.scatter(x=[i for i in range(2,12)],y=gm_score,s=150,edgecolor='k')
# plt.show()
#
