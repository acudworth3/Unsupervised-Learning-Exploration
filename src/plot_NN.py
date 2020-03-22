import pandas as pd
import preprocess as prp
from scipy.stats import kurtosis, norm
import seaborn as sns
from numpy.linalg import pinv
from scipy.stats import  norm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.random_projection import GaussianRandomProjection
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, FastICA, FactorAnalysis
from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, accuracy_score


#pkr_data
# pkr_data = prp.pkr_data()
# pkr_data.clean()
# pkr_data.init_model_data(target =['hand'],features = ['suit1','card1','suit2','card2','suit3','card3','suit4','card4'])
#
# #AB Run
# # X=ab_data.X
# # y=ab_data.Y
# # title = "AB Data"
#
# # #pkr Run
# X=pkr_data.X
# y=pkr_data.Y
# title = "Poker Data"
#
# scaler = MinMaxScaler()
# X_pk=scaler.fit_transform(X)
#
# #Poker Run
# rng=42
# pca_pk = PCA(random_state=rng, n_components=4)
# pca_s_pk = pca_pk.fit(X_pk).transform(X_pk)
# ica_pk = FastICA(random_state=rng, n_components=4)
# ica_s_pk = ica_pk.fit(X_pk).transform(X_pk)  # Estimate the sources
# rpa_pk = GaussianRandomProjection(random_state=rng,n_components=4)
# rpa_s_pk = rpa_pk.fit(X_pk).transform(X_pk)  # Estimate the sources
# fca_pk = FactorAnalysis(random_state=rng,n_components=4)
# fca_s_pk = fca_pk.fit(X_pk).transform(X_pk)
#
# km_test = KMeans(n_clusters=5, random_state=rng).fit(pca_s_pk)
# preds = km_test.predict(pca_s_pk)
# em_test = GaussianMixture(n_components=5, n_init=10, tol=1e-3, max_iter=1000).fit(pca_s_pk)
# em_preds = em_test.predict(pca_s_pk)


# clstr_data_obj()
class clstr_data_obj:
    """A simple example class"""
    def __init__(self):
        self.rng = 42
        self.init_pkr_data()
        # self.dim_red()



    def init_pkr_data(self):
        self.pkr_data = prp.pkr_data()
        self.pkr_data.clean()
        self.pkr_data.init_model_data(target=['hand'],
                                 features=['suit1', 'card1', 'suit2', 'card2', 'suit3', 'card3', 'suit4', 'card4'])
        scaler = MinMaxScaler()
        self.X_pk = scaler.fit_transform(self.pkr_data.x_train)
        self.X_test = scaler.fit_transform(self.pkr_data.x_test)
        self.dim_red()
        self.cluster()

    def dim_red(self):
        pca_pk = PCA(random_state=self.rng, n_components=4)
        ica_pk = FastICA(random_state=self.rng, n_components=4)
        rpa_pk = GaussianRandomProjection(random_state=self.rng,n_components=4)          # Estimate the sources
        fca_pk = FactorAnalysis(random_state=self.rng,n_components=7)


        self.pca_s_pk = pca_pk.fit(self.X_pk).transform(self.X_pk)
        self.ica_s_pk = ica_pk.fit(self.X_pk).transform(self.X_pk)  # Estimate the sources
        self.rpa_s_pk = rpa_pk.fit(self.X_pk).transform(self.X_pk)  # Estimate
        self.fca_s_pk = fca_pk.fit(self.X_pk).transform(self.X_pk)

        self.samples = [self.X_pk,self.pca_s_pk, self.ica_s_pk, self.rpa_s_pk, self.fca_s_pk]
        
        self.test_samples = [self.X_test,
        pca_pk.fit(self.X_test).transform(self.X_test),
        ica_pk.fit(self.X_test).transform(self.X_test),
        rpa_pk.fit(self.X_test).transform(self.X_test),
        fca_pk.fit(self.X_test).transform(self.X_test)]


    def cluster(self):
        self.kM_clster_cnt = {'none':8,'PCA':16,'ICA':16,'RPA':18,'FCA':18}
        self.eM_clster_cnt = {'none': 9, 'PCA': 11, 'ICA': 13, 'RPA': 4, 'FCA': 2}
        self.idx_dict = {'none': 0, 'PCA': 1, 'ICA': 2, 'RPA': 3, 'FCA': 4}
        
        self.km_clstr_keys = list(self.kM_clster_cnt.values())
        self.km_objs = [KMeans(n_clusters=self.km_clstr_keys[idx], random_state=self.rng).fit(self.samples[idx]) for idx in range(len(self.samples))]
        self.km_lbls = [self.km_objs[idx].predict(self.samples[idx]) for idx in range(len(self.samples))]
        self.km_tst_lbls = [self.km_objs[idx].predict(self.test_samples[idx]) for idx in range(len(self.samples))]
        


        self.em_clstr_keys = list(self.eM_clster_cnt.values())
        self.em_objs = [KMeans(n_clusters=self.em_clstr_keys[idx], random_state=self.rng).fit(self.samples[idx]) for idx in range(len(self.samples))]
        self.em_lbls = [self.em_objs[idx].predict(self.samples[idx]) for idx in range(len(self.samples))]
        self.em_tst_lbls = [self.em_objs[idx].predict(self.test_samples[idx]) for idx in range(len(self.samples))]


        dump(self, 'nn_res\clstr_obj_.joblib')

        marker=1
        # km_test = KMeans(n_clusters=5, random_state=rng).fit(pca_s_pk)
        # preds = km_test.predict(pca_s_pk)
        # em_test = GaussianMixture(n_components=5, n_init=10, tol=1e-3, max_iter=1000).fit(pca_s_pk)
        # em_preds = em_test.predict(pca_s_pk)


class NN_runner:
    """A simple example class"""
    def __init__(self,data_file=None):
        self.rng = 42
        if data_file == None:
            clstrd_data = clstr_data_obj()
            self.data = clstr_data_obj
        else:
            self.data=load(data_file)
        self.base_NN = load('nn_res\clf_NN_pkr data_final_roc4855.joblib')
    def poc_test(self):

        dim_only = {}
        em_rslt_clstr = {}
        km_rslt_clstr = {}
        #Km

        for algo in list(self.data.kM_clster_cnt.keys()):
            marker = 1
            idx = self.data.idx_dict[algo]

            train_dim = self.data.samples[idx]
            test_dim = self.data.test_samples[idx]

            train_dim_clstr_km = np.hstack((self.data.km_lbls[idx].reshape(len(self.data.km_lbls[idx]), 1), self.data.samples[0]))
            test_dim_clstr_km = np.hstack(
                (self.data.km_tst_lbls[idx].reshape(len(self.data.km_tst_lbls[0]), 1), self.data.test_samples[0]))

            train_dim_clstr_em = np.hstack((self.data.em_lbls[idx].reshape(len(self.data.em_lbls[idx]), 1), self.data.samples[0]))
            test_dim_clstr_em = np.hstack(
                (self.data.em_tst_lbls[idx].reshape(len(self.data.em_tst_lbls[0]), 1), self.data.test_samples[0]))


            NN_pkr_dim = MLPClassifier(activation='relu', alpha=0.1, batch_size='auto', beta_1=0.9,
                                   beta_2=0.999, early_stopping=False, epsilon=1e-08,
                                   hidden_layer_sizes=(train_dim.shape[1],train_dim.shape[1],8,10), learning_rate='constant',
                                   learning_rate_init=0.001, max_fun=15000, max_iter=1000,
                                   momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,
                                   power_t=0.5, random_state=None, shuffle=True, solver='lbfgs',
                                   tol=0.0001, validation_fraction=0.1, verbose=False,
                                   warm_start=False)

            NN_pkr_dim_clstr = MLPClassifier(activation='relu', alpha=0.1, batch_size='auto', beta_1=0.9,
                                   beta_2=0.999, early_stopping=False, epsilon=1e-08,
                                   hidden_layer_sizes=(train_dim_clstr_km.shape[1],train_dim_clstr_km.shape[1],8, 10), learning_rate='constant',
                                   learning_rate_init=0.001, max_fun=15000, max_iter=1000,
                                   momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,
                                   power_t=0.5, random_state=None, shuffle=True, solver='lbfgs',
                                   tol=0.0001, validation_fraction=0.1, verbose=False,
                                   warm_start=False)


            #dim only
            NN_pkr_dim.fit(train_dim,self.data.pkr_data.y_train)
            train_auc = roc_auc_score(self.data.pkr_data.y_train, NN_pkr_dim.predict(train_dim), average='macro')
            test_auc = roc_auc_score(self.data.pkr_data.y_test, NN_pkr_dim.predict(test_dim), average='macro')
            dim_only[algo]= (train_auc,test_auc)

            # EM
            NN_pkr_dim_clstr.fit(train_dim_clstr_em,self.data.pkr_data.y_train)
            train_auc = roc_auc_score(self.data.pkr_data.y_train,NN_pkr_dim_clstr.predict(train_dim_clstr_em))
            test_auc = roc_auc_score(self.data.pkr_data.y_test, NN_pkr_dim_clstr.predict(test_dim_clstr_em))
            em_rslt_clstr[algo] = (train_auc,test_auc)

            #KM
            NN_pkr_dim_clstr.fit(train_dim_clstr_km,self.data.pkr_data.y_train)
            train_auc = roc_auc_score(self.data.pkr_data.y_train,NN_pkr_dim_clstr.predict(train_dim_clstr_km))
            test_auc = roc_auc_score(self.data.pkr_data.y_test, NN_pkr_dim_clstr.predict(test_dim_clstr_km))

            train_acc = accuracy_score(self.data.pkr_data.y_train, NN_pkr_dim_clstr.predict(train_dim_clstr_km))
            test_acc = accuracy_score(self.data.pkr_data.y_test, NN_pkr_dim_clstr.predict(test_dim_clstr_km))

            km_rslt_clstr[algo] = (train_auc,test_auc)
            marker = 1

        self.result ={'DIM':dim_only,'EM':em_rslt_clstr,'KM':km_rslt_clstr}
        dump(self, 'nn_res\\NN_results_' + str(np.random.randint(0, 99999)) + '_.joblib')
        print(self.result)



# data_obj = clstr_data_obj()
# NN_1 = NN_runner(data_file=None)
#NN_1 = NN_runner(data_file='nn_res\clstr_obj_.joblib')
NN_1.poc_test()
marker = 1

