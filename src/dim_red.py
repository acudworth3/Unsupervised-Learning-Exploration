

#PCA
#ICA https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html#sklearn.decomposition.FastICA
#RPA (https://scikit-learn.org/stable/modules/unsupervised_reduction.html#random-projections)
#LDA


import pandas as pd
import preprocess as prp
from scipy.stats import kurtosis, norm
import seaborn as sns
from numpy.linalg import pinv
from scipy.stats import  norm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import PCA, FastICA, FactorAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture

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
# X=ab_data.X
# y=ab_data.Y
# title = "AB Data"

# #pkr Run
# X=pkr_data.X
# y=pkr_data.Y
# title = "Poker Data"
# ### Scaling
#TODO save all plots

# scaler = MinMaxScaler()
# X=scaler.fit_transform(X)


# Authors: Alexandre Gramfort, Gael Varoquaux
# License: BSD 3 clause
# #############################################################################
# Generate sample data
def plot_FCA_expl(ab_data,pkr_data):
    rng = np.random.RandomState(42)
    # AB Run
    # X = ab_data.X
    # y = ab_data.Y
    # title = "AB Data"
    scaler = MinMaxScaler()
    # X = scaler.fit_transform(X)

    pkr_dict = {'n_var':[],'cmpts':[],'data':'PKR'}
    ab_dict = {'n_var':[],'cmpts':[],'data':'AB'}

    #AB
    Xa = scaler.fit_transform(ab_data.X)
    for cmpts in range(2,Xa.shape[1]):
        fca_ab = FactorAnalysis(random_state=rng, n_components=cmpts)
        fca_s_ab = fca_ab.fit(Xa).transform(Xa)

        ab_dict['cmpts'].append(cmpts)
        ab_dict['n_var'].append(np.mean(fca_ab.noise_variance_))


        # fg = sns.pairplot(pd.DataFrame(fca_s_ab))
        # fg.fig.suptitle('Airbnb FC Components Correlation', y=1.0, fontsize=14)
        # plt.show()
        # plt.close()
    #PKR
    Xp = scaler.fit_transform(pkr_data.X)
    for cmpts in range(2, Xp.shape[1]):

        fca_pk = FactorAnalysis(random_state=rng, n_components=cmpts)
        fca_s_pk = fca_pk.fit(Xp).transform(Xp)

        pkr_dict['cmpts'].append(cmpts)
        pkr_dict['n_var'].append(np.mean(fca_pk.noise_variance_))
        # sns.pairplot(pd.DataFrame(fca_s_pk))
        # plt.show()
        # plt.close()

    #PKR 2cmpts
    fca_pk = FactorAnalysis(random_state=rng, n_components=2)
    fca_s_pk = fca_pk.fit(Xp).transform(Xp)

    fg = sns.pairplot(pd.DataFrame(fca_s_pk))
    fg.fig.suptitle('Poker FC Components Correlation', y=1.0, fontsize=14)
    plt.savefig('fca_pk_pp.png')
    plt.close()    # plt.hist(fca_s_ab,bins=100,type='step')

    #AB 2 Cmponents
    fca_ab = FactorAnalysis(random_state=rng, n_components=4)
    fca_s_ab = fca_ab.fit(Xa).transform(Xa)

    fg = sns.pairplot(pd.DataFrame(fca_s_ab))
    fg.fig.suptitle('Airbnb FC Components Correlation', y=1.0, fontsize=23)
    plt.savefig('fca_ab_pk.png')
    plt.show()
    plt.close()

    # plt.xlabel("Factor Value")
    # plt.ylabel('Sample Count')
    # plt.show()
    # plt.close()
    marker = 1



def plot_loss(ab_data,pkr_data):
    rng = np.random.RandomState(42)
    # AB Run
    X = ab_data.X
    y = ab_data.Y
    title = "AB Data"
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    ab_loss = {'PCA':[],'ICA':[],'RPA':[],'FCA':[],'cmpts':[],'data':'AB'}
    for cmpts in range(2,X.shape[1]):
        ab_loss['cmpts'].append(cmpts)
        pca = PCA(random_state=rng,n_components=cmpts)
        pca_s = pca.fit(X).transform(X)
        ab_loss['PCA'].append(((pca.inverse_transform(pca_s)-X)**2).mean()) #reconstruction loss

        ica = FastICA(random_state=rng,n_components=cmpts)
        ica_s = ica.fit(X).transform(X)  # Estimate the sources
        ab_loss['ICA'].append(((ica.inverse_transform(ica_s)-X)**2).mean()) #reconstruction loss

        rpa = GaussianRandomProjection(n_components=cmpts)
        rpa_s = rpa.fit(X).transform(X)  # Estimate the sources
        runs = 1
        rpa_losses = []
        for run in range(runs):
           rpa = GaussianRandomProjection(n_components=cmpts,random_state=np.random.randint(0,99999))
           rpa_losses.append(np.mean((np.dot(rpa.fit(X).transform(X),np.linalg.pinv(rpa.components_.T))-X)**2))
        ab_loss['RPA'].append(np.mean(rpa_losses))
        fca = FactorAnalysis(random_state=rng, n_components=cmpts)
        fca_s = fca.fit(X).transform(X)
        fca_loss = np.mean((np.dot(fca_s,np.linalg.pinv(fca.components_.T))-X)**2)
        ab_loss['FCA'].append(fca_loss)
    marker = 1






    # #pkr Run
    X=pkr_data.X
    y=pkr_data.Y
    title = "Poker Data"
    X = scaler.fit_transform(X)

    pk_loss = {'PCA': [], 'ICA': [],'RPA':[] ,'FCA':[],'cmpts': [],'data':'PKR'}
    for cmpts in range(2, X.shape[1]):
        pk_loss['cmpts'].append(cmpts)
        pca = PCA(random_state=rng, n_components=cmpts)
        pca_s = pca.fit(X).transform(X)
        pk_loss['PCA'].append(((pca.inverse_transform(pca_s) - X) ** 2).mean())  # reconstruction loss

        ica = FastICA(random_state=rng, n_components=cmpts)
        ica_s = ica.fit(X).transform(X)  # Estimate the sources
        pk_loss['ICA'].append(((ica.inverse_transform(ica_s) - X) ** 2).mean())  # reconstruction loss

        runs = 1
        rpa_losses = []
        for run in range(runs):
            rpa = GaussianRandomProjection(n_components=cmpts, random_state=np.random.randint(0, 99999))
            rpa_losses.append(np.mean((np.dot(rpa.fit(X).transform(X), np.linalg.pinv(rpa.components_.T)) - X) ** 2))
        pk_loss['RPA'].append(np.mean(rpa_losses))

        rpa = GaussianRandomProjection(random_state=rng,n_components=cmpts)
        rpa_s = rpa.fit_transform(X)

        sns.pairplot(pd.DataFrame(rpa_s))
        plt.show()
        plt.close()



        fca = FactorAnalysis(random_state=rng, n_components=cmpts)
        fca_s = fca.fit(X).transform(X)
        fca_loss = np.mean((np.dot(fca_s,np.linalg.pinv(fca.components_.T))-X)**2)
        pk_loss['FCA'].append(fca_loss)
    marker = 1

    for key in ['ICA','PCA','RPA']:
        if key == 'ICA':
            plt.plot(ab_loss['cmpts'],ab_loss[key],'bo-',label='AB_'+key)
            # plt.plot(pk_loss['cmpts'], pk_loss[key],'ro-', label='pk_' +key)
        else:
            plt.plot(ab_loss['cmpts'],ab_loss[key],'b^-',label='AB_'+key)
            # plt.plot(pk_loss['cmpts'], pk_loss[key],'r^-', label='pk_' +key)
        #TODO work on making idenitcalness clear

    plt.legend()
    plt.show()
    marker = 1


#PCA EVA
def plot_PCA_EV(ab_data,pkr_data):
    rng = np.random.RandomState(42)
    # AB Run
    X = ab_data.X
    y = ab_data.Y
    title = "AB Data"
    scaler = MinMaxScaler()
    # X = scaler.fit_transform(X)



    ab_exp_var_loss = {'ev': [],'evr': [],'cmpts': [], 'data': 'AB'}
    for cmpts in range(4, X.shape[1]):
        pca = PCA(random_state=rng, n_components=cmpts)
        pca_s = pca.fit(X).transform(X)
        plt.scatter(range(cmpts),pca.explained_variance_ratio_,label='AB_data'+str(cmpts))
        ab_exp_var_loss['cmpts'].append(cmpts)
        ab_exp_var_loss['ev'].append(pca.explained_variance_)
        ab_exp_var_loss['evr'].append(pca.explained_variance_ratio_)
    # #pkr Run
    X = pkr_data.X
    y = pkr_data.Y
    title = "Poker Data"
    # X = scaler.fit_transform(X)

    pk_exp_var_loss = {'ev': [],'evr': [],'cmpts': [],'data': 'PKR'}
    for cmpts in range(7, X.shape[1]):
        pca = PCA(random_state=rng, n_components=cmpts)
        pca_s = pca.fit(X).transform(X)

        plt.scatter(range(cmpts),pca.explained_variance_ratio_,label='PK_data'+str(cmpts))
        pk_exp_var_loss['cmpts'].append(cmpts)
        pk_exp_var_loss['ev'].append(pca.explained_variance_)
        pk_exp_var_loss['evr'].append(pca.explained_variance_ratio_)


    plt.legend()
    plt.show()
    marker=1


def plot_ICA_Kurt(ab_data,pkr_data):
    rng = np.random.RandomState(42)
    # AB Run
    X = ab_data.X
    y = ab_data.Y
    title = "AB Data"
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    ab_kurt = {'ICA': [],'cmpts': [],'avg_kurt':[], 'data': 'AB'}
    for cmpts in range(2, X.shape[1]):
        # ab_exp_var_loss['cmpts'].append(cmpts)
        ica = FastICA(random_state=rng, n_components=cmpts)
        ica_s = ica.fit(X).transform(X)  # Estimate the sources
        ab_kurt['avg_kurt'].append(kurtosis(ica_s).mean())
        ab_kurt['cmpts'].append(cmpts)
        offset=0
        for idx in range(ica_s.shape[1]):
            kurt_comp = ica_s[:,idx]
            plt.bar(cmpts+offset, kurtosis(kurt_comp), label='AB_data_' + str(cmpts)+'_'+str(idx),width=0.1)
            offset+=0.1
    # #pkr Run
    X = pkr_data.X
    y = pkr_data.Y
    title = "Poker Data"
    X = scaler.fit_transform(X)

    pk_kurt = {'ICA': [],'cmpts': [],'avg_kurt':[],'data': 'PKR'}
    for cmpts in range(2, X.shape[1]):
        # ab_exp_var_loss['cmpts'].append(cmpts)
        ica = FastICA(random_state=rng, n_components=cmpts)
        ica_s = ica.fit(X).transform(X)  # Estimate the sources
        pk_kurt['avg_kurt'].append(kurtosis(ica_s).mean())
        pk_kurt['cmpts'].append(cmpts)

        offset=0
        for idx in range(ica_s.shape[1]):
            kurt_comp = ica_s[:,idx]
            # plt.bar(cmpts+offset, kurtosis(kurt_comp), label='PKR_data_' + str(cmpts)+'_'+str(idx),width=0.1)
            plt.bar(idx+offset, kurtosis(kurt_comp), label='PKR_data_' + str(cmpts)+'_'+str(idx),width=0.1)

            offset+=0.1


    plt.legend()
    plt.show()
    marker=1

def plot_histos_ICA(ab_data,pkr_data):
    #attempt visual
    scaler = MinMaxScaler()
    X=scaler.fit_transform(ab_data.X)
    ica = FastICA(random_state=rng, n_components=4)
    ica_s_ab = ica.fit(X).transform(X)


    bins = 100
    fig, axes = plt.subplots(nrows=2, ncols=2,constrained_layout=True)
    ax0, ax1, ax2, ax3 = axes.flatten()
    ax0.hist(ica_s_ab[:,0],bins=bins)
    ax0.set_xlabel('IC 1')
    ax0.set_xlim(-0.025,0.025)
    ax1.hist(ica_s_ab[:,1], bins=bins)
    ax1.set_xlabel('IC 2')
    ax1.set_xlim(-0.025,0.025)
    ax2.hist(ica_s_ab[:,2], bins=bins)
    ax2.set_xlabel('IC 3')
    ax2.set_xlim(-0.01, 0.015)
    ax3.hist(ica_s_ab[:,3], bins=bins)
    ax3.set_xlabel('IC 4')
    ax3.set_xlim(-0.005, 0.005)

    ax0.set_title('AB Data 100 Bin ICA Histograms',y=0.98)
    fig.tight_layout()
    plt.savefig('ica_ab_his.png')
    plt.close()
    # plt.show()

    #PKR
    X=scaler.fit_transform(pkr_data.X)
    ica = FastICA(random_state=rng, n_components=4)
    ica_s_pk = ica.fit(X).transform(X)

    bins = 100
    fig, axes = plt.subplots(nrows=2, ncols=2,constrained_layout=True)
    ax0, ax1, ax2, ax3 = axes.flatten()
    ax0.hist(ica_s_pk[:,0],bins=bins)
    ax0.set_xlabel('IC 1')
    # ax0.set_xlim(-0.025,0.025)
    ax1.hist(ica_s_pk[:,1], bins=bins)
    ax1.set_xlabel('IC 2')
    # ax1.set_xlim(-0.025,0.025)
    ax2.hist(ica_s_pk[:,2], bins=bins)
    ax2.set_xlabel('IC 3')
    # ax2.set_xlim(-0.01, 0.015)
    ax3.hist(ica_s_pk[:,3], bins=bins)
    ax3.set_xlabel('IC 4')
    # ax3.set_xlim(-0.005, 0.005)

    ax0.set_title('PK Data 100 Bin ICA Histograms',y=0.98)
    fig.tight_layout()
    plt.show()
    plt.savefig('ica_pk_his.png')
    plt.close()

def plot_pair_plots(ab_data,pkr_data):
    sns.set(style="ticks", color_codes=True)
    rng = np.random.RandomState(42)
    scaler = MinMaxScaler()

    #TABLE 2
    fg = sns.pairplot(ab_data.X)
    fg.fig.suptitle('Airbnb Original Feature Correlation', y=1.0, fontsize=30)

    plt.tight_layout()
    plt.savefig('figs_tables/AB_Table2_pp.png')
    plt.close()

    fg = sns.pairplot(pkr_data.X)
    fg.fig.suptitle('Poker Original Feature Correlation', y=1.0, fontsize=30)

    plt.tight_layout()
    plt.savefig('figs_tables/PK_Table2_pp.png')
    plt.close()


    #Base
    sns.pairplot(ab_data.X, x_vars=['price'], y_vars=['number_of_reviews'])
    plt.title("AB Data Price vs Reviews")
    plt.grid()
    plt.tight_layout()
    plt.savefig('AB_pp_base.png')
    plt.close()
    # plt.xlabel("PCA 1")
    # plt.ylabel("PCA 2")

    #PCA
    #AB
    X=scaler.fit_transform(ab_data.X)
    pca = PCA(random_state=rng, n_components=2)
    pca_s_ab = pca.fit(X).transform(X)

    sns.pairplot(pd.DataFrame(pca_s_ab), x_vars=[0], y_vars=[1])
    plt.title("AB Data PCA 1 v PCA 2")
    plt.grid()
    plt.tight_layout()
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")

    plt.savefig('pca_ab_pp.png')
    # plt.show()
    plt.close()
    #PKR
    X=scaler.fit_transform(pkr_data.X)
    pca = PCA(random_state=rng, n_components=4)
    pca_s_pk = pca.fit(X).transform(X)
    sns.pairplot(pd.DataFrame(pca_s_pk))
    plt.savefig('pca_pkr_pp.png')
    # plt.show()
    plt.close()


    #ICA

    # ica = FastICA(random_state=rng, n_components=cmpts)
    # ica_s = ica.fit(X).transform(X)  # Estimate the sources

    # S_ica_ /= S_ica_.std(axis=0)
    scaler = MinMaxScaler()
    X=scaler.fit_transform(ab_data.X)
    ica = FastICA(random_state=rng, n_components=4)
    ica_s_ab = ica.fit(X).transform(X)

    sns.pairplot(pd.DataFrame(ica_s_ab))
    plt.savefig('ica_ab_pp.png')
    plt.close()
    #PKR
    X=scaler.fit_transform(pkr_data.X)
    ica = FastICA(random_state=rng, n_components=4)
    ica_s_pk = ica.fit(X).transform(X)
    sns.pairplot(pd.DataFrame(ica_s_pk))
    plt.savefig('ica_pkr_pp.png')
    # plt.show()
    plt.close()


    #RPA
    scaler = MinMaxScaler()
    X = scaler.fit_transform(ab_data.X)
    rpa = GaussianRandomProjection(random_state=rng, n_components=4)
    rpa_s_ab = rpa.fit(X).transform(X)

    sns.pairplot(pd.DataFrame(rpa_s_ab))
    plt.suptitle('Airbnb RP Pair Plot (4 Component)',y=1.08)

    plt.savefig('rpa_ab_pp.png')
    plt.show()
    plt.close()
    # PKR
    X = scaler.fit_transform(pkr_data.X)
    rpa = GaussianRandomProjection(random_state=rng, n_components=4)
    rpa_s_pk = rpa.fit(X).transform(X)
    sns.pairplot(pd.DataFrame(rpa_s_pk))
    plt.suptitle('Poker RP Pair Plot (4 Component)',y=1.08)
    plt.savefig('rpa_pkr_pp.png')
    plt.show()
    plt.close()

    #LDA
    scaler = MinMaxScaler()
    X = scaler.fit_transform(ab_data.X)
    lda = LinearDiscriminantAnalysis(n_components=2)
    lda_s_ab = lda.fit(X,ab_data.Y).transform(X)

    sns.pairplot(pd.DataFrame(lda_s_ab))
    plt.savefig('lda_ab_pp.png')
    # plt.show()
    plt.close()
    # PKR
    X = scaler.fit_transform(pkr_data.X)
    lda = LinearDiscriminantAnalysis(n_components=2)
    lda_s_pk = lda.fit(X,pkr_data.Y).transform(X)
    sns.pairplot(pd.DataFrame(lda_s_pk))
    plt.savefig('lda_pkr_pp.png')
    # plt.show()
    plt.close()
    marker=1


    #
    # lda = LinearDiscriminantAnalysis(n_components=cmpts)
    # lda_s = lda.fit(X,y).transform(X)  # Estimate the sources

    #RP

    #LDA
    marker=1
    pass

def plot_RPA_run_var(ab_data,pkr_data):
    #AB pass
    rng = np.random.RandomState(42)
    # AB Run
    X = ab_data.X
    y = ab_data.Y
    title = "AB Data"
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    # norm_rvs = []
    for title in ["AB Data","PKR Data"]:
        if title in ["AB Data"]:
            X = ab_data.X
            y = ab_data.Y
            X = scaler.fit_transform(X)
        else:
            X = pkr_data.X
            y = pkr_data.Y
            X = scaler.fit_transform(X)

        for cmpts in range(X.shape[1]-1,X.shape[1]):
            rpa = GaussianRandomProjection(n_components=cmpts)
            # rpa_s = rpa.fit(X).transform(X)  # Estimate the sources
            runs = 100
            rpa_values = []
            rp_tensor = np.zeros((X.shape[0],cmpts,runs))
            for run in range(runs):
                rpa = GaussianRandomProjection(n_components=cmpts, random_state=np.random.randint(0, 99999))
                rpa_s = rpa.fit(X).transform(X)  # Estimate the sources
                rp_tensor[:,:,run]+=rpa_s
            run_avgs = np.mean(rp_tensor,axis=2)
            run_stds = np.std(rp_tensor, axis=2)
            cmpts_avg = np.mean(run_avgs,axis=0)
            cmpts_std = np.mean(run_stds,axis=0)
        [norm.rvs(loc=cmpts_avg[idx],scale=cmpts_std[idx],size=10000) for idx in range(len(cmpts_std))]

        plt.hist([norm.rvs(loc=cmpts_avg[idx], scale=cmpts_std[idx], size=10000) for idx in range(len(cmpts_std))],
                 bins=100, histtype='step', label=['RP_1', 'RP_2', 'RP_3', 'RP_4']);
        [plt.plot([cmpts_avg[idx],cmpts_avg[idx]],[0,400],label='RP_'+str(idx)+'_avg',linestyle='dashed') for idx in range(len(cmpts_avg))]

        plt.legend()
        plt.title(title+'Random Components 100 run Normalized Distribution',font_size=14)
        plt.ylabel('Value Count (10k sample)')
        plt.xlabel('Random Component value')
        plt.tight_layout()
        plt.savefig('RP_'+title+'_histograms.png')
        # plt.grid()
        plt.show()
        plt.close()
    marker = 1

    #TODO guassian plot of components

def plot_re_clstr(pkr_data):
    #AB pass
    rng = np.random.RandomState(42)
    # AB Run
    X = pkr_data.X
    y = pkr_data.Y
    title = "AB Data"
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    fca_ab = FactorAnalysis(random_state=rng, n_components=2)
    fca_s_ab = fca_ab.fit(X).transform(X)

    gm = GaussianMixture(n_components=18, n_init=10, tol=1e-3, max_iter=1000).fit(fca_s_ab)
    lbls = preds = gm.fit_predict(fca_s_ab)
    # norm_rvs = []
    fig_17(df=1,title='fig_17')
    marker = 1

def fig_17(df,title):
    #for loop on features
    df = df.drop(['Unnamed: 0'], axis=1)

    # x = df.values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    # x_scaled = min_max_scaler.fit_transform(x)
    # df = pd.DataFrame(x_scaled)

    # https://stackoverflow.com/questions/33864578/matplotlib-making-labels-for-violin-plots
    labels = []
    def add_label(violin, label):
        color = violin["bodies"][0].get_facecolor().flatten()
        labels.append((mpatches.Patch(color=color), label))

    # pos = np.array(np.sort(df['cluster'].unique()), dtype=float)-0.35
    pos = np.array(list(range(3)), dtype=float)-0.35
    for feature in list(df.columns):

        # if feature in ['card1','card2','card3','card4']:
        if feature in ['suit1','suit2','suit3','suit4']:
            #normaliz

            df[feature] = (df[feature]- df[feature].min())/(df[feature].max()-df[feature].min())
            # clst_split = [np.array(df[df['cluster'] == clstr][feature]) for clstr in np.sort(df['cluster'].unique())]
            clst_split = [np.array(df[df['cluster'] == clstr][feature]) for clstr in list(range(3))]
            # plt.violinplot(clst_split)
            add_label(plt.violinplot(clst_split,widths = 0.2,positions=np.copy(pos)), feature)
            pos += 0.15


    plt.legend(*zip(*labels),loc=1)
    plt.grid(axis='y')
    plt.xticks([0,1,2])
    plt.xlabel('Clusters 0-2 of 9')
    plt.ylabel('Normalized Suite Value')
    plt.title('PKR Cluster Subset vs Norm. Feature Distribution (EM)')
    plt.tight_layout()

    plt.show()
    plt.savefig('PK_'+title+'_clusters.png') #TODO uncomment
    # plt.show()
    plt.close()

# plot_ICA_Kurt(ab_data=ab_data,pkr_data=pkr_data)
# plot_PCA_EV(ab_data=ab_data,pkr_data=pkr_data)
# plot_loss(ab_data=ab_data, pkr_data=pkr_data)
# plot_histos_ICA(ab_data=ab_data,pkr_data=pkr_data)
# plot_pair_plots(ab_data=ab_data,pkr_data=pkr_data)
# plot_RPA_run_var(ab_data=ab_data,pkr_data=pkr_data)
# plot_FCA_expl(ab_data=ab_data,pkr_data=pkr_data)
plot_re_clstr(pkr_data=pkr_data)
#base
# pca = PCA(random_state=rng, n_components=cmpts)
# pca_s = pca.fit(X).transform(X)
#
# ica = FastICA(random_state=rng, n_components=cmpts)
# ica_s = ica.fit(X).transform(X)  # Estimate the sources

# S_ica_ /= S_ica_.std(axis=0)

# rpa = GaussianRandomProjection(random_state=rng,n_components=cmpts)
# rpa_s = rpa.fit(X).transform(X)  # Estimate the sources
#
# lda = LinearDiscriminantAnalysis(n_components=cmpts)
# lda_s = lda.fit(X,y).transform(X)  # Estimate the sources
