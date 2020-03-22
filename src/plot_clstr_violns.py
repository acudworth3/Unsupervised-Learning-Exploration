import pandas as pd
import numpy as np
from  matplotlib import  pyplot as plt
import matplotlib.patches as mpatches
from sklearn import preprocessing


km_ab_data = pd.read_csv('km_ab_clstrdata.csv')
km_pk_data = pd.read_csv('km_PKR Data_clstrdata.csv')


em_ab_data = pd.read_csv('em_AB Data_clstrdata.csv')
em_pk_data = pd.read_csv('em_PKR Data_clstrdata.csv')


#TODO scale results


def ab_km_cluster_plots(df,title):
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

    pos = np.array(np.sort(df['cluster'].unique()), dtype=float)-0.25
    for feature in list(df.columns):

        if feature in ['label','price','minimum_nights']:
            #normaliz

            df[feature] = (df[feature]- df[feature].min())/(df[feature].max()-df[feature].min())
            clst_split = [np.array(df[df['cluster'] == clstr][feature]) for clstr in np.sort(df['cluster'].unique())]
            # plt.violinplot(clst_split)
            add_label(plt.violinplot(clst_split,widths = 0.25,positions=np.copy(pos)), feature)
            pos += 0.25

    plt.legend(*zip(*labels))
    plt.grid(axis='y')
    plt.xlabel('Cluster')
    plt.ylabel('Normalized Feature Value')
    plt.title('AB Cluster vs Norm. Feature Distribution (K-Means)')
    plt.tight_layout()

    plt.savefig('AB_'+title+'_clusters.png')
    plt.show()

def pkr_km_cluster_plots(df,title):
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

        if feature in ['card1','card2','card3','card4']:
            #normaliz

            df[feature] = (df[feature]- df[feature].min())/(df[feature].max()-df[feature].min())
            # clst_split = [np.array(df[df['cluster'] == clstr][feature]) for clstr in np.sort(df['cluster'].unique())]
            clst_split = [np.array(df[df['cluster'] == clstr][feature]) for clstr in list(range(3))]
            # plt.violinplot(clst_split)
            add_label(plt.violinplot(clst_split,widths = 0.2,positions=np.copy(pos)), feature)
            pos += 0.15


    plt.legend(*zip(*labels),loc=4)
    plt.grid(axis='y')
    plt.xticks([0,1,2])
    plt.xlabel('Clusters 0-2 of 8')
    plt.ylabel('Normalized Card Value')
    plt.title('PKR Cluster Subset vs Norm. Feature Distribution (K-Means)')
    plt.tight_layout()


    plt.savefig('AB_'+title+'_clusters.png') #TODO uncomment
    plt.show()
    plt.close()

def ab_em_cluster_plots(df,title):
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

    pos = np.array(np.sort(df['cluster'].unique()), dtype=float)-0.25
    # pos = np.array(list(range(5)), dtype=float)-0.15
    for feature in list(df.columns):

        if feature in ['price','minimum_nights','number_of_reviews','calculated_host_listings_count']:
        # if feature not in ['cluster']:

            #normaliz

            df[feature] = (df[feature]- df[feature].min())/(df[feature].max()-df[feature].min())
            clst_split = [np.array(df[df['cluster'] == clstr][feature]) for clstr in np.sort(df['cluster'].unique())]
            # clst_split = [np.array(df[df['cluster'] == clstr][feature]) for clstr in list(range(5))]
            # plt.violinplot(clst_split)
            if feature != 'calculated_host_listings_count':
                label = feature
            else:
                label = 'host_listings_count'

            add_label(plt.violinplot(clst_split,widths = 0.2,positions=np.copy(pos)), label)
            pos += 0.15

    plt.legend(*zip(*labels),loc=1)
    plt.grid(axis='y')
    plt.xlabel('Cluster')
    plt.xticks(list(range(3)))
    # plt.xticks(np.sort(df['cluster'].unique()))
    plt.ylabel('Normalized Feature Value')
    plt.title('AB Cluster vs Norm. Feature Distribution (EM)')
    plt.tight_layout()
    plt.show()
    plt.savefig('AB_'+title+'_clusters.png')
    # plt.show()

def pkr_em_cluster_plots(df,title):
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
    plt.savefig('AB_'+title+'_clusters.png') #TODO uncomment
    # plt.show()
    plt.close()


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
    plt.savefig('AB_'+title+'_clusters.png') #TODO uncomment
    # plt.show()
    plt.close()

# ab_km_cluster_plots(km_ab_data,'km')
# pkr_km_cluster_plots(km_pk_data,'km')
# ab_em_cluster_plots(em_ab_data,'em')
# pkr_em_cluster_plots(em_pk_data,'em')
pkr_em_cluster_plots(em_pk_data,'em')
