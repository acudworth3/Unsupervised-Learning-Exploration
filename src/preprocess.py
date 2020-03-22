import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np


class ab_data():
    """ Read and clean Airbnb Data"""
    def __init__(self,n=None):
        self.all = pd.read_csv('AB_NYC_2019.csv',nrows=n)
        self.drop_cols = ['id', 'name','host_id','host_name','last_review','neighbourhood_group','neighbourhood','latitude','longitude','availability_365']
        self.encode = ['room_type']
        self.num2val = {0: "Entire home/apt",1: "Private room",2:"Shared room" } #TODO validate this
        self.title = 'abnb'
        self.target = None
        self.features = None
        self.rand_seed = 105
        self.test_size = 0.2

    def clean(self):
        #influenced by https://www.kaggle.com/chirag9073/airbnb-analysis-visualization-and-prediction
        self.all.drop_duplicates(inplace=True)
        self.all.fillna({'reviews_per_month':0}, inplace=True) #no reviews
        self.all.dropna(how='any', inplace=True) #hosts with no name
        self.all.drop(self.drop_cols, axis=1, inplace=True)

    def init_model_data(self,target = ['price'],features = ['price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month','calculated_host_listings_count']):
        self.target = target
        self.features = features

        for col in self.encode:
            le = preprocessing.LabelEncoder()
            self.all[col] = le.fit_transform(self.all[col])

        self.X = self.all[self.features]
        self.Y = self.all[self.target]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size=self.test_size,
                                                            random_state=self.rand_seed,shuffle=True)




class pkr_data():
    """ Read and clean Poker Data"""
    def __init__(self,n=None):
        #header data for fost processing/testing
        self.all = pd.read_csv('poker-hand-training-true.csv')
        self.head = pd.read_csv('poker-hand-training-true.csv',nrows=None)
        # self.num2val = {0: "Nothing",1: "One pair",2: "Two pairs",3: "Three kind",4: "Straight",5: "Flush",6: "Full house",7: "Four kind",8: "S flush",9: "Royal flush"}
        # self.val2num = {"Nothing": 0, "One pair": 1, "Two pairs": 2, "Three kind": 3, "Straight": 4, "Flush": 5, "Full house": 6,
        self.num2val = {0: "bad_hand",1: "good_hand",2: "vgood_hand",3: "Three kind",4: "Straight",5: "Flush",6: "Full house",7: "Four kind",8: "S flush",9: "Royal flush"}
        self.val2num = {"bad_hand": 0, "good_hand": 1, "vgood_hand": 2, "Three kind": 3, "Straight": 4, "Flush": 5, "Full house": 6,
     "Four kind": 7, "S flush": 8, "Royal flush": 9}
        self.rand_seed = 105
        self.test_size = 0.2
        self.title = 'pkr data'

    def clean(self):
        self.all.dropna(how='any', inplace=True) #null vals

    def init_model_data(self,target =['hand'],features = ['suit1','card1','suit2','card2','suit3','card3','suit4','card4']):
        self.target = target
        self.features = features
        self.X = self.all[self.features]
        self.Y = self.all[self.target]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, np.ravel(self.Y), test_size=self.test_size,
                                                            random_state=self.rand_seed,shuffle=True)


