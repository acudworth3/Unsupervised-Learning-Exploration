import preprocess as prp
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import learning_curve
import pandas as pd
import numpy as np
import plot_learning_curve as plcv
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit

from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from plot_learning_curve import plot_learning_curve
from plot_roc import plot_roc
from plot_precision_recall import plot_precision_recall
from plot_confusion_matrix import local_plot_confusion_matrix


class generic_model():
    """ Read and clean Airbnb Data"""
    def __init__(self,data_obj,clf=None,title="KNN-defualt Learner"):
        # header data for fost processing/testing
        self.title = title
        self.clf = clf
        #Initiate data
        #creates train/test split
        self.data = data_obj
        self.data.clean()
        self.data.init_model_data()
        self.grid_params ={'n_neighbors':[1,2], 'leaf_size':[30,35,40]}
        self.grid_data = None
        self.lrn_crv_chart = None

    # def make_grid_data(self):
    #     clf = GridSearchCV(self.deft_learner, self.grid_params, refit=True)
    #     self.grid_data= clf.fit(self.data.x_train, np.ravel(self.data.y_train))
    #     self.best_learner = clf.best_estimator_
    #     self.grid_results = pd.DataFrame(clf.cv_results_)
    def make_plots(self,roc=True,lrn_crv=True,prec_rec=True,cnf_mtr=True):

        # Learning Curve

        # #ROC
        if roc:
            plot_roc(self.data, self.clf)
        # #Precision Recall
        if prec_rec:
            plot_precision_recall(self.data, self.clf)
        # #confusion matrix
        if cnf_mtr:
            local_plot_confusion_matrix(self.data, self.clf)
        if lrn_crv:
            plot_learning_curve(self.clf, self.title, self.data.x_train, np.ravel(self.data.y_train))


