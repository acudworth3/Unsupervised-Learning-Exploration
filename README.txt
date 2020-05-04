Andrew Cudworth
CS XXXX (Spring 2020)
Assignment 3 Unsupervised Learning
https://www.omscs.gatech.edu/cs-7641-machine-learning


Unupervised Learning Methods-analysis.pdf contains the analysis of learning models

Primary Topics:
Unsupervised Learning
Principle Component Analysis
Independent Component Analysis
Random Projections
Factor Component Analysis

Original Data:
https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data (Airbnb)
https://archive.ics.uci.edu/ml/datasets/Poker+Hand (Poker Data)

processed data is in files:
poker-hand-training-true.csv
AB_NYC_2019.csv

All files must be run with working directory set to src


FILES:
-plot_KM_clstr_range.py  
    -this generates figures 1,2,4,5 analyzing K-Means Cluster Scores
    -data set and cluster count are specified via clst_min and clst_max

-plot_KM_silhouette_analysis.py
    -this generates figures 3 and 6 visualizing clusters
    
-plot_EM_clstr_range.py  
    -this generates figures 9,10,12,13 analyzing EM Cluster Scores
    -data set and cluster count are specified via clst_min and clst_max

-plot_EM_silhouette_analysis.py
    -this generates figures 11 and 14 visualizing clusters

-plot_clstr_violns.py
    -this generates figures 15 and 16 visualizing clusters

-dim_red.py
    -contains functions for dimension reduction analysis
    plot_ICA_Kurt() "Kurtosis of ICA"
    plot_PCA_EV() "Explained variance of PCA" Table 6
    plot_loss() "Reconstruction loss" Table 5
    plot_histos_ICA() "Histograms of ICA" (figures 20 and 21)
    plot_pair_plots() "All pair plots in paper"
    plot_RPA_run_var() "Figures 22 and 23"
    plot_FCA_expl() "Average Noise FCA" Table 8

-re_clstr.py
    -this runs EM and KM analysis on reduced dimension data as discussed in the analysis
    -all plots are populated in re_clstr folder

-plot_NN.py
    -this takes in or generates a re_clstr object and generates an ROC_AUC train/test score
    for all analysis as summarized in Table 9
    -it pickles each NN in the nn_res folder
    
    
FOLDERS    
fig_tables:
-this folder contains the figures and tables generated for the analysis

nn_res:
    -this folder is populated via python plot_NN.py
    -clstr_obj.joblib
        -this object contains all reduced data, reclustering labels and original data        
            -it can be accessed via joblib.load('clstr_obj.joblib')
    -NN_results_XXXX_.joblib
        -these are pickled sklearn MLPclassifiers corresponding to the results
        in table 9
    

sklearn is used heavily and cited throughout analysis. 
All Necessary libraries are in requirenments.txt 

heavily used references:
[9]	Fodor, I. K. (2002). A Survey of Dimension Reduction Techniques. doi: 10.2172/15002155
[4]	Sarkar, T. (2019, September 6). Clustering metrics better than the elbow-method. Retrieved from https://towardsdatascience.com/clustering-metrics-better-than-the-elbow-method-6926e1f723a6


PLAGARISM DISCLAIMER:
If you are currently enrolled in the class this assignment was written for and copy exploration methods, report contents, or hyperparameters
without attribution it is plagiarism. Additionally values in the reports and code have been modified to in a manner specific to this 
repo allowing easy detection of copying.
