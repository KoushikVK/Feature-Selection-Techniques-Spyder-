# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 18:34:18 2021

#feature Selection Techniques

@author: Koushik V
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import streamlit as st

##statistical techniques are used while using feature selection
##implementing everything here

df = pd.read_csv('mobile_dataset.csv')
#univariate selection

X = df.drop(['price_range'],axis =1)

y = df['price_range']

#univariate Analysis

#SelectKBest selects top 'k' features using chi square test 
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
top_features_by_rank = SelectKBest(score_func = chi2,k=20)
top_features_by_rank=top_features_by_rank.fit(X,y)
top_features = pd.DataFrame(top_features_by_rank.scores_,columns =['Scores'])
top_features.index = X.columns
top_features.reset_index(inplace =True)

##give me top 10 values
top_features.nlargest(10,'Scores')

#Feature Importance gives you a score for each feature , the higher the score more important  it is

from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X,y)
extraTreesRank = pd.DataFrame(model.feature_importances_) 
extraTreesRank['Features'] = X.columns
extraTreesRank.rename({0:'Scores'},axis = 1,inplace = True)
# =============================================================================
# ##rearanging columns
# 
# column_names = ["Scores","Features"]
# 
# 
# extraTreesRank = extraTreesRank.reindex(columns=column_names)
# =============================================================================

extraTreesRank.nlargest(10,'Scores').sort_values(by='Scores',ascending = True).plot(kind= 'barh')
# =============================================================================
#correlation technique
from sklearn.model_selection import train_test_split
plt.figure(figsize = (18,18))
sns.heatmap(df.iloc[:,:-1].corr(), annot = True, cmap = "RdYlGn")

#setting a threshhold

threshold = 0.8  ##threshold may differ as per requirements
# find and remove correlated features
def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr
#Apply function
correlation(df.iloc[:,:-1].corr(),threshold)

#{'pc', 'three_g'}

# =============================================================================


#Using Information Gain 
from sklearn.feature_selection import mutual_info_classif
mutual_info = pd.Series(mutual_info_classif(X,y),index = X.columns)
mutual_info.sort_values(ascending = False)




































# =============================================================================
# #check duplicated columns
# 
# #df.columns.duplicated()
# 
# 
# 
# =============================================================================




















