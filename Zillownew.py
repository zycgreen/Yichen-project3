#!/usr/bin/env python
# coding: utf-8

# # Kaggle Project - Zillow 

# Blah blah

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import xgboost as xgb


# In[2]:


from sklearn import neighbor
from collections import Counter
import math
import time
from catboost import CatBoostRegressor


# # Overview

# Blah Blah

# # Read Data

# blah blah

# In[2]:


properties_16 = pd.read_csv('C:/Users/70785/Downloads/zillow-prize-1/properties_2016.csv', header = 0)
properties_17 = pd.read_csv('C:/Users/70785/Downloads/zillow-prize-1/properties_2017.csv', header = 0)


# In[3]:


properties_16.head()


# In[4]:


properties_16.shape


# In[5]:


properties_17.shape


# In[6]:


train_16 = pd.read_csv('C:/Users/70785/Downloads/zillow-prize-1/train_2016_v2.csv', header = 0)
train_17 = pd.read_csv('C:/Users/70785/Downloads/zillow-prize-1/train_2017.csv', header = 0)


# In[7]:


train_16.tail()


# In[8]:


train_16.shape


# In[9]:


train_17.shape


# # Correlations for Preperties_16

# In[10]:


#Identify numerical columns to produce a heatmap
catcols = ['parcelid','airconditioningtypeid','architecturalstyletypeid','buildingqualitytypeid',
           'buildingclasstypeid','decktypeid','fips','hashottuborspa','heatingorsystemtypeid','pooltypeid10',
           'pooltypeid2','pooltypeid7','propertycountylandusecode','propertylandusetypeid','propertyzoningdesc',
           'rawcensustractandblock','regionidcity','regionidcounty','regionidneighborhood','regionidzip',
           'storytypeid','typeconstructiontypeid','yearbuilt','taxdelinquencyflag']
numcols = [x for x in properties_16.columns if x not in catcols]

plt.figure(figsize = (12,8))
sns.heatmap(data=properties_16[numcols].corr())
plt.show()
plt.gcf().clear()


# In[11]:


numcols = [x for x in properties_17.columns if x not in catcols]

plt.figure(figsize = (12,8))
sns.heatmap(data=properties_16[numcols].corr())
plt.show()
plt.gcf().clear()


# 'calculatedfinishedsquarefeet' 'finishedsquarefeet12' 'finishedsquarefeet13' 'finishedsquarefeet15' 'finishedsquarefeet6'are all very strongly correlated, according to the heatmap.
# 
# 'structuretaxvaluedollarcnt' 'taxvaluedollarcnt' 'landtaxvaluedollarcnt' and 'taxamount'are also very strongly correlated, according to the heatmap.

# # Missing values

# In[12]:


#Check missing value in column dimension
col_missing_16 = properties_16.isnull().sum(axis=0)/properties_16.shape[0]
col_missing_17 = properties_17.isnull().sum(axis=0)/properties_17.shape[0]


# In[13]:


plt.figure(figsize=(20,10))
y_pos = np.arange(len(col_missing_16.index)) 
plt.bar(y_pos, col_missing_16)
plt.xticks(y_pos, col_missing_16.index, ha='right', rotation=55, fontsize=13, fontname='monospace')
plt.title('Features vs Missing Value Percentage')
plt.show()


# In[14]:


plt.figure(figsize=(20,10))
y_pos = np.arange(len(col_missing_17.index))
plt.bar(y_pos, col_missing_17)
plt.xticks(y_pos, col_missing_17.index, ha='right', rotation=55, fontsize=13, fontname='monospace')
plt.title('Features vs Missing Value Percentage')
plt.show()


# In[15]:


# row_missing = z_16.isnull().sum(axis=1)/z_16.shape[1]
# row_missing[row_missing>0.8]


# ## Removing redundant features

# In[16]:


#'calculatedfinishedsquarefeet' 'finishedsquarefeet12' 'finishedsquarefeet13' 'finishedsquarefeet15' 'finishedsquarefeet6' 
# They have really similar information based on the dictionary description
#'calculatedfinishedsquarefeet' has the fewest missing values
#Plus, 'finishedsquarefeet13' 'finishedsquarefeet15' 'finishedsquarefeet6' all have more than 96% missing values.
print(col_missing_16['finishedsquarefeet13'], col_missing_16['finishedsquarefeet15'], col_missing_16['finishedsquarefeet6'])
dropcols = ['finishedsquarefeet13', 'finishedsquarefeet15','finishedsquarefeet6']

#'finishedsquarefeet50' and 'finishedfloor1squarefeet' have exactly the same information according to the dictionary descriptions
#Plus, 'finishedsquarefeet50' and 'finishedfloor1squarefeet' have exactly the same amount of missing values.
print(col_missing_16['finishedsquarefeet50'], col_missing_16['finishedfloor1squarefeet'])
dropcols.append('finishedsquarefeet50')

#'bathroomcnt' and 'calculatedbathnbr' and 'fullbathcnt' have exactly the same information according to the dictionary descriptions
# Plus, only 'bathroomcnt' has least missing values.
print(col_missing_16['bathroomcnt'], col_missing_16['calculatedbathnbr'], col_missing_16['fullbathcnt'])
dropcols.append('calculatedbathnbr')
dropcols.append('fullbathcnt')


# ## Pool Features

# In[17]:


#'pooltypeid10' seems to be inconcistent with the 'hashottuborspa' 
#which should have the same information according to the dictionary descriptions
# print(properties_16.pooltypeid10.value_counts())
#check if 'pooltypeid10' contains any new information
index = properties_16.hashottuborspa.isnull()
index2 = properties_16.pooltypeid10.isnull()
properties_16.loc[index & ~index2,'hashottuborspa'] = 1
# print(properties_16['hashottuborspa'].value_counts())
#'pooltypeid10' does not contain any new information
index = properties_17.hashottuborspa.isnull()
index2 = properties_17.pooltypeid10.isnull()
properties_17.loc[index & ~index2,'hashottuborspa'] = 1
dropcols.append('pooltypeid10')

#NAs in 'hashottuborspa' means the there is no SPA or Hottub
index = properties_16.hashottuborspa.isnull()
properties_16.loc[index,'hashottuborspa'] = False
index = properties_17.hashottuborspa.isnull()
properties_17.loc[index,'hashottuborspa'] = False

#If 'pooltypeid's are null then pool/hottub doesnt exist
# print(properties_16.pooltypeid2.value_counts())
index = properties_16.pooltypeid2.isnull()
properties_16.loc[index,'pooltypeid2'] = 0
index = properties_17.pooltypeid2.isnull()
properties_17.loc[index,'pooltypeid2'] = 0

# print(properties_16.pooltypeid7.value_counts())
index = properties_16.pooltypeid7.isnull()
properties_16.loc[index,'pooltypeid7'] = 0
index = properties_17.pooltypeid7.isnull()
properties_17.loc[index,'pooltypeid7'] = 0

# print(properties_16.poolcnt.value_counts())
index = properties_16.poolcnt.isnull()
properties_16.loc[index,'poolcnt'] = 0
index = properties_17.poolcnt.isnull()
properties_17.loc[index,'poolcnt'] = 0

#Fill in median values for 'poolsizesum' missing values where 'poolcnt' is >0.
poolsizesum_median_16 = properties_16.loc[properties_16['poolcnt'] > 0, 'poolsizesum'].median()
properties_16.loc[(properties_16['poolcnt'] > 0) & (properties_16['poolsizesum'].isnull()), 'poolsizesum'] = poolsizesum_median_16
poolsizesum_median_17 = properties_17.loc[properties_17['poolcnt'] > 0, 'poolsizesum'].median()
properties_17.loc[(properties_17['poolcnt'] > 0) & (properties_17['poolsizesum'].isnull()), 'poolsizesum'] = poolsizesum_median_17

#If 'poolcnt' is 0, then 'poolsizesum' is 0
properties_16.loc[(properties_16['poolcnt'] == 0), 'poolsizesum'] = 0
properties_17.loc[(properties_17['poolcnt'] == 0), 'poolsizesum'] = 0


# ## Fireplace features

# In[18]:


#'fireplaceflag' seems to be inconcistent with the 'fireplacecnt' 
#which should have the same information according to the dictionary descriptions
# print(properties_16.fireplaceflag.value_counts())
# print(properties_16.fireplacecnt.value_counts())

#Fix 'fireplaceflag' according to 'fireplaceflag'
properties_16['fireplaceflag']= "No"
properties_16.loc[properties_16['fireplacecnt']>0,'fireplaceflag']= "Yes"
properties_17['fireplaceflag']= "No"
properties_17.loc[properties_17['fireplacecnt']>0,'fireplaceflag']= "Yes"

#If 'fireplacecnt' is null then fireplace doesnt exist
index = properties_16.fireplacecnt.isnull()
properties_16.loc[index,'fireplacecnt'] = 0
index = properties_17.fireplacecnt.isnull()
properties_17.loc[index,'fireplacecnt'] = 0


# ## Garage features

# In[19]:


#If 'garagecarcnt' is null then garbage doesnt exist
index = properties_16.garagecarcnt.isnull()
properties_16.loc[index,'garagecarcnt'] = 0
index = properties_17.garagecarcnt.isnull()
properties_17.loc[index,'garagecarcnt'] = 0

#If 'garagetotalsqft' is null then garbage doesnt exist
index = properties_16.garagetotalsqft.isnull()
properties_16.loc[index,'garagetotalsqft'] = 0
index = properties_17.garagetotalsqft.isnull()
properties_17.loc[index,'garagetotalsqft'] = 0


# ## Tax Features

# In[20]:


#If 'taxdelinquencyflag' is null then tax delinquency doesn't exist
# print(properties_16.taxdelinquencyflag.value_counts())
index = properties_16.taxdelinquencyflag.isnull()
properties_16.loc[index,'taxdelinquencyflag'] = 'N'
index = properties_17.taxdelinquencyflag.isnull()
properties_17.loc[index,'taxdelinquencyflag'] = 'N'

#If 'structuretaxvaluedollarcnt' is null then structure tax doesnt exist
properties_16.loc[properties_16['structuretaxvaluedollarcnt'].isnull(), 'structuretaxvaluedollarcnt'] = 0
properties_17.loc[properties_17['structuretaxvaluedollarcnt'].isnull(), 'structuretaxvaluedollarcnt'] = 0

#Fill in mean values for 'landtaxvaluedollarcnt' missing values where 'landtaxvaluedollarcnt' is >0.
landtaxvalue_mean_16 = properties_16.loc[~properties_16['landtaxvaluedollarcnt'].isnull(), 'landtaxvaluedollarcnt'].mean()
properties_16.loc[properties_16['landtaxvaluedollarcnt'].isnull(), 'landtaxvaluedollarcnt'] = landtaxvalue_mean_16
landtaxvalue_mean_17 = properties_17.loc[~properties_17['landtaxvaluedollarcnt'].isnull(), 'landtaxvaluedollarcnt'].mean()
properties_17.loc[properties_17['landtaxvaluedollarcnt'].isnull(), 'landtaxvaluedollarcnt'] = landtaxvalue_mean_17

#Fill in mean values for 'taxamount' missing values where 'taxamount' is >0.
tax_mean_16 = properties_16.loc[~properties_16['taxamount'].isnull(), 'taxamount'].mean()
properties_16.loc[properties_16['taxamount'].isnull(), 'taxamount'] = tax_mean_16
tax_mean_17 = properties_17.loc[~properties_17['taxamount'].isnull(), 'taxamount'].mean()
properties_17.loc[properties_17['taxamount'].isnull(), 'taxamount'] = tax_mean_17

#'structuretaxvaluedollarcnt' + 'landtaxvaluedollarcnt' = 'taxvaluedollarcnt' 
#Remove 'taxvaluedollarcnt'
dropcols.append('taxvaluedollarcnt')


# ## Other features 

# 'threequarterbathnbr'

# In[21]:


# print(properties_16['threequarterbathnbr'].value_counts())

#Fill in the NAs with majority which is 1 in 'threequarterbathnbr'
index = properties_16.threequarterbathnbr.isnull()
properties_16.loc[index,'threequarterbathnbr'] = 1

index = properties_17.threequarterbathnbr.isnull()
properties_17.loc[index,'threequarterbathnbr'] = 1


# 'heatingorsystemtypeid'

# In[22]:


# print(properties_16['heatingorsystemtypeid'].value_counts())

#Fill in the NAs with majority which is 2 in 'heatingorsystemtypeid'
index = properties_16.heatingorsystemtypeid.isnull()
properties_16.loc[index,'heatingorsystemtypeid'] = 2
index = properties_17.heatingorsystemtypeid.isnull()
properties_17.loc[index,'heatingorsystemtypeid'] = 2


# 'airconditioningtypeid'
# 

# In[23]:


# print(properties_16['airconditioningtypeid'].value_counts())

#Fill in the NAs with majority which is 1 in 'airconditioningtypeid'
index = properties_16.airconditioningtypeid.isnull()
properties_16.loc[index,'airconditioningtypeid'] = 1
index = properties_17.airconditioningtypeid.isnull()
properties_17.loc[index,'airconditioningtypeid'] = 1


# ## Geo features

# ### Use KNN to fill in NAs

# In[24]:


# # geocolumns = ['parcelid','latitude', 'longitude','propertycountylandusecode', 'propertylandusetypeid',
# #               'propertyzoningdesc','regionidcity','regionidcounty', 'regionidneighborhood', 
# #               'regionidzip','censustractandblock', 'rawcensustractandblock']
# # geo = z_16[geocolumns]
# # geo.dropna(axis = 0, subset = ['latitude', 'longitude'], inplace = True)
# def fillna_knn(df, base, target, fraction = 1, threshold = 10, n_neighbors = 5):
#     assert isinstance(base, list) or isinstance(base, np.ndarray) and isinstance(target, str) 
#     whole = [target] + base
    
#     miss = df[target].isnull()
#     notmiss = ~miss 
#     nummiss = miss.sum()
    
#     enc = OneHotEncoder()
#     X_target = df.loc[notmiss, whole].sample(frac = fraction)
    
#     enc.fit(X_target[target].unique().reshape((-1,1)))
    
#     Y = enc.transform(X_target[ target ].values.reshape((-1,1))).toarray()
#     X = X_target[base]
    
#     print('fitting')
#     n_neighbors = n_neighbors
#     clf = neighbors.KNeighborsClassifier(n_neighbors, weights = 'uniform')
#     clf.fit( X, Y )
    
#     print('the shape of active features: ',enc.active_features_.shape)
    
#     print('predicting')
#     Z = clf.predict(df.loc[miss, base])
    
#     numunperdicted = Z[:,0].sum()
#     if numunperdicted / nummiss *100 < threshold :
#         print('writing result to df')    
#         df.loc[miss, target]  = np.dot(Z , enc.active_features_)
#         print('num of unperdictable data: ', numunperdicted)
#         return enc
#     else:
#         print('out of threshold: {}% > {}%'.format( numunperdicted / nummiss *100 , threshold ))

# #function to deal with variables that are actually string/categories
# def zoningcode2int( df, target ):
#     storenull = df[target].isnull()
#     enc = LabelEncoder( )
#     df[target] = df[ target ].astype( str )

#     print('fit and transform')
#     df[target]= enc.fit_transform(df[target].values)
#     print('num of categories: ', enc.classes_.shape)
#     df.loc[storenull, target] = np.nan
#     print('recover the nan value')
#     return enc


# In[25]:


# properties_16_knn = properties_16.copy()
# a = properties_16_knn.dropna(axis = 0, subset = ['latitude', 'longitude'])


# In[26]:


# #Fill 'regionidzip'
# fillna_knn(df = a, base = ['latitude', 'longitude'] ,
#            target = 'regionidzip', fraction = 1, n_neighbors = 1)

# #Fill 'propertycountylandusecode'
# zoningcode2int(df = a, target = 'propertycountylandusecode')

# fillna_knn(df = a, base = ['latitude', 'longitude'],
#            target = 'propertycountylandusecode', fraction = 1, n_neighbors = 1)


# In[ ]:





# In[ ]:





# ### Use Clustering to fill in NAs

# In[27]:


def fillna_clustering( df, base, target, threshold = 10, n_clusters = 5 ):
    assert isinstance( base , list ) or isinstance( base , np.ndarray ) and isinstance( target, str ) 
    whole = [target] + base
    
    miss = df[target].isnull()
    notmiss = ~miss 
    nummiss = miss.sum()
    missing_idx = df.loc[miss,target].index
    
    X = df.loc[notmiss, base]
    
    print('fitting')
    kmc = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    
    print( 'predicting' )
    Z = kmc.predict(df.loc[miss, base])
    
    numunperdicted = Z.shape[0]
    print( 'writing result to df' )
    id_label=kmc.labels_
    pred = pd.DataFrame(data={'cluster': Z}) 
    pred[target] = 0
    uniq_label=np.unique(Z)
    for i in uniq_label:
        a = np.where(id_label==i)
        c = Counter(df[target][a[0]])
        if(isinstance(c.most_common(1)[0][0], str)):
            pred.loc[pred['cluster'] == i,target] = c.most_common(1)[0][0]
        elif(math.isnan(c.most_common(1)[0][0])):
            pred.loc[pred['cluster'] == i,target] = c.most_common(2)[1][0]
        else:
            pred.loc[pred['cluster'] == i,target] = c.most_common(1)[0][0]
    df[target][missing_idx] = pred[target]
    #print(df[target].isnull().sum())
    print( 'num of unperdictable data: ', numunperdicted )


# In[28]:


# z_16_clustering = z_16.copy()

# df = z_16_clustering
# base = ['latitude', 'longitude']
# target = 'propertyzoningdesc'
# n_clusters = 20


# miss = df[target].isnull()
# notmiss = ~miss 
# nummiss = miss.sum()
# missing_idx = df.loc[miss,target].index

# X = df.loc[notmiss, base]
    
# # print('fitting')
# kmc = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
# id_label=kmc.labels_

# # print( 'predicting' )
# Z = kmc.predict(df.loc[miss, base])
# pred = pd.DataFrame(data={'cluster': Z}) 
# pred[target] = 0

# uniq_label=np.unique(Z)
# #i = uniq_label[1]
# for i in uniq_label:
#     a = np.where(id_label==i)
#     c = Counter(df[target][a[0]])
#     if(isinstance(c.most_common(1)[0][0], str)):
#         pred.loc[pred['cluster'] == i,target] = c.most_common(1)[0][0]
#     elif(math.isnan(c.most_common(1)[0][0])):
#         pred.loc[pred['cluster'] == i,target] = c.most_common(2)[1][0]
#     else:
#         pred.loc[pred['cluster'] == i,target] = c.most_common(1)[0][0]

# df[target][missing_idx] = pred[target]
# df[target].isnull().sum()


# In[29]:


properties_16_clustering = properties_16.copy()
b_16 = properties_16_clustering.dropna(axis = 0, subset = ['latitude', 'longitude'])
properties_17_clustering = properties_17.copy()
b_17 = properties_17_clustering.dropna(axis = 0, subset = ['latitude', 'longitude'])


# In[30]:


#Fill 'regionidzip'
fillna_clustering(df = b_16, base = ['latitude', 'longitude'] ,
                  target = 'regionidzip', n_clusters = 8)
fillna_clustering(df = b_17, base = ['latitude', 'longitude'] ,
                  target = 'regionidzip', n_clusters = 8)

#Fill 'propertycountylandusecode'
fillna_clustering(df = b_16, base = ['latitude', 'longitude'] ,
                  target = 'propertycountylandusecode', n_clusters = 8)
fillna_clustering(df = b_17, base = ['latitude', 'longitude'] ,
                  target = 'propertycountylandusecode', n_clusters = 8)


# In[31]:


latitude_flag = ~properties_16_clustering['latitude'].isnull()
longitude_flag = ~properties_16_clustering['longitude'].isnull()
properties_16_clustering.loc[latitude_flag & longitude_flag, 'regionidzip'] = b_16['regionidzip']
properties_16_clustering.loc[latitude_flag & longitude_flag, 'propertycountylandusecode'] = b_16['propertycountylandusecode']

latitude_flag = ~properties_17_clustering['latitude'].isnull()
longitude_flag = ~properties_17_clustering['longitude'].isnull()
properties_17_clustering.loc[latitude_flag & longitude_flag, 'regionidzip'] = b_17['regionidzip']
properties_17_clustering.loc[latitude_flag & longitude_flag, 'propertycountylandusecode'] = b_17['propertycountylandusecode']


# ## Getting rid of features with missing value > 97%?

# In[32]:


#a = properties_16_knn.drop(dropcols, axis=1)
b_16 = properties_16_clustering.drop(dropcols, axis=1)
col_missing_16 = b_16.isnull().sum(axis=0)/b_16.shape[0]
b_17 = properties_17_clustering.drop(dropcols, axis=1)
col_missing_17 = b_17.isnull().sum(axis=0)/b_17.shape[0]
# print(col_missing[col_missing>0.5])
missing_more_than_97 = ['architecturalstyletypeid', 'basementsqft', 'buildingclasstypeid', 'decktypeid', 
                        'storytypeid', 'typeconstructiontypeid', 'yardbuildingsqft17','yardbuildingsqft26',
                        'taxdelinquencyyear']
#properties_16_knn = a.drop(missing_more_than_97, axis=1)
properties_16_clustering = b_16.drop(missing_more_than_97, axis=1)
properties_17_clustering = b_17.drop(missing_more_than_97, axis=1)


# # Categorical with Multiple Levels¶

# In[33]:


#All the categorical features
cat_col = ['propertycountylandusecode', 'propertyzoningdesc', 'airconditioningtypeid', 'buildingqualitytypeid',
          'heatingorsystemtypeid', 'pooltypeid2', 'pooltypeid7', 'propertylandusetypeid', 'regionidcity',
          'regionidcounty', 'regionidneighborhood', 'regionidzip', 'hashottuborspa', 'fireplaceflag',
           'taxdelinquencyflag']
#Check their levels
for i in cat_col:
    print (i, len(properties_16_clustering[i].value_counts()))
    print (i, len(properties_17_clustering[i].value_counts()))


# ## 'propertycountylandusecode'

# In[34]:


#According to http://www.titleadvantage.com/mdocs/LA%20Use%20Codes.pdf and https://www.ocpafl.org/Searches/Lookups.aspx/Code/PropertyUse
properties_16_clustering['propertycountylandusecode'] = [str(i) for i in properties_16_clustering['propertycountylandusecode']]

uniq_code_16=np.unique(properties_16_clustering['propertycountylandusecode'])
for i in uniq_code_16:
    if i.startswith('0'):
        if i.startswith('00'):
            properties_16_clustering.loc[properties_16_clustering['propertycountylandusecode'] == i, 'propertycountylandusecode'] = 'Vacant'
        elif i.startswith('01'):
            properties_16_clustering.loc[properties_16_clustering['propertycountylandusecode'] == i, 'propertycountylandusecode'] = 'Single Residence'
        else:
            properties_16_clustering.loc[properties_16_clustering['propertycountylandusecode'] == i, 'propertycountylandusecode'] = 'Multi-family Residence'
    else:
        properties_16_clustering.loc[properties_16_clustering['propertycountylandusecode'] == i, 'propertycountylandusecode'] = 'Commercial'
          
properties_17_clustering['propertycountylandusecode'] = [str(i) for i in properties_17_clustering['propertycountylandusecode']]
uniq_code_17=np.unique(properties_17_clustering['propertycountylandusecode'])
for i in uniq_code_17:
    if i.startswith('0'):
        if i.startswith('00'):
            properties_17_clustering.loc[properties_17_clustering['propertycountylandusecode'] == i, 'propertycountylandusecode'] = 'Vacant'
        elif i.startswith('01'):
            properties_17_clustering.loc[properties_17_clustering['propertycountylandusecode'] == i, 'propertycountylandusecode'] = 'Single Residence'
        else:
            properties_17_clustering.loc[properties_17_clustering['propertycountylandusecode'] == i, 'propertycountylandusecode'] = 'Multi-family Residence'
    else:
        properties_17_clustering.loc[properties_17_clustering['propertycountylandusecode'] == i, 'propertycountylandusecode'] = 'Commercial'
          


# In[35]:


properties_16_clustering['propertycountylandusecode'].value_counts()


# ## 'propertyzoningdesc'

# In[36]:


#Accoding to the https://planning.lacity.org/zone_code/Appendices/sum_of_zone.pdf 
#you get almost the same information as 'propertycountylandusecode' if you reduce its levels
properties_16_clustering.drop('propertyzoningdesc', axis=1, inplace=True)
properties_17_clustering.drop('propertyzoningdesc', axis=1, inplace=True)


# ##  'regionidcity' and 'regionidneighborhood'

# In[37]:


#You get almost the same information as 'regionidzip' if you reduce its levels
properties_16_clustering.drop(['regionidcity', 'regionidneighborhood'], axis=1, inplace=True)
properties_17_clustering.drop(['regionidcity', 'regionidneighborhood'], axis=1, inplace=True)


# ## 'regionidzip'

# In[38]:


properties_16_clustering['regionidzip'] = [str(i) for i in properties_16_clustering['regionidzip']]
properties_17_clustering['regionidzip'] = [str(i) for i in properties_17_clustering['regionidzip']]


# In[39]:


properties_16_clustering['regionidzip'] = properties_16_clustering['regionidzip'].apply(lambda x: x[1]+x[-3])
properties_17_clustering['regionidzip'] = properties_17_clustering['regionidzip'].apply(lambda x: x[1]+x[-3])


# In[40]:


properties_16_clustering['regionidzip'].value_counts()


# # Impute rest of the NAs

# In[41]:


properties_16_clustering.isnull().sum()


# In[42]:


#Impute NA with -1
properties_16_clustering = properties_16_clustering.fillna(-1)
properties_17_clustering = properties_17_clustering.fillna(-1)


# missing value final check

# In[43]:


#final_z_16_knn.isnull().sum()


# In[44]:


sum(properties_16_clustering.isnull().sum())


# In[45]:


sum(properties_17_clustering.isnull().sum())


# # Merge data

# In[46]:


z_16_clustering = pd.merge(train_16, properties_16_clustering,on = 'parcelid',how = 'left')
z_17_clustering = pd.merge(train_17, properties_17_clustering,on = 'parcelid',how = 'left')


# check missing values

# In[47]:


sum(z_16_clustering.isnull().sum())


# ## 'transactiondate' make items into datetime object

# In[48]:


z_16_clustering['transactiondate'] = z_16_clustering['transactiondate'].apply(
        lambda x: datetime.strptime(x,'%Y-%m-%d'))
z_17_clustering['transactiondate'] = z_17_clustering['transactiondate'].apply(
        lambda x: datetime.strptime(x,'%Y-%m-%d'))


# ## Creat 'year' and 'month' features

# In[49]:


z_16_clustering['year'] = z_16_clustering['transactiondate'].dt.year
z_16_clustering['month'] = z_16_clustering['transactiondate'].dt.month
z_17_clustering['year'] = z_17_clustering['transactiondate'].dt.year
z_17_clustering['month'] = z_17_clustering['transactiondate'].dt.month


# In[50]:


#Remove 'transactiondate'
z_16_clustering.drop('transactiondate', axis=1, inplace=True)
z_17_clustering.drop('transactiondate', axis=1, inplace=True)


# # Correlations for z_16_clustering

# ## Numerical features

# In[51]:


# get correlation of all numerical variable
correlation = z_16_clustering.select_dtypes(include = [np.number]).corr()
print(correlation['logerror'].sort_values(ascending = False))


# ## Categorical features - Boxplot

# In[ ]:





# # Outliers

# In[52]:


cat_col = ['propertycountylandusecode', 'airconditioningtypeid', 'buildingqualitytypeid',
          'heatingorsystemtypeid', 'pooltypeid2', 'pooltypeid7', 'propertylandusetypeid',
          'regionidcounty', 'regionidzip', 'hashottuborspa', 'fireplaceflag', 'taxdelinquencyflag']
num_col = z_16_clustering.columns.difference(cat_col)
# train_df=train_df[ train_df.logerror > -0.4 ]
# train_df=train_df[ train_df.logerror < 0.419 ]




# list of features without intervention

# count    8.871900e+04
# mean     6.033366e+13
# std      3.096115e+12
# min     -1.000000e+00
# 50%      6.037604e+13
# 99.8%    6.111008e+13
# max      6.111009e+13
# Name: censustractandblock, dtype: float64


# count    88273.000000
# mean      6048.725216
# std         20.440648
# min       6037.000000
# 50%       6037.000000
# 99.8%     6111.000000
# max       6111.000000
# Name: fips, dtype: float64


# count    88273.000000
# mean         0.122302
# std          0.382543
# min          0.000000
# 50%          0.000000
# 99.8%        3.000000
# max          4.000000
# Name: fireplacecnt, dtype: float64



# count    8.762600e+04
# mean     3.400580e+07
# std      2.647829e+05
# min      3.333930e+07
# 50%      3.401991e+07
# 99.8%    3.470700e+07
# max      3.481601e+07
# Name: latitude, dtype: float64


# count    8.718500e+04
# mean    -1.181964e+08
# std      3.577270e+05
# min     -1.194479e+08
# 0.2%    -1.192825e+08
# 50%     -1.181708e+08
# 99.8%   -1.175798e+08
# max     -1.175549e+08
# Name: longitude, dtype: float64




# count    86976.000000
# mean         5.862560
# std          2.811006
# min          1.000000
# 0.2%         1.000000
# 50%          6.000000
# 99.8%       12.000000
# max         12.000000
# Name: month, dtype: float64


# count    86976.000000
# mean        -0.443295
# std          1.056214
# min         -1.000000
# 0.2%        -1.000000
# 50%         -1.000000
# 99.8%        3.000000
# max          4.000000
# Name: numberofstories, dtype: float64


# count    8.697600e+04
# mean     1.297369e+07
# std      2.520129e+06
# min      1.071174e+07
# 0.2%     1.072128e+07
# 50%      1.254645e+07
# 99.8%    1.729324e+07
# max      1.629608e+08
# Name: parcelid, dtype: float64



# count    86976.000000
# mean         0.193065
# std          0.394706
# min          0.000000
# 0.2%         0.000000
# 50%          0.000000
# 99.8%        1.000000
# max          1.000000
# Name: poolcnt, dtype: float64



# count    8.675800e+04
# mean     6.048829e+07
# std      2.004896e+05
# min      6.037101e+07
# 0.2%     6.037102e+07
# 50%      6.037603e+07
# 99.8%    6.111008e+07
# max      6.111009e+07
# Name: rawcensustractandblock, dtype: float64



# count    86471.0
# mean      2016.0
# std          0.0
# min       2016.0
# 0.2%      2016.0
# 50%       2016.0
# 99.8%     2016.0
# max       2016.0
# Name: year, dtype: float6


# count    86471.000000
# mean      1966.047253
# std         70.018119
# min         -1.000000
# 0.2%      1901.000000
# 50%       1969.000000
# 99.8%     2013.000000
# max       2015.000000
# Name: yearbuilt, dtype: float64



import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')



# z_16_clustering.head
num_col




# count    90275.000000
# mean         2.279474
# std          1.004271
# min          0.000000
# 50%          2.000000
# 99.8%        7.000000
# max         20.000000
# Name: bathroomcnt, dtype: float64

# upper_bnd = z_16_clustering[num_col[idx]].quantile(0.9975)


idx = 1
# print(num_col[idx])
# print(z_16_clustering.shape)
# z_16_clustering[num_col[idx]].hist(bins=30)
# z_16_clustering[num_col[idx]].describe(percentiles=[.9975])
upper_bnd = 7.1
z_16_clustering = z_16_clustering[z_16_clustering[num_col[idx]] <  upper_bnd]

# z_16_clustering[num_col[idx]].hist(bins=30)


# bedroomcnt - remove outliers that have > 8 bedrooms
idx = 2
# print(num_col[idx])
# print(z_16_clustering.shape)
# z_16_clustering[num_col[idx]].hist(bins=30)
# z_16_clustering[num_col[idx]].describe(percentiles=[.9975])
upper_bnd = 8.1
z_16_clustering = z_16_clustering[z_16_clustering[num_col[idx]] <  upper_bnd]
# z_16_clustering[num_col[idx]].hist(bins=30)





# calculatedfinishedsquarefeet removed outliers > 6131 and < 0

# count    89605.000000
# mean      1734.612979
# std        867.362040
# min         -1.000000
# 50%       1528.000000
# 99.8%     6130.960000
# max      22741.000000
# Name: calculatedfinishedsquarefeet, dtype: float64


idx = 3
# print(num_col[idx])
# print(z_16_clustering.shape)
# z_16_clustering[num_col[idx]].hist(bins=30)
# z_16_clustering[num_col[idx]].describe(percentiles=[.9975])
upper_bnd = 6131
lower_bnd = 0
z_16_clustering = z_16_clustering[z_16_clustering[num_col[idx]] <  upper_bnd]
z_16_clustering = z_16_clustering[z_16_clustering[num_col[idx]] >  lower_bnd]
# z_16_clustering[num_col[idx]].hist(bins=30)


# finishedfloor1squarefeet - removed outliers > 2738
# count    88719.000000
# mean       101.755216
# std        394.466879
# min         -1.000000
# 50%         -1.000000
# 99.8%     2738.410000
# max       5701.000000
# Name: finishedfloor1squarefeet, dtype: float64



idx = 5
# print(num_col[idx])
# print(z_16_clustering.shape)
# z_16_clustering[num_col[idx]].hist(bins=30)
# z_16_clustering[num_col[idx]].describe(percentiles=[.9975])
upper_bnd = 2738
z_16_clustering = z_16_clustering[z_16_clustering[num_col[idx]] <  upper_bnd]
# z_16_clustering[num_col[idx]].hist(bins=30)

# z_16_clustering[num_col[idx]].value_counts()


# finishedsquarefeet12 - removed outliers > 5383
# count    88495.000000
# mean      1640.680016
# std        849.893801
# min         -1.000000
# 50%       1479.000000
# 99.8%     5382.530000
# max       6127.000000
# Name: finishedsquarefeet12, dtype: float64


idx = 6
# print(num_col[idx])
# print(z_16_clustering.shape)
# z_16_clustering[num_col[idx]].hist(bins=30)
# z_16_clustering[num_col[idx]].describe(percentiles=[.9975])
upper_bnd = 5383
z_16_clustering = z_16_clustering[z_16_clustering[num_col[idx]] <  upper_bnd]
# z_16_clustering[num_col[idx]].hist(bins=30)




# garagecarcnt - removed outliers > 4

# count    88273.000000
# mean         0.597782
# std          0.899559
# min          0.000000
# 50%          0.000000
# 99.8%        3.000000
# max         13.000000
# Name: garagecarcnt, dtype: float64

idx = 9
# print(num_col[idx])
# print(z_16_clustering.shape)
# z_16_clustering[num_col[idx]].hist(bins=30)
# z_16_clustering[num_col[idx]].describe(percentiles=[.9975])
upper_bnd = 4.1
z_16_clustering = z_16_clustering[z_16_clustering[num_col[idx]] <  upper_bnd]
# z_16_clustering[num_col[idx]].hist(bins=30)



# garagetotalsqft - removed outliers > 768
# count    88068.000000
# mean       110.349957
# std        209.241486
# min          0.000000
# 50%          0.000000
# 99.8%      768.000000
# max        878.000000
# Name: garagetotalsqft, dtype: float64


idx = 10
# print(num_col[idx])
# print(z_16_clustering.shape)
# z_16_clustering[num_col[idx]].hist(bins=30)
# z_16_clustering[num_col[idx]].describe(percentiles=[.9975])
upper_bnd = 768
z_16_clustering = z_16_clustering[z_16_clustering[num_col[idx]] <  upper_bnd]
# z_16_clustering[num_col[idx]].hist(bins=30)





# landtaxvaluedollarcnt - removed outliers > 2363546
# count    8.784600e+04
# mean     2.637081e+05
# std      3.244894e+05
# min      2.200000e+01
# 50%      1.904020e+05
# 99.8%    2.363546e+06
# max      1.315000e+07
# Name: landtaxvaluedollarcnt, dtype: float64

idx = 11
# print(num_col[idx])
# print(z_16_clustering.shape)
# z_16_clustering[num_col[idx]].hist(bins=30)
# z_16_clustering[num_col[idx]].describe(percentiles=[.9975])
upper_bnd = 2363546
z_16_clustering = z_16_clustering[z_16_clustering[num_col[idx]] <  upper_bnd]
# z_16_clustering[num_col[idx]].hist(bins=30)



# logerror - removed outliers > 0.9685 and outliers < -0.7319
# count    87626.000000
# mean         0.011158
# std          0.157813
# min         -4.605000
# 0.2%        -0.731900
# 50%          0.005000
# 99.8%        0.968406
# max          4.737000
# Name: logerror, dtype: float64


idx = 13
# print(num_col[idx])
# print(z_16_clustering.shape)
# z_16_clustering[num_col[idx]].hist(bins=30)
# z_16_clustering[num_col[idx]].describe(percentiles=[.9975, .0025])
upper_bnd = 0.9685
lower_bnd = -0.7319
z_16_clustering = z_16_clustering[z_16_clustering[num_col[idx]] <  upper_bnd]
z_16_clustering = z_16_clustering[z_16_clustering[num_col[idx]] >  lower_bnd]
# z_16_clustering[num_col[idx]].hist(bins=30)



# lotsizesquarefeet - removed outliers > 815788
# count    8.718500e+04
# mean     2.560400e+04
# std      1.150715e+05
# min     -1.000000e+00
# 0.2%    -1.000000e+00
# 50%      6.742000e+03
# 99.8%    8.157880e+05
# max      6.971010e+06
# Name: lotsizesquarefeet, dtype: float64


idx = 15
# print(num_col[idx])
# print(z_16_clustering.shape)
# z_16_clustering[num_col[idx]].hist(bins=30)
# z_16_clustering[num_col[idx]].describe(percentiles=[.9975, .0025])
upper_bnd = 815789
z_16_clustering = z_16_clustering[z_16_clustering[num_col[idx]] <  upper_bnd]

# z_16_clustering[num_col[idx]].hist(bins=30)



# poolsizesum - removed outliers > 545
# count    86976.000000
# mean        95.570065
# std        195.773394
# min          0.000000
# 0.2%         0.000000
# 50%          0.000000
# 99.8%      545.125000
# max        960.000000
# Name: poolsizesum, dtype: float64

idx = 20
# print(num_col[idx])
# print(z_16_clustering.shape)
# z_16_clustering[num_col[idx]].hist(bins=30)
# z_16_clustering[num_col[idx]].describe(percentiles=[.9975, .0025])
upper_bnd = 546
z_16_clustering = z_16_clustering[z_16_clustering[num_col[idx]] <  upper_bnd]


# roomcnt - removed outliers > 10
# count    86758.000000
# mean         1.451739
# std          2.775266
# min          0.000000
# 0.2%         0.000000
# 50%          0.000000
# 99.8%       10.000000
# max         13.000000
# Name: roomcnt, dtype: float64


idx = 22
# print(num_col[idx])
# print(z_16_clustering.shape)
# z_16_clustering[num_col[idx]].hist(bins=30)
# z_16_clustering[num_col[idx]].describe(percentiles=[.9975, .0025])
upper_bnd = 10.1
z_16_clustering = z_16_clustering[z_16_clustering[num_col[idx]] <  upper_bnd]




# structuretaxvaluedollarcnt - removed outliers > 1053985
# count    8.641400e+04
# mean     1.645139e+05
# std      1.424486e+05
# min      0.000000e+00
# 0.2%     7.115000e+03
# 50%      1.291290e+05
# 99.8%    1.053985e+06
# max      3.500000e+06
# Name: structuretaxvaluedollarcnt, dtype: float64


idx = 23
# print(num_col[idx])
# print(z_16_clustering.shape)
# z_16_clustering[num_col[idx]].hist(bins=30)
# z_16_clustering[num_col[idx]].describe(percentiles=[.9975, .0025])
upper_bnd = 1053985
z_16_clustering = z_16_clustering[z_16_clustering[num_col[idx]] <  upper_bnd]





# taxamount - removed outliers > 30864
# count     86197.000000
# mean       5450.939176
# std        4322.904711
# min          49.080000
# 0.2%        504.623000
# 50%        4467.740000
# 99.8%     30863.119500
# max      173218.070000
# Name: taxamount, dtype: float64

idx = 24
# print(num_col[idx])
# print(z_16_clustering.shape)
# z_16_clustering[num_col[idx]].hist(bins=30)
# z_16_clustering[num_col[idx]].describe(percentiles=[.9975, .0025])
upper_bnd = 30864
z_16_clustering = z_16_clustering[z_16_clustering[num_col[idx]] <  upper_bnd]



# count    85982.000000
# mean         1.000419
# std          0.021019
# min          1.000000
# 0.2%         1.000000
# 50%          1.000000
# 99.8%        1.000000
# max          3.000000
# Name: threequarterbathnbr, dtype: float64


idx = 25
# print(num_col[idx])
# print(z_16_clustering.shape)
# z_16_clustering[num_col[idx]].hist(bins=30)
# z_16_clustering[num_col[idx]].describe(percentiles=[.9975, .0025])
upper_bnd = 1.1
z_16_clustering = z_16_clustering[z_16_clustering[num_col[idx]] <  upper_bnd]
z_16_clustering[num_col[idx]].hist(bins=30)


# unitcnt - removed outliers > 4.1
# count    86475.000000
# mean         0.377785
# std          1.186620
# min         -1.000000
# 0.2%        -1.000000
# 50%          1.000000
# 99.8%        4.000000
# max        143.000000
# Name: unitcnt, dtype: float64

idx = 26
# print(num_col[idx])
# print(z_16_clustering.shape)
# z_16_clustering[num_col[idx]].hist(bins=30)
# z_16_clustering[num_col[idx]].describe(percentiles=[.9975, .0025])
upper_bnd = 4.1
z_16_clustering = z_16_clustering[z_16_clustering[num_col[idx]] <  upper_bnd]






# #List of features without intervention
# #finishedsquarefeet12
# # count    2.700485e+06
# # mean     1.737672e+03
# # std      8.189221e+02
# # min      1.000000e+00
# # 50%      1.537000e+03
# # 99.8%    5.929000e+03
# # max      7.147000e+03
# # Name: finishedsquarefeet12, dtype: float64



# #List of features with intervention
# # bathroomcnt 0 - 20 99.75% - 7.5 remove entries beyond 99.75%
# # bedroomcnt 0 - 20 99.75% - 7.5 remove entries beyond 99.75%

# # finishedfloor1squarefeet
# # count    202473.000000
# # mean       1377.550780
# # std         619.081217
# # min           3.000000
# # 50%        1282.000000
# # 99.8%      4776.820000
# # max       31303.000000
# # Name: finishedfloor1squarefeet, dtype: float64


# # calculatedfinishedsquarefeet - remove beyond 99.75%

# # fireplacecnt - remove > 4 from histogram
# # count    311847.000000
# # mean          1.164953
# # std           0.448701
# # min           1.000000
# # 50%           1.000000
# # 99.8%         3.000000
# # max           9.000000
# # Name: fireplacecnt, dtype: float64

# # garagecarcnt - remove beyond 99.75%
# # count    881017.000000
# # mean          1.816656
# # std           0.576970
# # min           0.000000
# # 50%           2.000000
# # 99.8%         5.000000
# # max          19.000000
# # Name: garagecarcnt, dtype: float64








# idx = 5
# print(col_num[idx])
# print(properties_16.shape)
# properties_16[col_num[idx]].hist(bins=5)
# properties_16[col_num[idx]].describe(percentiles=[.9975])
# properties_16 = properties_16[properties_16[col_num[idx]] <  properties_16[col_num[idx]].quantile(0.9975)]




# idx = 13
# print(col_num[idx])
# print(properties_16.shape)
# properties_16[col_num[idx]].hist(bins=10)
# properties_16[col_num[idx]].describe(percentiles=[.9975])
# properties_16 = properties_16[properties_16[col_num[idx]] <  properties_16[col_num[idx]].quantile(0.9975)]


# idx = 14
# print(col_num[idx])
# print(properties_16.shape)
# properties_16[col_num[idx]].hist(bins=10)
# properties_16[col_num[idx]].describe(percentiles=[.9975])
# properties_16_3 = properties_16[properties_16[col_num[idx]] < properties_16[col_num[idx]].quantile(0.9975)]
# properties_16_4 = properties_16[properties_16[col_num[idx]].isna()]

# frames = [properties_16_3, properties_16_4]
# res = pd.concat(frames)


# idx = 11
# print(col_num[idx])
# print(properties_16.shape)
# res[col_num[idx]].hist(bins=5)
# res[col_num[idx]].describe(percentiles=[.9975])
# properties_16_1 = res[res[col_num[idx]] <  4]
# properties_16_2 = res[res[col_num[idx]].isna()]

# frames = [properties_16_1, properties_16_2]
# res = pd.concat(frames)
# res.shape





# # #If 'garagecarcnt' is null then garbage doesnt exist
# # index = z_16.garagecarcnt.isnull()
# # z_16.loc[index,'garagecarcnt'] = 0



# In[53]:


def detect_outlier(data):
    
    """
    input: numerical data, Array
    return: new data without outlier
    """
    tmp = data.copy()
    
    q1, q3 = np.percentile(tmp, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (iqr * 1.5)
    upper_bound = q3 + (iqr * 1.5)
    
    has_outlier = np.max(tmp) > upper_bound or np.min(tmp) < lower_bound
    if not has_outlier:
        return tmp
    upper_idx = np.where(tmp > upper_bound)
    lower_idx = np.where(tmp < lower_bound)
    tmp[upper_idx] = upper_bound
    tmp[lower_idx] = lower_bound
    
    if np.max(tmp) <= upper_bound and np.min(tmp) >= lower_bound:
        print ("Done with handling outliers")
    else:
        print ("Warning!! Fail handling outliers")
    
    return tmp


# In[54]:


for f in num_col:
    z_16_clustering[f] = detect_outlier(z_16_clustering[f].values)


# In[55]:


for f in num_col:
    z_17_clustering[f] = detect_outlier(z_17_clustering[f].values)


# # Model Training

# ## Get Dummy

# In[56]:


cat_col = ['propertycountylandusecode', 'airconditioningtypeid', 'buildingqualitytypeid',
          'heatingorsystemtypeid', 'pooltypeid2', 'pooltypeid7', 'propertylandusetypeid',
          'regionidcounty', 'regionidzip', 'hashottuborspa', 'fireplaceflag', 'taxdelinquencyflag']


# In[57]:


z_16_clustering = pd.get_dummies(z_16_clustering, columns = cat_col)
z_17_clustering = pd.get_dummies(z_17_clustering, columns = cat_col)


# In[58]:


z_16_clustering.shape


# In[59]:


z_17_clustering.shape


# In[60]:


z_16_clustering.to_csv('z_16_clustering.csv', index=True)
z_17_clustering.to_csv('z_16_clustering.csv', index=True)


# In[61]:


# z_16_clustering = pd.read_csv('C:/Users/70785/z_16_clustering.csv', header = 0)
# z_17_clustering = pd.read_csv('C:/Users/70785/z_17_clustering.csv', header = 0)


# ## Split training and test

# In[194]:


x = z_16_clustering[z_16_clustering.columns.difference(['logerror'])].values
y = z_16_clustering['logerror'].values
x_17 = z_17_clustering[z_17_clustering.columns.difference(['logerror'])].values
y_17 = z_17_clustering['logerror'].values

# Split data into train and test (80% & 20%)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, random_state = 0)
x_17_train, x_17_test, y_17_train, y_17_test = train_test_split(
    x_17, y_17, test_size = 0.2, random_state = 0)


# ## Standarization

# In[195]:


# initialize a scaler object
scaler = StandardScaler()

# transform training set
x_train_std = scaler.fit_transform(x_train)

# the same transform for test set
x_test_std = scaler.transform(x_test)

# transform whole dataset 
X = scaler.fit_transform(x)

# initialize a scaler object
scaler = StandardScaler()

# transform training set
x_17_train_std = scaler.fit_transform(x_17_train)

# the same transform for test set
x_17_test_std = scaler.transform(x_17_test)

# transform whole dataset 
X_17 = scaler.fit_transform(x_17)


# ## Model selection

# In[196]:


# Function to evaluate the model using AIC with sum of squared error
def model_evaluation(X, y, cv, regressor, **kwargs):
    
    start = time.time()
    kf = KFold(n_splits=cv,shuffle=True)
    y_predict = y.copy()
    clf = regressor(**kwargs)
    
    rmse = 0
    mae = 0
    for train_index, validate_index in kf.split(X):
        
        # split to training and validation set
        X_train, X_validate = X[train_index], X[validate_index]
        y_train, y_validate = y[train_index], y[validate_index]
        
        # fit current model
        clf.fit(X_train, y_train)
        
        # predict on validation set
        y_predict = clf.predict(X_validate)
        
        # rmse
        rmse += np.sqrt(np.mean((y_predict - y_validate)**2))
        
        # mae 
        mae += np.mean(abs(y_predict - y_validate))
    end = time.time()    
    # return average score among all validation sets
    return rmse / cv, mae / cv, end-start


# In[65]:


models = [LinearRegression, Ridge, Lasso, ElasticNet, DecisionTreeRegressor, BaggingRegressor, RandomForestRegressor, ExtraTreesRegressor, 
          AdaBoostRegressor, GradientBoostingRegressor, SVR]
# results
results_dict = {}

# train models
for model in models:
    print ("Training model: ", model)
    if(model == Ridge):
        rmse, mae, runtime = model_evaluation(x_train, y_train, 5, model, alpha = 0.1)
    elif(model == Lasso):
        rmse, mae, runtime = model_evaluation(x_train, y_train, 5, model, alpha = 0.001)
    elif(model == ElasticNet):
        rmse, mae, runtime = model_evaluation(x_train, y_train, 5, model, alpha = 0.001, l1_ratio = 0.5)
    elif(model == SVR):
        rmse, mae, runtime = model_evaluation(x_train, y_train, 5, model, gamma='scale', C=1.0, epsilon=0.2)
    else:
        rmse, mae, runtime = model_evaluation(x_train, y_train, 5, model)
    results_dict[model] = [rmse, mae, runtime]

print ('Done with model selection!')


# In[66]:


df = pd.DataFrame(results_dict).T
df.columns = [ 'rmse', 'mae', 'time']
df


# Tune Lasso

# In[197]:


from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error

candidate_alpha = []
alpha = 0.000001
for i in range(200):
    candidate_alpha.append(alpha)
    alpha += 0.000005
mae_list = []

for alpha in candidate_alpha:
    
    # initialize a Lasso model
    lasso = Lasso(alpha)

    # fit training data
    model_lasso = lasso.fit(x_train_std,y_train)

    # predict on test data
    y_pred_lasso = model_lasso.predict(x_test_std)
    
    # calculate mae
    mae = mean_absolute_error(y_test, y_pred_lasso)
    
    # append to list
    mae_list.append(mae)


# In[198]:


plt.plot(candidate_alpha, mae_list)
plt.xlabel("alphas")
plt.ylabel("MAE")
plt.title("Hyperparameter tuning in Lasso Regression")
plt.show()


# In[199]:


best_alpha = candidate_alpha[mae_list.index(min(mae_list))]
best_alpha


# In[200]:


lasso = Lasso(best_alpha)
model_lasso = lasso.fit(x_train_std,y_train)
lasso_17 = Lasso(best_alpha)
model_lasso_17 = lasso_17.fit(x_17_train_std,y_17_train)


# Feature selection

# In[201]:


parameter_16 = model_lasso.coef_
parameter_17 = model_lasso_17.coef_
print(parameter_16)


# In[203]:


negative_index =z_16_clustering[z_16_clustering.columns.difference(['logerror'])].columns[np.argwhere(parameter_16<-1e-6).reshape(1,-1)][0,]
negative_index_17 = z_17_clustering[z_17_clustering.columns.difference(['logerror'])].columns[np.argwhere(parameter_17<-1e-6).reshape(1,-1)][0,]
print(negative_index)


# In[204]:


positive_index =z_16_clustering[z_16_clustering.columns.difference(['logerror'])].columns[np.argwhere(parameter_16>1e-6).reshape(1,-1)][0,]
positive_index_17 = z_17_clustering[z_17_clustering.columns.difference(['logerror'])].columns[np.argwhere(parameter_17>1e-6).reshape(1,-1)][0,]
print(positive_index)


# In[205]:


z_16_lasso = z_16_clustering[np.hstack([positive_index ,negative_index])]
z_17_lasso = z_17_clustering[np.hstack([positive_index_17 ,negative_index_17])]


# In[207]:


x_lasso = z_16_lasso.values
y_lasso = z_16_clustering['logerror'].values
x_17_lasso = z_17_lasso.values
y_17_lasso = z_17_clustering['logerror'].values

# Split data into train and test (80% & 20%)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, random_state = 0)
x_17_train, x_17_test, y_17_train, y_17_test = train_test_split(
    x_17, y_17, test_size = 0.2, random_state = 0)


# In[208]:


# initialize a scaler object
scaler = StandardScaler()

# transform training set
x_train_std = scaler.fit_transform(x_train)

# the same transform for test set
x_test_std = scaler.transform(x_test)

# transform whole dataset 
X = scaler.fit_transform(x)

# initialize a scaler object
scaler = StandardScaler()

# transform training set
x_17_train_std = scaler.fit_transform(x_17_train)

# the same transform for test set
x_17_test_std = scaler.transform(x_17_test)

# transform whole dataset 
X_17 = scaler.fit_transform(x_17)


# Catboost

# In[209]:


cb = CatBoostRegressor(learning_rate = 0.09, depth = 9)
cb.fit(x_train_std, y_train)
cb_17 = CatBoostRegressor()
cb_17.fit(x_17_train_std, y_17_train)


# Xgboost

# In[210]:


dtrain = xgb.DMatrix(x_train_std,label = y_train)
dtest = xgb.DMatrix(x_test_std,label = y_test)
dtrain_17 = xgb.DMatrix(x_17_train_std,label = y_17_train)
dtest_17 = xgb.DMatrix(x_17_test_std,label = y_17_test)


# In[211]:


param1 = {'max_depth':4, 'eta':0.1, 'subsample':0.8, 'colsample_bytree':0.8, 'alpha':0.01, 'objective':'reg:linear', 'early_stopping_rounds':30}

# specify validations set to watch performance
watchlist = [(dtest, 'eval'), (dtrain, 'train')]
watchlist_17 = [(dtest_17, 'eval'), (dtrain_17, 'train')]
num_round = 300
bst1 = xgb.train(param1, dtrain, num_round, watchlist)
bst1_17 = xgb.train(param1, dtrain_17, num_round, watchlist_17)


# In[212]:


param2 = {'max_depth':5, 'eta':0.2, 'subsample':0.8, 'colsample_bytree':0.8, 'alpha':0.01, 'objective':'reg:linear', 'early_stopping_rounds':30}

# specify validations set to watch performance
watchlist = [(dtest, 'eval'), (dtrain, 'train')]
watchlist_17 = [(dtest_17, 'eval'), (dtrain_17, 'train')]
num_round = 300
bst2 = xgb.train(param2, dtrain, num_round, watchlist)
bst2_17 = xgb.train(param2, dtrain_17, num_round, watchlist_17)


# Random Forest

# In[213]:


rf1 = RandomForestRegressor(bootstrap=True,criterion='mse', max_depth=None,max_features='auto',max_leaf_nodes=None,min_samples_leaf=35,min_samples_split=50,n_estimators=50,min_weight_fraction_leaf=0.0)
rf1.fit(x_train_std,y_train)
rf2 = RandomForestRegressor(bootstrap=True,criterion='mse', max_depth=None,max_features='auto',max_leaf_nodes=None,min_samples_leaf=35,min_samples_split=50,n_estimators=50,min_weight_fraction_leaf=0.0)
rf2.fit(x_train_std,y_train)
rf3 = RandomForestRegressor(bootstrap=True,criterion='mse', max_depth=None,max_features='auto',max_leaf_nodes=None,min_samples_leaf=35,min_samples_split=50,n_estimators=50,min_weight_fraction_leaf=0.0)
rf3.fit(x_train_std,y_train)

rf1_17 = RandomForestRegressor(bootstrap=True,criterion='mse', max_depth=None,max_features='auto',max_leaf_nodes=None,min_samples_leaf=35,min_samples_split=50,n_estimators=50,min_weight_fraction_leaf=0.0)
rf1_17.fit(x_17_train_std,y_17_train)
rf2_17 = RandomForestRegressor(bootstrap=True,criterion='mse', max_depth=None,max_features='auto',max_leaf_nodes=None,min_samples_leaf=35,min_samples_split=50,n_estimators=50,min_weight_fraction_leaf=0.0)
rf2_17.fit(x_17_train_std,y_17_train)
rf3_17 = RandomForestRegressor(bootstrap=True,criterion='mse', max_depth=None,max_features='auto',max_leaf_nodes=None,min_samples_leaf=35,min_samples_split=50,n_estimators=50,min_weight_fraction_leaf=0.0)
rf3_17.fit(x_17_train_std,y_17_train)


# In[226]:


y_pred_1 = cb.predict(x_test)
y_pred_2 = bst1.predict(xgb.DMatrix(x_test_std))
y_pred_3 = bst2.predict(xgb.DMatrix(x_test_std))
y_pred_4 = rf1.predict(x_test)
y_pred_5 = rf2.predict(x_test)
y_pred_6 = rf3.predict(x_test)

y_17_pred_1 = cb_17.predict(x_17_test)
y_17_pred_2 = bst1_17.predict(xgb.DMatrix(x_17_test_std))
y_17_pred_3 = bst2_17.predict(xgb.DMatrix(x_17_test_std))
y_17_pred_4 = rf1_17.predict(x_17_test)
y_17_pred_5 = rf2_17.predict(x_17_test)
y_17_pred_6 = rf3_17.predict(x_17_test)


# Stacked Model

# In[227]:


y_pred = np.vstack((y_pred_1,y_pred_2,y_pred_3,y_pred_4,y_pred_6))
y_pred = pd.DataFrame(y_pred.transpose(),columns=list('ABCDE'))

y_pred_17 = np.vstack((y_17_pred_1,y_17_pred_2,y_17_pred_3,y_17_pred_4,y_17_pred_6))
y_pred_17 = pd.DataFrame(y_pred_17.transpose(),columns=list('ABCDE'))


# In[228]:


stacked_model = LinearRegression()
stacked_model.fit(y_pred, y_test)
stacked_model.score(y_pred, y_test)


# In[229]:


stacked_model_17 = LinearRegression()
stacked_model_17.fit(y_pred_17, y_17_test)
stacked_model_17.score(y_pred_17, y_17_test)


# # Make Prediction

# In[169]:


properties_16_clustering['month'] = 10
properties_16_clustering['year'] = 2016
properties_17_clustering['month'] = 10
properties_17_clustering['year'] = 2017


# In[170]:


properties_16_clustering = pd.get_dummies(properties_16_clustering, columns = cat_col)
properties_17_clustering = pd.get_dummies(properties_17_clustering, columns = cat_col)


# In[171]:


properties_16_clustering.drop('airconditioningtypeid_12.0', axis=1, inplace=True)
properties_16_clustering['buildingqualitytypeid_1.0'] += properties_16_clustering['buildingqualitytypeid_2.0'] +properties_16_clustering['buildingqualitytypeid_3.0']
properties_16_clustering.drop('buildingqualitytypeid_2.0', axis=1, inplace=True)
properties_16_clustering.drop('buildingqualitytypeid_3.0', axis=1, inplace=True)


# In[172]:


properties_16_clustering['buildingqualitytypeid_4.0'] += properties_16_clustering['buildingqualitytypeid_5.0'] +properties_16_clustering['buildingqualitytypeid_6.0']
properties_16_clustering.drop('buildingqualitytypeid_5.0', axis=1, inplace=True)
properties_16_clustering.drop('buildingqualitytypeid_6.0', axis=1, inplace=True)
properties_16_clustering['buildingqualitytypeid_8.0'] += properties_16_clustering['buildingqualitytypeid_9.0'] 
properties_16_clustering.drop('buildingqualitytypeid_9.0', axis=1, inplace=True)
properties_16_clustering['buildingqualitytypeid_10.0'] += properties_16_clustering['buildingqualitytypeid_11.0'] 
properties_16_clustering.drop('buildingqualitytypeid_11.0', axis=1, inplace=True)
properties_16_clustering['heatingorsystemtypeid_24.0'] += properties_16_clustering['heatingorsystemtypeid_19.0'] +properties_16_clustering['heatingorsystemtypeid_21.0']
properties_16_clustering.drop('heatingorsystemtypeid_19.0', axis=1, inplace=True)
properties_16_clustering.drop('heatingorsystemtypeid_21.0', axis=1, inplace=True)
properties_16_clustering.drop('propertylandusetypeid_-1.0', axis=1, inplace=True)
properties_16_clustering.drop('propertylandusetypeid_47.0', axis=1, inplace=True)
properties_16_clustering.drop('propertylandusetypeid_270.0', axis=1, inplace=True)
properties_16_clustering.drop('regionidcounty_-1.0', axis=1, inplace=True)
properties_16_clustering.drop('regionidzip_an', axis=1, inplace=True)


# In[173]:


properties_17_clustering.drop('airconditioningtypeid_3.0', axis=1, inplace=True)
properties_17_clustering.drop('airconditioningtypeid_12.0', axis=1, inplace=True)
properties_17_clustering['heatingorsystemtypeid_24.0'] += properties_17_clustering['heatingorsystemtypeid_12.0'] +properties_17_clustering['heatingorsystemtypeid_14.0']++properties_17_clustering['heatingorsystemtypeid_19.0']++properties_17_clustering['heatingorsystemtypeid_21.0']
properties_17_clustering.drop('heatingorsystemtypeid_12.0', axis=1, inplace=True)
properties_17_clustering.drop('heatingorsystemtypeid_14.0', axis=1, inplace=True)
properties_17_clustering.drop('heatingorsystemtypeid_19.0', axis=1, inplace=True)
properties_17_clustering.drop('heatingorsystemtypeid_21.0', axis=1, inplace=True)
properties_17_clustering.drop('propertylandusetypeid_47.0', axis=1, inplace=True)
properties_17_clustering.drop('propertylandusetypeid_270.0', axis=1, inplace=True)
properties_17_clustering.drop('propertylandusetypeid_279.0', axis=1, inplace=True)


# In[52]:


# properties_16_clustering.to_csv('properties_16_clustering.csv', index=True)
# properties_17_clustering.to_csv('properties_17_clustering.csv', index=True)


# In[ ]:


# properties_16_clustering = pd.read_csv('C:/Users/70785/properties_16_clustering.csv', header = 0)
# properties_17_clustering = pd.read_csv('C:/Users/70785/properties_17_clustering.csv', header = 0)


# In[220]:


properties_16_lasso = properties_16_clustering[np.hstack([positive_index ,negative_index])]
properties_17_lasso = properties_17_clustering[np.hstack([positive_index_17 ,negative_index_17])]


# In[221]:


# initialize a scaler object
scaler = StandardScaler()
# transform set
properties_16_clustering_std = scaler.fit_transform(properties_16_clustering)

# initialize a scaler object
scaler = StandardScaler()
# transform set
properties_17_clustering_std = scaler.fit_transform(properties_17_clustering)


# In[230]:


y_pred_1 = cb.predict(properties_16_clustering)
y_pred_2 = bst1.predict(xgb.DMatrix(properties_16_clustering_std))
y_pred_3 = bst2.predict(xgb.DMatrix(properties_16_clustering_std))
y_pred_4 = rf1.predict(properties_16_clustering)
y_pred_5 = rf2.predict(properties_16_clustering)
y_pred_6 = rf3.predict(properties_16_clustering)

y_17_pred_1 = cb_17.predict(properties_17_clustering)
y_17_pred_2 = bst1_17.predict(xgb.DMatrix(properties_17_clustering_std))
y_17_pred_3 = bst2_17.predict(xgb.DMatrix(properties_17_clustering_std))
y_17_pred_4 = rf1_17.predict(properties_17_clustering)
y_17_pred_5 = rf2_17.predict(properties_17_clustering)
y_17_pred_6 = rf3_17.predict(properties_17_clustering)


# In[231]:


y_pred = np.vstack((y_pred_1,y_pred_2,y_pred_3,y_pred_4,y_pred_6))
y_pred = pd.DataFrame(y_pred.transpose(),columns=list('ABCDE'))

y_pred_17 = np.vstack((y_17_pred_1,y_17_pred_2,y_17_pred_3,y_17_pred_4,y_17_pred_6))
y_pred_17 = pd.DataFrame(y_pred_17.transpose(),columns=list('ABCDE'))

# y_pred = np.vstack((y_pred_1,y_pred_2,y_pred_3,y_pred_4,y_pred_5,y_pred_6,y_pred_10,y_pred_11,y_pred_12,y_pred_13,y_pred_14))
# y_pred = pd.DataFrame(y_pred.transpose(),columns=list('ABCDEFGHIJK'))

# y_pred_17 = np.vstack((y_17_pred_1,y_17_pred_2,y_17_pred_3,y_17_pred_4,y_17_pred_5,y_17_pred_6,y_17_pred_10,y_17_pred_11,y_17_pred_12,y_17_pred_13,y_17_pred_14))
# y_pred_17 = pd.DataFrame(y_pred_17.transpose(),columns=list('ABCDEFGHIJK'))


# In[232]:


y_new = stacked_model.predict(y_pred)
y_17_new = stacked_model_17.predict(y_pred_17)


# In[233]:


dataframe = pd.DataFrame({'ParcelId':properties_16_clustering['parcelid'],'201610':y_new,'201611':y_new,'201612':y_new,'201710':y_17_new,'201711':y_17_new,'201712':y_17_new})
dataframe.to_csv("test.csv",index=False)


# In[ ]:




