#!/usr/bin/env python
# coding: utf-8

# # Sales Prediction for Big Mart Outlets
# The data scientists at BigMart have collected 2013 sales data for 1559 products across 10 stores in different cities. Also, certain attributes of each product and store have been defined. The aim is to build a predictive model and predict the sales of each product at a particular outlet.
# 
# Using this model, BigMart will try to understand the properties of products and outlets which play a key role in increasing sales.
# 
# Please note that the data may have missing values as some stores might not report all the data due to technical glitches. Hence, it will be required to treat them accordingly. 

# ### Importing Libraries

# In[1]:


import pandas as pd
import numpy  as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy import stats
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')
import pandas_profiling


# ### Loading Train and Test Data

# In[2]:


#LOAD THE TRAIN DATA SHEET TO PYTHON ENVIRONMENT
train = pd.read_csv('TRAIN.csv')


# In[3]:


#PRINT THE FIRST 5 LINES OF THE TRAIN DATASET
train.head()


# In[4]:


#SHAPE OF THE TRAIN DATASET
train.shape


# In[5]:


#LOAD THE TEST DATA SHEET TO PYTHON ENVIRONMENT
test = pd.read_csv('TEST_BM.csv')


# In[6]:


#PRINT THE FIRST 5 LINES OF THE TEST DATASET
test.head()


# In[7]:


#SHAPE OF THE TRAIN DATASET
test.shape


# ### Data Pre-Processing

# In[8]:


#DISPLAY THE FULL SUMMARY OF THE TRAIN DATAFRAME.
train.info()


# In[9]:


#STATISTICS SUMMARY OF THE TRAIN  DATAFRAME.
train.describe()


# In[10]:


# Check for duplicates of train data
idsTotal = train.shape[0]
idsDupli = train[train['Item_Identifier'].duplicated()]
print(f'There are {len(idsDupli)} duplicate IDs for {idsTotal} total entries')


# In[11]:


# Check datatypes
train.dtypes


# In[12]:


freqgraph = train.select_dtypes(include = ['float'])
freqgraph.hist(figsize =(20,15))
plt.show()


# In[13]:


sns.countplot(train['Outlet_Establishment_Year'])
plt.show()


# In[14]:


#DISPLAY THE FULL SUMMARY OF THE Test DATAFRAME.
test.info()


# In[15]:


#STATISTICS SUMMARY OF THE TEST  DATAFRAME.
test.describe()


# ### Missing Values 

# In[16]:


# Finding the null values in train dataset.
train.isnull().sum()


# In[17]:


# Check datatypes
train.dtypes


# In[18]:


train['Item_Identifier'].nunique()


# In[19]:


train['Outlet_Identifier'].unique()


# In[20]:


train['Outlet_Size'].unique()


# In[21]:


train['Outlet_Size'].mode()


# In[22]:


train['Item_Weight'].nunique()


# In[23]:


#Item_Weight is numerical column so we fill it with Median Imputation


# In[24]:


train['Item_Weight']= train['Item_Weight'].fillna(train['Item_Weight'].median())


# In[25]:


#Outlet_Size is catagorical column so we fill it with Mode Imputation


# In[26]:


train['Outlet_Size']=train['Outlet_Size'].fillna(train['Outlet_Size'].mode()[0])


# In[27]:


# Check datatypes
train.isnull().sum()


# In[28]:


# Finding the null values in test dataset.
test.isnull().sum()


# In[29]:


#Item_Weight is numerical column so we fill it with Median Imputation


# In[30]:


test['Item_Weight']= test['Item_Weight'].fillna(test['Item_Weight'].median())


# In[31]:


#Outlet_Size is catagorical column so we fill it with Mode Imputation


# In[32]:


test['Outlet_Size']=test['Outlet_Size'].fillna(test['Outlet_Size'].mode()[0])


# In[33]:


test['Item_Identifier'].nunique()


# In[34]:


test['Outlet_Identifier'].unique()


# In[35]:


test.isnull().sum()


# In[36]:


test.dtypes


# #### We have successfully removed null values from train and test dataset.

# In[37]:


train['Item_Fat_Content'].value_counts()


# In[38]:


train['Item_Fat_Content'].replace(['low fat','LF','reg'],['Low Fat','Low Fat','Regular'],inplace = True)
test['Item_Fat_Content'].replace(['low fat','LF','reg'],['Low Fat','Low Fat','Regular'],inplace = True)


# Some values in Item_Fat_Content were miscode as 'LF' and 'low fat' which meant same as 'Low Fat' , and 'reg' which meant 'Regular'.
# 
# So, we have replaced those miscoded values properly.

# ## EXPLORATORY DATA ANALYSIS (EDA) USING PANDAS PROFILING

# In[39]:


from pandas_profiling import ProfileReport


# In[40]:


profile = ProfileReport(train , title = " PANDAS PROFILING REPORT")


# In[41]:


profile.to_widgets()


# In[42]:


dtest = test.drop(['Item_Identifier','Outlet_Identifier'],axis=1)


# In[43]:


dtest


# In[44]:


dtrain = train.drop(['Item_Identifier','Outlet_Identifier'],axis=1)


# In[45]:


dtrain


# In[46]:


dtest.dtypes


# ## Preprocessing Task before Model Building
# 

# #### Label Encoding

# In[47]:


from sklearn.preprocessing import LabelEncoder
lbl=LabelEncoder()


# In[48]:


# Encoding train


for col in dtrain.columns:
    if dtrain[col].dtype == 'object':
        
        lbl.fit(list(dtrain[col].values))
        dtrain[col]=lbl.transform(dtrain[col].values)
        
# Encoding test

for coll in dtest.columns:
    if dtest[coll].dtype == 'object':
        
        lbl.fit(list(dtest[coll].values))
        dtest[coll]=lbl.transform(dtest[coll].values)   


# In[49]:


dtrain


# In[50]:


dtest


# ### Splitting our data into train and test

# In[51]:


X=dtrain.drop('Item_Outlet_Sales',axis=1)


# In[52]:


y=dtrain['Item_Outlet_Sales']


# In[53]:


Xtest = dtest


# In[54]:


X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42, test_size=0.2)


# #### LINEAR REGRESSION

# In[55]:


from sklearn.linear_model import LinearRegression


# In[56]:


# Instantiate learning mode
lr = LinearRegression()
# Fitting the model
lr.fit(X_train,y_train)
# Predicting the Test set results
y_pred_lr = lr.predict(X_test)


# In[57]:


y_pred_lr1 = lr.predict(Xtest)
y_pred_lr1


# In[58]:


from sklearn.metrics import mean_squared_error,r2_score


# In[59]:


print("R squared value :", r2_score(y_test,y_pred_lr))
print("Mean squared value :", mean_squared_error(y_test,y_pred_lr))
r1 = np.sqrt(mean_squared_error(y_test,y_pred_lr))
print("Root Mean Square Error :", np.sqrt(mean_squared_error(y_test,y_pred_lr)))


# #### kNN NEIGHBOR

# In[60]:


from sklearn.neighbors import KNeighborsRegressor


# In[61]:


knn = KNeighborsRegressor(n_neighbors=10)
knn.fit(X_train,y_train)
y_pred_knn = knn.predict(X_test)


# In[62]:


y_pred_knn1 = knn.predict(Xtest)
y_pred_knn1


# In[63]:


print("R squared value :", r2_score(y_test,y_pred_knn))
print("Mean squared value :", mean_squared_error(y_test,y_pred_knn))
r2 = np.sqrt(mean_squared_error(y_test,y_pred_knn))
print("Root Mean Square Error :",np.sqrt(mean_squared_error(y_test,y_pred_knn)))


# #### DECISION TREE

# In[64]:


from sklearn.tree import DecisionTreeRegressor


# In[65]:


dt = DecisionTreeRegressor(max_depth=15, min_samples_leaf=100)
dt.fit(X_train,y_train)
y_pred_dt = dt.predict(X_test)


# In[66]:


y_pred_dt1 = dt.predict(Xtest)
y_pred_dt1


# In[67]:


print("R squared value :", r2_score(y_test,y_pred_dt))
print("Mean squared value :", mean_squared_error(y_test,y_pred_dt))
r3 = np.sqrt(mean_squared_error(y_test,y_pred_dt))
print("Root Mean Square Error :",np.sqrt(mean_squared_error(y_test,y_pred_dt)))


# #### RANDOM FOREST

# In[68]:


from sklearn.ensemble import RandomForestRegressor


# In[69]:


rf1=RandomForestRegressor(n_estimators=400,max_depth=6, min_samples_leaf=76,n_jobs=4)
rf1.fit(X_train,y_train)
y_pred_rf2 = rf1.predict(X_test)


# In[70]:


y_pred_rf3 = rf1.predict(Xtest)
y_pred_rf3


# In[71]:


print("R squared value :", r2_score(y_test,y_pred_rf2))
print("Mean squared value :", mean_squared_error(y_test,y_pred_rf2))
r4 = np.sqrt(mean_squared_error(y_test,y_pred_rf2))
print("Root Mean Square Error :",np.sqrt(mean_squared_error(y_test,y_pred_rf2)))


# #### Standarization

# In[72]:


X.describe()


# In[73]:


from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
sc= StandardScaler()


# In[74]:


X_train_std= sc.fit_transform(X_train)


# In[75]:


X_test_std= sc.transform(X_test)


# Model building

# In[76]:


X_test.head()


# In[77]:


# Instantiate learning mode
lr = LinearRegression()
# Fitting the model
lr.fit(X_train_std,y_train)
# Predicting the Test set results
y_pred_lr = lr.predict(X_test_std)


# In[78]:


# prediction on test data
test_data_prediction = lr.predict(X_test_std)


# In[79]:


print("R squared value :", r2_score(y_test,y_pred_lr))
print("Mean squared value :", mean_squared_error(y_test,y_pred_lr))
r1 = np.sqrt(mean_squared_error(y_test,y_pred_lr))
print("Root Mean Square Error :", np.sqrt(mean_squared_error(y_test,y_pred_lr)))


# ### Model Results

# In[80]:


result= {'Model': ['Linear Regression', 'KNN Neighbors', 'Decision Tree','Random Forest'], 
                 'Root Mean Square Error': [r1, r2, r3,r4]}
metrics= pd.DataFrame(result)
metrics


# In[81]:


# RANDOM FOREST GIVES THE BEST MODEL.


# In[ ]:





# In[82]:


#upload the sample submission file to python environment.
sample = pd.read_csv('sample_submission_8RXa3c6.csv')


# In[83]:


sample


# In[84]:


target = rf1.predict(Xtest)


# In[85]:


test.shape


# In[86]:


test.head()


# In[87]:


submitting_df = test[['Item_Identifier','Outlet_Identifier']]


# In[88]:


submitting_df


# In[89]:


submitting_df['Item_Outlet_Sales'] = target


# In[90]:


submitting_df


# #### SUBMISSION FILE

# In[91]:


pd.DataFrame(submitting_df).to_csv("mysubmissionbigmartfinal.csv",index=False)


# In[ ]:




