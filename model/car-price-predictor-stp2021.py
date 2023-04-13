#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[2]:


df = pd.read_csv('Car details.csv')


# In[3]:


df.head()


# In[4]:


# shape of data
df.shape


# In[5]:


# missing values
df.isnull().sum()


# In[6]:


# info
df.info()


# In[7]:


# duplicate data
df.duplicated().sum()


# In[8]:


# describe function
df.describe()


# In[9]:


1.000000e+07


# In[10]:


# Observation
# 1. Missing values in some cols
# 2. Seats is float should be int
# 3. more than 1000 rows are duplicates
# 4. Outliers in year,selling price,seats
# 5. Torque,engine,mileage and max_power have unnecceary units


# In[11]:


# drop any row with missing values
df.dropna(inplace=True)


# In[12]:


df.shape


# In[13]:


# no of dropped rows
8128 - 7906
df.isnull().sum()


# In[14]:


# remove duplicate rows
df = df.drop_duplicates(keep='first')


# In[15]:


df.shape


# In[16]:


df.duplicated().sum()


# In[17]:


# change data type of seats col
df['seats'] = df['seats'].astype('int32')


# In[18]:


df.info()


# In[19]:


# Handling mileage col
df['mileage'] = df['mileage'].str.split(expand=True)[0]
df['mileage'] = df['mileage'].astype('float64')


# In[20]:


# Handling engine col
df['engine'] = df['engine'].str.split(expand=True)[0]
df['engine'] = df['engine'].astype('int32')


# In[21]:


# Handling max_power col
df['max_power'] = df['max_power'].str.split(expand=True)[0]
df['max_power'] = df['max_power'].astype('float64')


# In[22]:


df.head()


# In[23]:


# dropping the torque col
df.drop(columns=['torque'],inplace=True)


# In[24]:


df.head()


# In[25]:


df['name'].unique().shape


# In[26]:


# extracting brand from name
df['brand'] = df['name'].str.split(expand=True)[0]


# In[27]:


df.drop(columns=['name'],inplace=True)
df.head()


# In[28]:


freq_brands = df['brand'].value_counts()[df['brand'].value_counts()>100].index.tolist()


# In[29]:


df = df[df['brand'].isin(freq_brands)]


# In[30]:


df.head()


# In[31]:


freq_fuel = ['Diesel','Petrol']
df = df[df['fuel'].isin(freq_fuel)]


# In[32]:


df['seller_type'].value_counts()


# In[33]:


df = df[df['seller_type'].isin(['Individual','Dealer'])]


# In[34]:


df = df[df['owner'].isin(['First Owner','Second Owner','Third Owner'])]


# In[35]:


df.shape


# In[36]:


import seaborn as sns
sns.boxplot(df['km_driven'])


# In[37]:


q1 = df['km_driven'].quantile(0.25)
q3 = df['km_driven'].quantile(0.75)
iqr = q3 - q1


# In[38]:


max_val_km_driven = q3 + 1.5*iqr


# In[39]:


max_val_km_driven


# In[40]:


df.head()


# In[41]:


df['km_driven'] = np.where(df['km_driven']>max_val_km_driven,max_val_km_driven,df['km_driven'])


# In[42]:


sns.boxplot(df['year'])


# In[43]:


df = df[df['year'] >= 2000]


# In[44]:


sns.boxplot(df['mileage'])


# In[45]:


mean_mileage = df[df['mileage'] !=0]['mileage'].median()
mean_mileage


# In[46]:


df['mileage'] = np.where(df['mileage'] == 0,mean_mileage,df['mileage'])


# In[47]:


df.shape


# In[48]:


df.head()


# In[49]:


X = df.drop(columns=['selling_price'])
y = df['selling_price']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)


# In[50]:


# Ordinal encoding on Owner col
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer


# In[51]:


tnf = ColumnTransformer(
    [
        ('ordinal',OrdinalEncoder(),['owner']),
        ('nominal',OneHotEncoder(drop='first',sparse=False),['fuel','seller_type','transmission','brand'])
    ], remainder='passthrough'
)


# In[52]:


X_train_tnf = tnf.fit_transform(X_train)
X_test_tnf = tnf.transform(X_test)


# In[53]:


from sklearn.linear_model import LinearRegression


# In[54]:


lr = LinearRegression()


# In[55]:


lr.fit(X_train_tnf,y_train)


# In[56]:


y_pred = lr.predict(X_test_tnf)


# In[57]:


from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


# In[58]:


from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)


# In[59]:


X_train_poly = poly.fit_transform(X_train_tnf)
X_test_poly = poly.transform(X_test_tnf)


# In[60]:


lr = LinearRegression()
lr.fit(X_train_poly,y_train)
y_pred = lr.predict(X_test_poly)
r2_score(y_test,y_pred)


# In[61]:


# pipeline


# In[62]:


from sklearn.pipeline import Pipeline
pipe = Pipeline(
    [
        ('col-transformer',tnf),
        ('poly',poly),
        ('lr',lr)
    ]
)

pipe.fit(X_train,y_train)
y_pred = pipe.predict(X_test)
r2_score(y_test,y_pred)


# In[63]:


import pickle
pickle.dump(pipe,open('pipe.pkl','wb'))


# In[64]:


X_train['brand'].value_counts()


# In[65]:


y_train


# In[66]:


X_train


# In[ ]:




