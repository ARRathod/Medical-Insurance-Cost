#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# In[2]:


data=pd.read_csv("insurance.csv")
data


# In[3]:


data.shape


# # Preprocessing Data

# In[4]:


def find_outlier(column):
    number_cols_ph = data[column].count()
    first_quartile = np.quantile(data[column], 0.25)
    third_quartile = np.quantile(data[column], 0.75)
    IQR = third_quartile - first_quartile
    min_outlier = first_quartile - (1.5 * IQR)
    max_outlier = third_quartile + (1.5 * IQR)
    print(f'the data less than {min_outlier} and more than {max_outlier} is the outlier data in colunm {column}')
    return min_outlier, max_outlier


# # Remove The Outlier Data 

# In[5]:


def remove_outlier(column):
    min_outlier, max_outlier = find_outlier(column)
    count_min = data[column].loc[data[column]<min_outlier].count()
    count_max = data[column].loc[data[column]>max_outlier].count()
    data[column].loc[data[column]<min_outlier] = np.nan
    data[column].loc[data[column]>max_outlier] = np.nan
    count = count_min + count_max
    return count


# In[6]:


cols = []
for col in data.columns:
    if(data[col].dtype == np.int64 or data[col].dtype == np.float64):
        cols.append(col)
cols


# In[7]:


for col in cols[0:3]:
    count_outliers = remove_outlier(col)
    print(count_outliers)


# # Data Is Null?

# In[8]:


data.isna().sum()


# In[9]:


data.dropna(inplace=True)


# # Data Is Duplicated?

# In[10]:


data.duplicated().sum()


# In[11]:


data.drop_duplicates(inplace=True)


# # Show some of Information and Describe the Data

# In[12]:


data.info()


# In[13]:


round(data.describe(),2)


# # Exploratory Data Analysis

# In[14]:


data


# # Distribution of Region

# In[15]:


region_data_proportion = data['region'].value_counts()


# In[16]:


myexplode = [0.1, 0.1, 0.1, 0.1]
plt.pie(region_data_proportion, labels=region_data_proportion.index, autopct='%1.1f%%', explode= myexplode,shadow = True)
plt.show()


# In[17]:


smoker_data_proportion = data['smoker'].value_counts()


# # Distribution of Smoker

# In[18]:



myexplode = [0.1, 0.1]
plt.pie(smoker_data_proportion, labels=smoker_data_proportion.index, autopct='%1.1f%%', explode= myexplode,shadow = True, colors=['green','red'])
plt.show()


# # Distribution of Sex¶

# In[19]:


sex_data_proportion = data['sex'].value_counts()


# In[20]:


myexplode = [0.1, 0.1]
plt.pie(sex_data_proportion, labels=sex_data_proportion.index, autopct='%1.1f%%', startangle =90, explode= myexplode, shadow = True, colors=['c','hotpink'])
plt.show()


# # Count of Smokers by Sex and Region

# In[21]:


data.groupby(['region', 'sex'])['smoker'].count()


# In[22]:


pd.crosstab(index=data.region, columns=data.sex, values=data.smoker, aggfunc='count')


# In[23]:


sns.catplot(row='sex', x='smoker', col='region', data=data, kind='count')


# # Distribution of age by smokers

# In[24]:


sns.histplot(x='age', hue='smoker',data=data)


# In[25]:


sns.catplot(y='age', x='sex', hue='smoker', kind='box', data=data)
plt.show


# # Distribution of age by Children and Sex

# In[26]:


plt.figure(figsize=(18,6))
sns.violinplot(x=data.sex, y=data.age, hue=data.children)
plt.show()


# 
# # Machine Learning

# In[27]:


data.head()


# In[28]:


data_ml = data.copy()


# # Encoder Columns from Categorical to Numerical

# In[29]:



lb_encoder = LabelEncoder()
data_ml['sex'] = lb_encoder.fit_transform(data_ml.sex)
data_ml['smoker'] = lb_encoder.fit_transform(data_ml.smoker)
data_ml['region'] = lb_encoder.fit_transform(data_ml.region)


# In[30]:


data_ml


# In[31]:


sns.heatmap(data_ml.corr(), annot=True)


# # Data Slicing

# In[32]:


X = data_ml.iloc[:, :-1].values
y = data_ml.iloc[:, -1].values


# # Train and Test the Data¶

# In[33]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=98)


# In[34]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=98)


# # Linear Regression Model

# In[35]:


model_lg = LinearRegression()
model = model_lg.fit(X_train, y_train)


# In[36]:


y_pred = model.predict(X_test)


# In[37]:


results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
results


# In[38]:


sns.scatterplot(x=results.Actual, y=results.Predicted)


# # Evaluation the Model

# In[39]:


print(model.score(X_test, y_test)*100) 
print(model.score(X_train, y_train)*100) 


# In[40]:


r2_score(y_test, y_pred)*100


# # Cross Validation

# In[41]:


k_folds = KFold(n_splits = 5)

scores = cross_val_score(model, X, y, cv = k_folds)

print("Average CV Score: ", scores.mean()*100)


# # Comparing between actual values and predict values by plotting

# In[42]:


plt.plot(y_test, 'o', label='Actual')
plt.plot(y_pred, 'o', label='Prediction')
plt.legend()
plt.show()





