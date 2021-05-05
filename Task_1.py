#!/usr/bin/env python
# coding: utf-8

# # Task-1

# # Linear_Regression Algorithm

# ## Predicting using Supervised ML

# ### Importing standard library

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # Performimg EDA

# ### Importing Dataset

# In[2]:


#Now I will read titanic dataset using Pandas read_csv method
df=pd.read_csv('Project_1.csv')
df


# ### Getting infromation about dataset

# In[3]:


df.info()


# ### Describing dataset

# In[4]:


df.describe()
# it use to desrcibe whole data_set 
# count= it will give total count of parameter in specific columns
# mean= it will give average value of parameter in specific columns
# min= it will give minimum value of parameter in specific columns
# max= it will give maximum value of parameter in specific columns


# In[5]:


# checking dimension of dataset
df.shape


# In[6]:


# checking null value in dataset
df.isnull().sum()
# as from the output it is clear that there is no null value in dataset


# In[7]:


# checking head of dataset using .head() method
df.head()


# ### Analyazing data with the help of plot 

# In[8]:


# Plotting the distribution of scores

df.plot(x='Hours',y='Scores',style='o',c='g')
plt.title('Hour vs Percentage')
plt.xlabel('Hour')
plt.ylabel('percentage')
plt.show()


# In[9]:


# The below graphs are about compairisions of each column with the other one in the datasets 

sns.pairplot(data=df)
plt.show()


# In[10]:


# Plotting countplot with respect to Hours Column 
plt.figure(figsize=(10,5))
sns.countplot(df['Hours'])


# In[11]:


# Plotting countplot with respect to Scores Column 

plt.figure(figsize=(10,5))
sns.countplot(df['Scores'])


# # Applying Linear_Regression On Dataset

# In[12]:


# X=train data(input)   and y=test data(output)
X = df.iloc[:, :-1].values  
y = df.iloc[:, 1].values  
# here we apply indexing on X where the parameter area represents value of rows that in the data set
# column 1 (i.e, train data)
# column price in the dataset represents test data


# In[13]:


#spliting the data into train and test modules....
# we specify the spliting of dataset into various portions(here 4 various portions) using train_test_split() method.

#Printing dimensions of all the portions created from the dataset using train_test_split
# Dataset is divided into four parts:
# i) X_train    ii) X_test       iii) y_train      iv)y_test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[14]:


# Training data using .fit()
from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

print("Training complete.")


# In[15]:


# Predicting the data.....
regressor.predict([[8.5]])
# With help of training data we predict here


# In[16]:


# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X, y,c='r')
plt.plot(X, line);
plt.show()


# In[17]:


# here we predict the model output through .predict() method 
# now, machine predicts the ouput.
print(X_test)
y_pred=regressor.predict(X_test)


# In[18]:


# Comparing Actual vs Predicted
df1=pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df1


# In[19]:


#checking our model is working porperly or not ,using our own data
hours = [[9.25]]
own_pred = regressor.predict(hours)
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# In[20]:


#checking accuracy score of our model
regressor.score(X,y)


# In[21]:


# as we can observe our model have good accuracy of 95%
print("LR_model is : {}% accurate".format(regressor.score(X,y)*100))


# In[22]:


regressor


# # Thank you

# In[ ]:




