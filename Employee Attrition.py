#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns


# In[2]:


df = pd.read_csv(r"C:\Users\User\Desktop\WA_Fn-UseC_-HR-Employee-Attrition.csv")


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.dtypes


# In[7]:


df.isna().sum()


# In[8]:


df.isnull().values.any()


# In[9]:


df.describe()


# In[10]:


df["Attrition"].value_counts()


# In[11]:


sns.countplot(df["Attrition"])


# In[14]:


import matplotlib.pyplot as plt
plt.subplots(figsize = (12,4))
sns.countplot(x="Age", hue="Attrition", data = df, palette="colorblind")


# In[16]:


for column in df.columns:
    if df[column].dtype==object:
        print(str(column) + " : "+str(df[column].unique()))
        print(df[column].value_counts())
        print("----------------------------")


# In[17]:


df = df.drop("Over18", axis=1)


# In[18]:


df = df.drop("EmployeeNumber", axis=1)
df = df.drop("StandardHours", axis=1)
df = df.drop("EmployeeCount", axis=1)


# In[19]:


df.corr()


# In[20]:


plt.figure(figsize=(14,14))
sns.heatmap(df.corr(), annot=True, fmt=".0%")


# In[21]:


from sklearn.preprocessing import LabelEncoder


# In[22]:


for column in df.columns:
    if df[column].dtype== np.number:
        continue
    df[column] = LabelEncoder().fit_transform(df[column])


# In[23]:


df["Age_Years"] = df["Age"]


# In[24]:


df = df.drop("Age", axis=1)


# In[25]:


X = df.iloc[:, 1:df.shape[1]].values
y = df.iloc[:, 0].values


# In[26]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)


# In[28]:


from sklearn.ensemble import RandomForestClassifier


# In[29]:


forest = RandomForestClassifier(n_estimators = 10, criterion="entropy")
forest.fit(X_train, y_train)


# In[30]:


forest.score(X_train, y_train)


# In[32]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, forest.predict(X_test))

TN = cm[0][0]
TP = cm[1][1]
FN = cm[1][0]
FP = cm[0][1]

print(cm)
print("Model Testing Accuracy = {}".format((TP+TN)/(TP+TN+FN+FP)))


# In[ ]:




