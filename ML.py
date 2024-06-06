#!/usr/bin/env python
# coding: utf-8

# <p style="text-align:center">
#     <a href="https://skills.network/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkML0101ENSkillsNetwork1047-2023-01-01">
#     <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/assets/logos/SN_web_lightmode.png" width="200" alt="Skills Network Logo"  />
#     </a>
# </p>
# 
# 
# # Classification with Python
# 
# 
# Estimated time needed: **25** minutes
#     
# 
# ## Objectives
# 
# After completing this lab you will be able to:
# 
# * Confidently create classification models
# 

# In this notebook we try to practice all the classification algorithms that we learned in this course.
# 
# We load a dataset using Pandas library, apply the following algorithms, and find the best one for this specific dataset by accuracy evaluation methods.
# 
# Let's first load required libraries:
# 

# In[1]:


import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# ### About dataset
# 

# This dataset is about the performance of basketball teams. The __cbb.csv__ data set includes performance data about five seasons of 354 basketball teams. It includes the following fields:
# 
# | Field          | Description                                                                           |
# |----------------|---------------------------------------------------------------------------------------|
# |TEAM |	The Division I college basketball school|
# |CONF|	The Athletic Conference in which the school participates in (A10 = Atlantic 10, ACC = Atlantic Coast Conference, AE = America East, Amer = American, ASun = ASUN, B10 = Big Ten, B12 = Big 12, BE = Big East, BSky = Big Sky, BSth = Big South, BW = Big West, CAA = Colonial Athletic Association, CUSA = Conference USA, Horz = Horizon League, Ivy = Ivy League, MAAC = Metro Atlantic Athletic Conference, MAC = Mid-American Conference, MEAC = Mid-Eastern Athletic Conference, MVC = Missouri Valley Conference, MWC = Mountain West, NEC = Northeast Conference, OVC = Ohio Valley Conference, P12 = Pac-12, Pat = Patriot League, SB = Sun Belt, SC = Southern Conference, SEC = South Eastern Conference, Slnd = Southland Conference, Sum = Summit League, SWAC = Southwestern Athletic Conference, WAC = Western Athletic Conference, WCC = West Coast Conference)|
# |G|	Number of games played|
# |W|	Number of games won|
# |ADJOE|	Adjusted Offensive Efficiency (An estimate of the offensive efficiency (points scored per 100 possessions) a team would have against the average Division I defense)|
# |ADJDE|	Adjusted Defensive Efficiency (An estimate of the defensive efficiency (points allowed per 100 possessions) a team would have against the average Division I offense)|
# |BARTHAG|	Power Rating (Chance of beating an average Division I team)|
# |EFG_O|	Effective Field Goal Percentage Shot|
# |EFG_D|	Effective Field Goal Percentage Allowed|
# |TOR|	Turnover Percentage Allowed (Turnover Rate)|
# |TORD|	Turnover Percentage Committed (Steal Rate)|
# |ORB|	Offensive Rebound Percentage|
# |DRB|	Defensive Rebound Percentage|
# |FTR|	Free Throw Rate (How often the given team shoots Free Throws)|
# |FTRD|	Free Throw Rate Allowed|
# |2P_O|	Two-Point Shooting Percentage|
# |2P_D|	Two-Point Shooting Percentage Allowed|
# |3P_O|	Three-Point Shooting Percentage|
# |3P_D|	Three-Point Shooting Percentage Allowed|
# |ADJ_T|	Adjusted Tempo (An estimate of the tempo (possessions per 40 minutes) a team would have against the team that wants to play at an average Division I tempo)|
# |WAB|	Wins Above Bubble (The bubble refers to the cut off between making the NCAA March Madness Tournament and not making it)|
# |POSTSEASON|	Round where the given team was eliminated or where their season ended (R68 = First Four, R64 = Round of 64, R32 = Round of 32, S16 = Sweet Sixteen, E8 = Elite Eight, F4 = Final Four, 2ND = Runner-up, Champion = Winner of the NCAA March Madness Tournament for that given year)|
# |SEED|	Seed in the NCAA March Madness Tournament|
# |YEAR|	Season
# 

# ### Load Data From CSV File  
# 

# Let's load the dataset [NB Need to provide link to csv file]
# 

# In[2]:


df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%206/cbb.csv')
df.head()


# In[3]:


df.shape


# ## Add Column
# Next we'll add a column that will contain "true" if the wins above bubble are over 7 and "false" if not. We'll call this column Win Index or "windex" for short. 
# 

# In[4]:


df['windex'] = np.where(df.WAB > 7, 'True', 'False')


# # Data visualization and pre-processing
# 
# 

# Next we'll filter the data set to the teams that made the Sweet Sixteen, the Elite Eight, and the Final Four in the post season. We'll also create a new dataframe that will hold the values with the new column.
# 

# In[5]:


df1 = df.loc[df['POSTSEASON'].str.contains('F4|S16|E8', na=False)]
df1.head()


# In[6]:


df1['POSTSEASON'].value_counts()


# 32 teams made it into the Sweet Sixteen, 16 into the Elite Eight, and 8 made it into the Final Four over 5 seasons. 
# 

# Lets plot some columns to underestand the data better:
# 

# In[7]:


# notice: installing seaborn might takes a few minutes
get_ipython().system('conda install -c anaconda seaborn -y')


# In[8]:


import seaborn as sns

bins = np.linspace(df1.BARTHAG.min(), df1.BARTHAG.max(), 10)
g = sns.FacetGrid(df1, col="windex", hue="POSTSEASON", palette="Set1", col_wrap=6)
g.map(plt.hist, 'BARTHAG', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# In[9]:


bins = np.linspace(df1.ADJOE.min(), df1.ADJOE.max(), 10)
g = sns.FacetGrid(df1, col="windex", hue="POSTSEASON", palette="Set1", col_wrap=2)
g.map(plt.hist, 'ADJOE', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# # Pre-processing:  Feature selection/extraction
# 

# ### Lets look at how Adjusted Defense Efficiency plots
# 

# In[10]:


bins = np.linspace(df1.ADJDE.min(), df1.ADJDE.max(), 10)
g = sns.FacetGrid(df1, col="windex", hue="POSTSEASON", palette="Set1", col_wrap=2)
g.map(plt.hist, 'ADJDE', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()


# We see that this data point doesn't impact the ability of a team to get into the Final Four. 
# 

# ## Convert Categorical features to numerical values
# 

# Lets look at the postseason:
# 

# In[11]:


df1.groupby(['windex'])['POSTSEASON'].value_counts(normalize=True)


# 13% of teams with 6 or less wins above bubble make it into the final four while 17% of teams with 7 or more do.
# 

# Lets convert wins above bubble (winindex) under 7 to 0 and over 7 to 1:
# 

# In[12]:


df1['windex'].replace(to_replace=['False','True'], value=[0,1],inplace=True)
df1.head()


# ### Feature selection
# 

# Let's define feature sets, X:
# 

# In[13]:


X = df1[['G', 'W', 'ADJOE', 'ADJDE', 'BARTHAG', 'EFG_O', 'EFG_D',
       'TOR', 'TORD', 'ORB', 'DRB', 'FTR', 'FTRD', '2P_O', '2P_D', '3P_O',
       '3P_D', 'ADJ_T', 'WAB', 'SEED', 'windex']]
X[0:5]


# What are our lables? Round where the given team was eliminated or where their season ended (R68 = First Four, R64 = Round of 64, R32 = Round of 32, S16 = Sweet Sixteen, E8 = Elite Eight, F4 = Final Four, 2ND = Runner-up, Champion = Winner of the NCAA March Madness Tournament for that given year)|
# 

# In[14]:


y = df1['POSTSEASON'].values
y[0:5]


# ## Normalize Data 
# 

# Data Standardization gives data zero mean and unit variance (technically should be done after train test split )
# 

# In[15]:


X= preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# ## Training and Validation 
# 

# Split the data into Training and Validation data.
# 

# In[16]:


# We split the X into train and test to find the best k
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Validation set:', X_val.shape,  y_val.shape)


# # Classification 
# 

# Now, it is your turn, use the training set to build an accurate model. Then use the validation set  to report the accuracy of the model
# You should use the following algorithm:
# - K Nearest Neighbor(KNN)
# - Decision Tree
# - Support Vector Machine
# - Logistic Regression
# 
# 

# # K Nearest Neighbor(KNN)
# 
# <b>Question  1 </b> Build a KNN model using a value of k equals five, find the accuracy on the validation data (X_val and y_val)
# 

# You can use <code> accuracy_score</cdoe>
# 

# In[19]:


from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


# <b>Question  2</b> Determine and print the accuracy for the first 15 values of k on the validation data:
# 

# In[21]:


knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_val = knn_model.predict(X_val)
accuracy_val = accuracy_score(y_val, y_pred_val)
print(f"Validation Accuracy using KNN (k=5): {accuracy_val:.4f}")



# # Decision Tree
# 

# The following lines of code fit a <code>DecisionTreeClassifier</code>:
# 

# In[22]:


from sklearn.tree import DecisionTreeClassifier


# <b>Question  3</b> Determine the minumum   value for the parameter <code>max_depth</code> that improves results 
# 

# In[26]:


decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(X_train, y_train)
y_pred_val_dt = decision_tree_model.predict(X_val)
accuracy_val_dt = accuracy_score(y_val, y_pred_val_dt)
print(f"Validation Accuracy using Decision Tree: {accuracy_val_dt:.4f}")


# # Support Vector Machine
# 

# <b>Question  4</b> Train the support  vector machine model and determine the accuracy on the validation data for each kernel. Find the kernel (linear, poly, rbf, sigmoid) that provides the best score on the validation data and train a SVM using it.
# 

# In[ ]:


from sklearn import svm


# In[24]:


kernels = ['linear', 'poly', 'rbf', 'sigmoid']
best_kernel = None
best_accuracy = 0.0

for kernel in kernels:
    svm_model = SVC(kernel=kernel)
    svm_model.fit(X_train, y_train)
    y_pred_val_svm = svm_model.predict(X_val)
    accuracy_val_svm = accuracy_score(y_val, y_pred_val_svm)
    print(f"Validation Accuracy using SVM ({kernel} kernel): {accuracy_val_svm:.4f}")

    if accuracy_val_svm > best_accuracy:
        best_accuracy = accuracy_val_svm
        best_kernel = kernel

print(f"Best SVM kernel: {best_kernel}")


# # Logistic Regression
# 

# <b>Question 5</b> Train a logistic regression model and determine the accuracy of the validation data (set C=0.01)
# 

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[25]:


lr_model = LogisticRegression(C=0.01, solver='liblinear')
lr_model.fit(X_train, y_train)
y_pred_val_lr = lr_model.predict(X_val)
accuracy_val_lr = accuracy_score(y_val, y_pred_val_lr)
print(f"Validation Accuracy using Logistic Regression: {accuracy_val_lr:.4f}")


# # Model Evaluation using Test set
# 

# In[ ]:


from sklearn.metrics import f1_score
# for f1_score please set the average parameter to 'micro'
from sklearn.metrics import log_loss


# In[ ]:


def jaccard_index(predictions, true):
    if (len(predictions) == len(true)):
        intersect = 0;
        for x,y in zip(predictions, true):
            if (x == y):
                intersect += 1
        return intersect / (len(predictions) + len(true) - intersect)
    else:
        return -1


# <b>Question  5</b> Calculate the  F1 score and Jaccard score for each model from above. Use the Hyperparameter that performed best on the validation data. **For f1_score please set the average parameter to 'micro'.**
# 

# ### Load Test set for evaluation 
# 

# In[27]:


test_df = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0120ENv3/Dataset/ML0101EN_EDX_skill_up/basketball_train.csv',error_bad_lines=False)
test_df.head()


# In[28]:


test_df['windex'] = np.where(test_df.WAB > 7, 'True', 'False')
test_df1 = test_df[test_df['POSTSEASON'].str.contains('F4|S16|E8', na=False)]
test_Feature = test_df1[['G', 'W', 'ADJOE', 'ADJDE', 'BARTHAG', 'EFG_O', 'EFG_D',
       'TOR', 'TORD', 'ORB', 'DRB', 'FTR', 'FTRD', '2P_O', '2P_D', '3P_O',
       '3P_D', 'ADJ_T', 'WAB', 'SEED', 'windex']]
test_Feature['windex'].replace(to_replace=['False','True'], value=[0,1],inplace=True)
test_X=test_Feature
test_X= preprocessing.StandardScaler().fit(test_X).transform(test_X)
test_X[0:5]


# In[29]:


test_y = test_df1['POSTSEASON'].values
test_y[0:5]


# KNN
# 

# In[47]:


from sklearn.metrics import f1_score, jaccard_score
f1_knn = f1_score(test_y, y_pred_test_knn, average='micro')
jaccard_knn = jaccard_score(test_y, y_pred_test_knn, average='micro')
print(f"F1 Score (KNN): {f1_knn:.4f}")
print(f"Jaccard Score (KNN): {jaccard_knn:.4f}")


# Decision Tree
# 

# In[38]:


f1_dt = f1_score(test_y, y_pred_test_dt, average='micro')
jaccard_dt = jaccard_score(test_y, y_pred_test_dt, average='micro')
print(f"F1 Score (Decision Tree): {f1_dt:.4f}")
print(f"Jaccard Score (Decision Tree): {jaccard_dt:.4f}")


# SVM
# 

# In[39]:


f1_svm = f1_score(test_y, y_pred_test_svm, average='micro')
jaccard_svm = jaccard_score(test_y, y_pred_test_svm, average='micro')
print(f"F1 Score (SVM): {f1_svm:.4f}")
print(f"Jaccard Score (SVM): {jaccard_svm:.4f}")


# Logistic Regression
# 

# In[46]:


f1_lr = f1_score(test_y, y_pred_test_lr, average='micro')
jaccard_lr = jaccard_score(test_y, y_pred_test_lr, average='micro')
print(f"F1 Score (Logistic Regression): {f1_lr:.4f}")
print(f"Jaccard Score (Logistic Regression): {jaccard_lr:.4f}")


# # Report
# You should be able to report the accuracy of the built model using different evaluation metrics:
# 

# | Algorithm          | Accuracy | Jaccard  | F1-score  | LogLoss |
# |--------------------|----------|----------|-----------|---------|
# | KNN                |   0.6667 |  0.4583  |  0.6286   | NA      |
# | Decision Tree      |   0.4167 |  0.5909  |  0.7429   | NA      |
# | SVM                |   0.6667 |  0.5217  |  0.6857   | NA      |
# | LogisticRegression |   0.5833 |  0.5217  |  0.6857   | 0.6931  |
# 

# Something to keep in mind when creating models to predict the results of basketball tournaments or sports in general is that is quite hard due to so many factors influencing the game. Even in sports betting an accuracy of 55% and over is considered good as it indicates profits.
# 

# <h2>Want to learn more?</h2>
# 
# IBM SPSS Modeler is a comprehensive analytics platform that has many machine learning algorithms. It has been designed to bring predictive intelligence to decisions made by individuals, by groups, by systems – by your enterprise as a whole. A free trial is available through this course, available here: <a href="https://www.ibm.com/analytics/spss-statistics-software?utm_source=Exinfluencer&utm_content=000026UJ&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkML0101ENSkillsNetwork1047-2023-01-01&utm_medium=Exinfluencer&utm_term=10006555">SPSS Modeler</a>
# 
# Also, you can use Watson Studio to run these notebooks faster with bigger datasets. Watson Studio is IBM's leading cloud solution for data scientists, built by data scientists. With Jupyter notebooks, RStudio, Apache Spark and popular libraries pre-packaged in the cloud, Watson Studio enables data scientists to collaborate on their projects without having to install anything. Join the fast-growing community of Watson Studio users today with a free account at <a href="https://www.ibm.com/cloud/watson-studio?utm_source=Exinfluencer&utm_content=000026UJ&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkML0101ENSkillsNetwork1047-2023-01-01&utm_medium=Exinfluencer&utm_term=10006555">Watson Studio</a>
# 
# 

# ### Thank you for completing this lab!
# 
# 
# ## Author
# 
# Saeed Aghabozorgi
# 
# 
# ### Other Contributors
# 
# <a href="https://www.linkedin.com/in/joseph-s-50398b136/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkML0101ENSkillsNetwork1047-2023-01-01">Joseph Santarcangelo</a>
# 
# 
# 
# 
# ## Change Log
# 
# 
# |  Date (YYYY-MM-DD) |  Version | Changed By  |  Change Description |
# |---|---|---|---|
# |2021-04-03   | 2.1  | Malika Singla| Updated the Report accuracy |
# | 2020-08-27  | 2.0  | Lavanya  |  Moved lab to course repo in GitLab |
# |   |   |   |   |
# |   |   |   |   |
# 
# 
# ## <h3 align="center"> © IBM Corporation 2020. All rights reserved. <h3/>
# 
