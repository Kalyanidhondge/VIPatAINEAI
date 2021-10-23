#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import os

from scipy.stats import shapiro
import scipy.stats as stats

#parameter settings
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[7]:


#edit the file location of raw data set
cust_df = pd.read_csv(r'C:\Users\Dkalyani\AINE AI - Intern\Telecom Data - Machine Learning\Telecom Data - Machine Learning\Telecom Data.csv')
cust_df.head()


# In[8]:


#shape of the data i.e. no. of rows and columns
cust_df.shape


# In[9]:


cust_df.info()


# In[10]:


#all the column names in the dataset
cust_df.columns


# In[11]:


#total number of null values in each column
cust_df.isnull().sum()


# In[12]:


zero_monthlyrev = len(cust_df[cust_df.MonthlyRevenue == 0].index)
zero_monthlyrev


# In[13]:


#missing value percentage
missing_val_per = cust_df.isnull().sum() / cust_df.shape[0] * 100
missing_val_per


# In[14]:


missing_val_5per = missing_val_per[missing_val_per > 5].keys()
missing_val_5per


# In[15]:


#dropping the rows with null values
cust_df = cust_df.dropna()


# In[16]:


#checking the number of null values at current state
cust_df.isnull().sum().sum()


# In[17]:


#plot box plot using pandas for columns "UniqueSubs" and "DirectorAssistedCalls"
cols=["UniqueSubs","DirectorAssistedCalls"]
cust_df.boxplot(column=cols)


# In[18]:


#top 1% of the values
qValue = [i/100 for i in range(95,101,1)] 
cust_df[cols].quantile(qValue)


# In[19]:


min_thres, max_thres = cust_df.UniqueSubs.quantile([0.000,0.95])
min_thres, max_thres


# In[20]:


#records below the minimum threshold of UniqueSubs
cust_df[cust_df.UniqueSubs<min_thres]


# In[21]:


#records above the maximum threshold of UniqueSubs
cust_df[cust_df.UniqueSubs>max_thres]


# In[22]:


#refining the the dataset with values present within the threshold limits of UniqueSubs
cust_df = cust_df[(cust_df.UniqueSubs<max_thres) & (cust_df.UniqueSubs>=min_thres)]
cust_df.shape


# In[23]:


#threshold limits for DirectorAssistedCalls
min_thres1, max_thres1 = cust_df.DirectorAssistedCalls.quantile([0.000,0.95])
min_thres1, max_thres1


# In[24]:


#refining the the dataset with values present within the threshold limits of DirectorAssistedCalls
cust_df = cust_df[(cust_df.DirectorAssistedCalls<max_thres1) & (cust_df.DirectorAssistedCalls>=min_thres1)]
cust_df.shape


# In[25]:


#checking the boxplot for outliers again
cols=["UniqueSubs","DirectorAssistedCalls"]
cust_df.boxplot(column=cols)


# In[26]:


#scatter plot to find the correlation between monthly revenue and overage minutes
sns.scatterplot(x = 'MonthlyRevenue', y = 'OverageMinutes', data = cust_df)


# In[27]:


#category plot to plot monthly revenue for each active subs category
sns.catplot(x='ActiveSubs',y ='MonthlyRevenue',data=cust_df)


# In[28]:


pd.crosstab(cust_df.CreditRating, cust_df.Churn)


# In[29]:


sns.countplot(x='Churn',hue='CreditRating',data=cust_df)


# In[30]:


# calculating correlation among numeric variable
corr_matrix = cust_df.corr()

# plot correlation matrix
plt.figure(figsize=(20,20))

sns.heatmap(corr_matrix,cmap='Spectral',annot=True);


# In[31]:


#a count check for the categorical variable
sns.countplot(x="Churn",data=cust_df)
plt.show()


# In[32]:


sns.countplot(x="Churn",hue="Occupation",data=cust_df)


# In[34]:


#wrapper function to create additional features for churn prediction
def create_features(cust_df):
    
    #3.1 Percent of current active subs over total subs
    cust_df['perc_active_subs'] = cust_df['ActiveSubs'] / cust_df['UniqueSubs']
    
    #3.2 Percent of recurrent charge to monthly charge
    #type your code here to create a new column in cust_df
    cust_df['perc_recurrent_charge'] = cust_df['TotalRecurringCharge']/cust_df['MonthlyRevenue']
    
    #3.3 Percent of overage minutes over total monthly minutes
    #type your code here to create a new column in cust_df
    cust_df['perc_avg_minutes'] = cust_df['OverageMinutes']/cust_df['MonthlyMinutes']
    
    #type your code here to creat any other additional features which you think will help improve your model accuracy
    
    
    return cust_df


# In[35]:


#adding the new features in the main dataset
cust_df=create_features(cust_df)
cust_df


# In[36]:


# List of variables to map

varlist =  ['Churn']

# Defining the map function
def binary_map(x):
    return x.map({'Yes': 1, "No": 0})

# Applying the function to the housing list
cust_df[varlist] = cust_df[varlist].apply(binary_map)



cust_df.Churn


# In[37]:


cust_df.isnull().sum()


# In[38]:


cust_df = cust_df.dropna()


# In[39]:


cust_df.isnull().sum()


# In[40]:


#replacing infinite values if any
cust_df.replace({-np.inf: -1_000_000, np.inf: 1_000_000}, inplace=True)


# In[41]:


count_class_0, count_class_1 = cust_df.Churn.value_counts()

df_class_0 = cust_df[cust_df['Churn']==0]
df_class_1 = cust_df[cust_df['Churn']==1]


# In[42]:


df_class_0.shape


# In[43]:


df_class_1.shape


# In[44]:


df_class_1_over = df_class_1.sample(count_class_0, replace=True)
df_test_over = pd.concat([df_class_0, df_class_1_over], axis=0)

print('Random over-sampling:')
print(df_test_over.Churn.value_counts())


# In[45]:


#Train - test split to train and test model accuracy
from sklearn.model_selection import train_test_split

#Define columns to be included in X and y
X = df_test_over.drop(columns=['Churn'])
#Create dummy variables for all categorical variables
X = pd.get_dummies(X)


y = df_test_over['Churn']


# In[46]:


#Feature scaling for all continuous variable
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_scaled = sc.fit_transform(X)


# In[47]:


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=0)


# In[48]:


from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()

log_reg.fit(X_train,y_train)


# In[49]:


y_pred = log_reg.predict(X_test)


# In[50]:


from sklearn.metrics import accuracy_score, confusion_matrix

#accuracy score
accuracy = accuracy_score(y_test,y_pred)
accuracy


# In[51]:


#confusion matrix
conf_mat = confusion_matrix(y_test,y_pred)
conf_mat


# In[52]:


true_positive = conf_mat[0][0]
false_positive = conf_mat[0][1]
false_negative = conf_mat[1][0]
true_negative = conf_mat[1][1]


# In[53]:


Accuracy = (true_positive + true_negative) / (true_positive +false_positive + false_negative + true_negative)
Accuracy


# In[54]:


Precision = true_positive/(true_positive+false_positive)
Precision


# In[55]:


Recall = true_positive/(true_positive+false_negative)
Recall


# In[56]:


f1=2*((Precision * Recall )) /(Precision + Recall)
f1


# In[57]:


from sklearn.metrics import  roc_curve, roc_auc_score

# Area Under Curve
auc = roc_auc_score(y_test, y_pred)
auc


# In[58]:


y_pred_prob = log_reg.predict_proba(X_test)
y_pred_prob[:,1]


# In[59]:


fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:,1])

plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--',label='ROC curve (area = %0.2f)' % auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()


# In[60]:


from sklearn.metrics import average_precision_score
average_precision = average_precision_score(y_test, y_pred)
average_precision


# In[61]:


from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve

disp = plot_precision_recall_curve(log_reg, X_test, y_test)
disp.ax_.set_title('2-class Precision-Recall curve: '
                   'AP={0:0.2f}'.format(average_precision))


# In[66]:


from sklearn.ensemble import RandomForestClassifier

# creating a RF classifier
clf = RandomForestClassifier(n_estimators = 100)  
  
# Training the model on the training dataset
# fit function is used to train the model using the training sets as parameters
clf.fit(X_train, y_train)
  
# performing predictions on the test dataset
y_pred = clf.predict(X_test)


# In[67]:


y_pred


# In[68]:


accuracy = accuracy_score(y_test,y_pred)
accuracy


# In[69]:


conf_mat = confusion_matrix(y_test,y_pred)
conf_mat


# In[70]:


true_positive = conf_mat[0][0]
false_positive = conf_mat[0][1]
false_negative = conf_mat[1][0]
true_negative = conf_mat[1][1]


# In[71]:


Accuracy = (true_positive + true_negative) / (true_positive +false_positive + false_negative + true_negative)
Accuracy


# In[72]:


Precision = true_positive/(true_positive+false_positive)
Precision


# In[73]:


Recall = true_positive/(true_positive+false_negative)
Recall


# In[74]:


f1=2*((Precision * Recall )) /(Precision + Recall)
f1


# In[75]:


# Area Under Curve
auc = roc_auc_score(y_test, y_pred)
auc


# In[76]:


y_pred_prob = log_reg.predict_proba(X_test)
y_pred_prob[:,1]


# In[77]:


fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:,1])

plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--',label='ROC curve (area = %0.2f)' % auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()


# In[78]:


average_precision = average_precision_score(y_test, y_pred)
average_precision


# In[79]:


disp = plot_precision_recall_curve(clf, X_test, y_test)
disp.ax_.set_title('2-class Precision-Recall curve: '
                   'AP={0:0.2f}'.format(average_precision))


# In[80]:


import xgboost as xgb
my_model = xgb.XGBClassifier()
my_model.fit(X_train, y_train)

y_pred = my_model.predict(X_test)


# In[81]:


accuracy = accuracy_score(y_test,y_pred)
accuracy


# In[82]:


from sklearn.tree import DecisionTreeClassifier

dt_classifier= DecisionTreeClassifier(criterion='entropy', random_state=0)  
dt_classifier.fit(X_train, y_train)  

y_pred = dt_classifier.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)


# In[83]:


accuracy


# In[84]:


pd.Series(clf.feature_importances_, index=X.columns)


# In[85]:


feat_importances = pd.Series(clf.feature_importances_, index=X.columns)
feat_importances.nlargest(20).plot(kind='barh')


# In[ ]:




