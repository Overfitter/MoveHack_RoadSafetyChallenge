
# coding: utf-8

# # Alarm Alert/Type Prediction

# # Loading Required Libraries

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
import dill as pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection, preprocessing, ensemble
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import xgboost as xgb
from datetime import datetime


# # Loading Alarm Alert Data

# In[2]:


alarm_data = pd.read_csv('./Bangalore-CAS-alerts/bangalore-cas-alerts.csv',
                         index_col=None,encoding = 'ISO-8859-1')


# # Checking Data

# In[3]:


#Top 5 rows of data
alarm_data.head()


# In[4]:


#check the columns names
alarm_data.columns


# In[5]:


#Checking data type
alarm_data.dtypes


# In[6]:


#Checking missing values in data
alarm_data.isnull().sum()


# In[7]:


#creating a copy of alarm data
alarm_df = alarm_data.copy()


# In[8]:


#info about data
alarm_df.info()


# In[9]:


#Summary of given data
alarm_df.describe()


# # Data Cleaning and Preprocessing

# In[10]:


#Data cleaning-----------------------------------------------------------------------------------
#extracting day & hour variable from given date
alarm_df['day'] = [datetime.strptime(d, '%Y-%m-%dT%H:%M:%S.000Z').weekday() for d in alarm_df['time_date']]
alarm_df['hour'] = [datetime.strptime(d, '%Y-%m-%dT%H:%M:%S.000Z').hour for d in alarm_df['time_date']]


# # Feature Engineering

# In[11]:


#creating Var1 - Mean of acceleration for every bus no.
df_1 =  alarm_df.groupby(['alarm_type','ward_name'])['speed'].mean().to_frame()
df_1.columns = ['mean_speed_ward_name']  
df_1['alarm_type'] = [df_1.index[i][0] for i in range(len(df_1))]
df_1['ward_name'] = [df_1.index[j][1] for j in range(len(df_1))]

df_1.reset_index(drop = True, inplace = True)
df_1.head()      

df_data_1 = pd.merge(alarm_df,df_1,how = 'inner',on = ['ward_name','alarm_type'],left_index = False,right_index = False)

df_data_1.head()


# In[12]:


#data preparation for modelling-----------------------------------------------------------------
#Converting route variable (object) to integer [Label Encoding]
cat_col = ['ward_name']
for var in cat_col:
    le = preprocessing.LabelEncoder()
    df_data_1[var]=le.fit_transform(df_data_1[var].astype('str'))


# # Feature Analysis

# In[13]:


# Correlation matrix between numerical values (SibSp Parch Age and Fare values) and Survived 
g = sns.heatmap(df_data_1[['speed','latitude', 'longitude','mean_speed_ward_name']].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")


# In[14]:


# Explore Latitude vs Event Name
g = sns.FacetGrid(df_data_1, col='alarm_type')
g = g.map(sns.distplot, "latitude")


# In[15]:


# Explore Latitude vs Event Name
g = sns.FacetGrid(df_data_1, col='alarm_type')
g = g.map(sns.distplot, "longitude")


# In[16]:


# Explore Latitude vs Event Name
g = sns.FacetGrid(df_data_1, col='alarm_type')
g = g.map(sns.distplot, "speed")


# In[17]:


# Explore Latitude vs Event Name
g = sns.FacetGrid(df_data_1, col='alarm_type')
g = g.map(sns.distplot, "mean_speed_ward_name")


# In[18]:


#boxplots for speed variables
var = 'alarm_type'
data = pd.concat([df_data_1['speed'], df_data_1[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y='speed', data=data)
fig.axis(ymin=0, ymax=90);
plt.xticks(rotation=90);


# In[19]:


var = 'alarm_type'
data = pd.concat([df_data_1['mean_speed_ward_name'], df_data_1[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y='mean_speed_ward_name', data=data)
fig.axis(ymin=0, ymax=40);
plt.xticks(rotation=90);


# In[20]:


# Explore Day vs Event Name
g = sns.FacetGrid(df_data_1, col='alarm_type')
g = g.map(sns.countplot, "day")


# In[21]:


# Explore Day vs Event Name
g = sns.FacetGrid(df_data_1, col='alarm_type')
g = g.map(sns.countplot, "hour")


# # Modeling

# In[22]:


#XGBoost function
def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=0, num_rounds=1000):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.1
    param['max_depth'] = 6
    param['silent'] = 1
    param['num_class'] = len(np.unique(df_data_1['alarm_type']))
    param['eval_metric'] = "mlogloss"
    param['min_child_weight'] = 1
    param['subsample'] = 0.7
    param['colsample_bytree'] = 0.7
    param['seed'] = seed_val
    num_rounds = num_rounds

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=20)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds)

    pred_test_y = model.predict(xgtest)
    return pred_test_y, model


# In[23]:


#Preparation data for xgboost model
#Coverting event name (target) to numerical
target_num_map = {'FCW':0, 'HMW':1, 'LDWL':2, 'LDWR':3, 'Overspeed':4, 'PCW':5, 'UFCW':6}
df_data_1['alarm_type'] = df_data_1['alarm_type'].apply(lambda x: target_num_map[x])


# In[24]:


#Splitting the dataset into traninig and testing

cols_to_use = ['day','hour','speed','latitude', 'longitude','mean_speed_ward_name']


train, test = train_test_split(df_data_1, test_size=0.3,random_state = 100)

X_train = train.loc[:,cols_to_use]
Y_train = train['alarm_type']

X_test = test.loc[:,cols_to_use]
Y_test = test['alarm_type']

print(X_train.shape)
print(X_test.shape)


# In[25]:


cv_scores = []
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=216)
for dev_index, val_index in kf.split(range(X_train.shape[0])):
        dev_X, val_X = np.array(X_train)[dev_index,:], np.array(X_train)[val_index,:]
        dev_y, val_y = np.array(Y_train)[dev_index], np.array(Y_train)[val_index]
        preds, model = runXGB(dev_X, dev_y, val_X, val_y)
        cv_scores.append(log_loss(val_y, preds))
        print(cv_scores)
        break


# In[26]:


preds, model = runXGB(X_train, Y_train, X_test, num_rounds=400)
probs = np.ones((len(Y_test), 7))
probs = np.multiply(probs, preds)
preds = np.array([np.argmax(prob) for prob in preds])

#f1_score
score = f1_score(Y_test, preds, average='weighted')
print(score)


# In[27]:


# Confusion plot

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[29]:


# Compute confusion matrix
cnf_matrix = confusion_matrix(Y_test, preds)
np.set_printoptions(precision=2)
# Plot confusion matrix
unique_type_list = ['FCW', 'HMW', 'LDWL', 'LDWR', 'Overspeed', 'PCW	','UFCW']
lab_encoder = LabelEncoder().fit(unique_type_list)
plot_confusion_matrix(cnf_matrix, classes=lab_encoder.inverse_transform(range(7)), normalize=True,
                      title=('Confusion matrix'))


# In[30]:


#Variable importance by xgboost
xgb.plot_importance(model)


# In[31]:


#Pickle the model
filename = 'model_v1_alarm_alert.pk'
with open('./scripts/alarm_alert/'+filename, 'wb') as file:
	pickle.dump(model, file)

