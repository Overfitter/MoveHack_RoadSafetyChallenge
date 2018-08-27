
# coding: utf-8

# # Driver Behaviour Prediction

# # Loading Libraries

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


# # Loading Events Data

# In[2]:


#loading the dataset----------------------------------------------------------------------------
event_data = pd.read_csv('./Events Report of Drivers in Ahmedabad_16-07 to 31-07/Events Report of Drivers in Ahmedabad_16-07 to 31-07.csv',
                         index_col=None,encoding = 'ISO-8859-1')


# # Checking Data

# In[3]:


#Top 5 rows of dataset
event_data.head()


# In[4]:


#check the columns names
event_data.columns


# In[5]:


#check datatypes
event_data.dtypes


# In[6]:


#Checking missing values in data
event_data.isnull().sum()


# In[7]:


#Creating a copy of original data
event_df = event_data.copy()


# In[8]:


#info about data
event_df.info()


# In[9]:


#Summary of given data
event_df.describe()


# # Data Cleaning and Preprocessing

# In[10]:


#Data cleaning-----------------------------------------------------------------------------------

#extracting day & hour variable from given date
event_df['Day'] = [datetime.strptime(d, '%d.%m.%Y %H:%M:%S').weekday() for d in event_df['Event Time']]
event_df['Hour'] = [datetime.strptime(d, '%d.%m.%Y %H:%M:%S').hour for d in event_df['Event Time']]


# In[11]:


#Renaming column names
event_df.rename(columns={'Bus no.': 'Bus_no', 'Route Name': 'Route_Name','Stop Code': 'Stop_Code','Event Unit': 'Event_Unit',
                         'Event Value (Upper limit =51km/h)': 'Event_Value','Event Name': 'Event_Name'}, inplace=True)


# # Feature Engineering

# In[12]:


#Feature Engineering-------------------------------------------------------------------------------

#creating three variables speed, acceleration ,and time by given variables Event value & Event unit
##Seconds -  Time variable
##m/s2 - Acceleration variable
##km/h - speed variable

#intialization of all three variables
event_df['Time'] = " "
event_df['Acceleration'] = " "
event_df['Speed'] = " "

event_df['Time'] = pd.to_numeric(np.where(event_df['Event_Unit']=='Seconds', event_df['Event_Value'],None))
event_df['Speed'] = pd.to_numeric(np.where(event_df['Event_Unit']=='km/h', event_df['Event_Value'],None))
event_df['Acceleration'] = pd.to_numeric(np.where(event_df['Event_Unit']=='m/s2', event_df['Event_Value'],None))


# In[13]:


#Replace missing values in time, acceleration and speed by -1
event_df.fillna(-1,inplace = True)


# In[14]:


#Converting route variable (object) to integer [Label Encoding]
cat_col = ['Route_Name']
for var in cat_col:
    le = preprocessing.LabelEncoder()
    event_df[var]=le.fit_transform(event_df[var].astype('str'))


# # Feature Analysis

# In[15]:


# Correlation matrix between numerical values (SibSp Parch Age and Fare values) and Survived 
g = sns.heatmap(event_df[['Time','Speed', 'Acceleration', 'Latitude', 'Longitude']].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")


# In[16]:


# Explore Latitude vs Event Name
g = sns.FacetGrid(event_df, col='Event_Name')
g = g.map(sns.distplot, "Latitude")


# In[17]:


# Explore longitude vs Event Name
g = sns.FacetGrid(event_df, col='Event_Name')
g = g.map(sns.distplot, "Longitude")


# In[18]:


# Explore Day vs Event Name
g = sns.FacetGrid(event_df, col='Event_Name')
g = g.map(sns.countplot, "Day")


# In[19]:


# Explore Hour vs Event Name
g = sns.FacetGrid(event_df, col='Event_Name')
g = g.map(sns.countplot, "Hour")


# In[20]:


#Boxplot of Acceleration variables
var = 'Event_Name'
data = pd.concat([event_df['Acceleration'], event_df[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y='Acceleration', data=data)
fig.axis(ymin=0, ymax=1.6);
plt.xticks(rotation=90);


# # Modeling

# In[21]:


#XGBoost function
def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=0, num_rounds=2000):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.1
    param['max_depth'] = 6
    param['silent'] = 1
    param['num_class'] = len(np.unique(event_df['Event_Name']))
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


# In[22]:


#Preparation data for xgboost model
#Coverting Event_Name (target) to numerical
target_num_map = {'Harsh Braking':0, 'Harsh Acceleration':1, 'Idling':2,'Sharp Corner':3, 'Speeding':4}
event_df['Event_Name'] = event_df['Event_Name'].apply(lambda x: target_num_map[x])

#features to use for prediction
cols_to_use = ['Time', 'Day', 'Hour', 'Speed', 'Acceleration', 'Latitude', 'Longitude','Bus_no','Route_Name','Stop_Code']


# In[23]:


train, test = train_test_split(event_df, test_size=0.3, random_state = 45)

X_train = train.loc[:,cols_to_use]
Y_train = train['Event_Name']

X_test = test.loc[:,cols_to_use]
Y_test = test['Event_Name']

print(X_train.shape)
print(X_test.shape)


# In[47]:


cv_scores = []
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2016)
for dev_index, val_index in kf.split(range(X_train.shape[0])):
        dev_X, val_X = np.array(X_train)[dev_index,:], np.array(X_train)[val_index,:]
        dev_y, val_y = np.array(Y_train)[dev_index], np.array(Y_train)[val_index]
        preds, model = runXGB(dev_X, dev_y, val_X, val_y)
        cv_scores.append(log_loss(val_y, preds))
        print(cv_scores)
        break


# In[48]:


preds, model = runXGB(X_train, Y_train, X_test, num_rounds=1295)
probs = np.ones((len(Y_test), 5))
probs = np.multiply(probs, preds)
preds = np.array([np.argmax(prob) for prob in preds])

#f1_score
score = f1_score(Y_test, preds, average='weighted')
print(score)


# In[49]:


# Confusion plot function

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


# In[52]:


# Compute confusion matrix
cnf_matrix = confusion_matrix(Y_test, preds)
np.set_printoptions(precision=2)
# Plot confusion matrix
unique_type_list = ['Harsh Braking', 'Harsh Acceleration', 'Idling','Sharp Corner', 'Speeding']
lab_encoder = LabelEncoder().fit(unique_type_list)
plot_confusion_matrix(cnf_matrix, classes=lab_encoder.inverse_transform(range(5)), normalize=True,
                      title=('Confusion matrix'))


# In[51]:


#Feature importance plot
xgb.plot_importance(model)


# In[ ]:


#Pickle the model
filename = 'model_v1_driver_behaviour.pk'
with open('./scripts/driver_behaviour/'+filename, 'wb') as file:
	pickle.dump(model, file)

