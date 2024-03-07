#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score


# In[2]:


get_ipython().system('pip install imbalanced-learn')


# In[80]:


get_ipython().system('pip install xgboost')


# In[3]:


df = pd.read_csv("C:/Users/thana/Desktop/Py/ml/lab3DT/seattle-weather.csv")
df


# In[4]:


from datetime import datetime
arr_date = df['date']
season = np.zeros(df.shape[0])
duration = len(arr_date)
for i in range(duration):
    date = datetime.strptime(arr_date[i], '%Y-%m-%d')
    month = date.month
    if month == 12 or month == 1 or month == 2 or month == 3:
        season[i] = 0 #winter
    if month == 4 or month == 5 or month == 6:
        season[i] = 1 #spring
    if month == 7 or month == 8:
        season[i] = 2 #summer
    if month == 9 or month == 10 or month == 11:
        season[i] = 3 #autumn
df['Encode Season'] = season


# In[5]:


df.drop('date', inplace=True, axis=1)


# In[6]:


df.tail(50)


# In[7]:


countrain=len(df[df.weather=="rain"])
countsun=len(df[df.weather=="sun"])
countdrizzle=len(df[df.weather=="drizzle"])
countsnow=len(df[df.weather=="snow"])
countfog=len(df[df.weather=="fog"])
print('rain =',countrain)
print('sun =',countsun)
print('drizzle =',countdrizzle)
print('fog =',countfog)
print('snow =',countsnow)
print('total =',df['weather'].count())


# In[8]:


from sklearn import preprocessing

#creating labelEncoder
number = preprocessing.LabelEncoder()
df['weather'] = number.fit_transform(df['weather'])


# In[9]:


df.head()


# In[10]:


x=((df.loc[:,df.columns!="weather"]).astype(int)).values[:,0:]
y=df["weather"].values


# In[11]:


# แก้ไขตัว target ที่ขาดความหลากหลายด้วย SMOTE Algorithm
from imblearn.over_sampling import SMOTE 
sm = SMOTE(random_state = 2) 
X_train_res, y_train_res = sm.fit_resample(x, y) 


# In[12]:


y_train_res


# In[13]:


print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape)) 
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape)) 
  
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res == 0))) 
print("After OverSampling, counts of label '1': {}".format(sum(y_train_res == 1))) 
print("After OverSampling, counts of label '2': {}".format(sum(y_train_res == 2))) 
print("After OverSampling, counts of label '3': {}".format(sum(y_train_res == 3))) 
print("After OverSampling, counts of label '4': {}".format(sum(y_train_res == 4))) 


# In[14]:


X_train,X_test,y_train,y_test=train_test_split(X_train_res,y_train_res,test_size=0.2,random_state=2)


# In[18]:


#KNeighborsClassifier
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
max = 0
imax = 0
for i in range(30):
    knn00 = KNeighborsClassifier(n_neighbors=i+1)
    knn00.fit(X_train,y_train)
    y_pred = knn00.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred,average='macro')
    recall = metrics.recall_score(y_test, y_pred,average='macro')
    F1 = metrics.f1_score(y_test, y_pred,average='macro')
    if max<accuracy:
        max = accuracy
        imax = i+1
    print(i+1,"Accuracy:",accuracy)
    print(i+1,"Precision:", precision)
    print(i+1,"Recall:", recall)
    print(i+1,"F1-score:", F1)
    print("\n")
print('i max = ',imax)
print('acc max = ',max)


# In[19]:


import matplotlib.pyplot as plt
import numpy
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
knn00 = KNeighborsClassifier(n_neighbors=6)
knn00.fit(X_train,y_train)
matrix = plot_confusion_matrix(knn00, X_test, y_test, cmap=plt.cm.Blues)
matrix.ax_.set_title('Confusion Matrix')
plt.show()


# In[62]:


#DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
max = 0
imax = 0
for i in range(30):
    DTC = tree.DecisionTreeClassifier(max_depth=i+1)
    DTC = DTC.fit(X_train,y_train)
    y_pred = DTC.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred,average='macro')
    recall = metrics.recall_score(y_test, y_pred,average='macro')
    F1 = metrics.f1_score(y_test, y_pred,average='macro')
    if max<accuracy:
        max = accuracy
        imax = i+1
    print(i+1,"Accuracy:",accuracy)
    print(i+1,"Precision:", precision)
    print(i+1,"Recall:", recall)
    print(i+1,"F1-score:", F1)
    print("\n")
print('i max = ',imax)
print('acc max = ',max)


# In[20]:


import matplotlib.pyplot as plt
import numpy
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
DTC = tree.DecisionTreeClassifier(max_depth=22)
DTC = DTC.fit(X_train,y_train)
matrix = plot_confusion_matrix(DTC, X_test, y_test, cmap=plt.cm.Blues)
matrix.ax_.set_title('Confusion Matrix')
plt.show()


# In[21]:


#SVM
from sklearn import svm
SVM = svm.SVC()
SVM = SVM.fit(X_train, y_train)
y_pred = SVM.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred,average='weighted')
recall = metrics.recall_score(y_test, y_pred,average='weighted')
F1 = metrics.f1_score(y_test, y_pred,average='weighted')
print("Accuracy:",accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", F1)


# In[22]:


import matplotlib.pyplot as plt
import numpy
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
matrix = plot_confusion_matrix(SVM, X_test, y_test, cmap=plt.cm.Blues)
matrix.ax_.set_title('Confusion Matrix')
plt.show()


# In[23]:


from sklearn.linear_model import Ridge
import numpy as np
clf = Ridge(alpha=1.0)
clf.fit(x,y)
clf.coef_


# In[24]:


#XGB
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')
xgb=XGBClassifier()
xgb=xgb.fit(X_train,y_train)
y_pred = xgb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred,average='weighted')
recall = metrics.recall_score(y_test, y_pred,average='weighted')
F1 = metrics.f1_score(y_test, y_pred,average='weighted')
print("Accuracy:",accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", F1)


# In[25]:


import matplotlib.pyplot as plt
import numpy
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
matrix = plot_confusion_matrix(xgb, X_test, y_test, cmap=plt.cm.Blues)
matrix.ax_.set_title('Confusion Matrix')
plt.show()


# In[29]:


X_train,X_test,y_train,y_test=train_test_split(X_train_res,y_train_res,test_size=0.2,random_state=2)


# In[34]:


#LogisticRegression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
# fit the model on the whole dataset
model = model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred,average='weighted')
recall = metrics.recall_score(y_test, y_pred,average='weighted')
F1 = metrics.f1_score(y_test, y_pred,average='weighted')
print("Accuracy:",accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", F1)


# In[35]:


import matplotlib.pyplot as plt
import numpy
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
matrix = plot_confusion_matrix(model, X_test, y_test, cmap=plt.cm.Blues)
matrix.ax_.set_title('Confusion Matrix')
plt.show()


# In[ ]:




