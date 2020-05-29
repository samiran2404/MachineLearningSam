# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 10:49:10 2019

@author: m.samiran.ashok
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report,roc_curve,auc,accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.decomposition import PCA


train = pd.read_csv('titanic_train.csv')
test = pd.read_csv('titanic_test.csv')

train['dataset'] = 'train'
test['dataset'] = 'test'
labels = np.array(train['Survived'])

passenger_id = np.array(test['PassengerId'].copy()).reshape(-1,1)

train  = train.drop('Survived',axis=1)

col_names = train.columns

data = pd.concat((train,test),axis=0)

data['Age'] = data['Age'].fillna(np.median(data['Age'].dropna()))

data['Embarked'] = data['Embarked'].fillna(method='ffill')

#data = data.drop('Cabin',axis=1)

data['Fare'] = data['Fare'].fillna(method='ffill')

x=[]

for item in data['Name']:
    if 'Mr.' in item:
        x.append('Mr')
    elif 'Mrs.' in item:
         x.append('Mrs')
    elif 'Miss' in item:
         x.append('Miss')
    elif 'Master' in item:
         x.append('Master')
    elif 'Rev' in item:
         x.append('Rev')
    else:
         x.append('Others')
        
data['Title'] = x

for item in data['Sex']:
    for changer in data['Title']:
        if item == 'female':
            changer.replace('Mr','Mrs')
            
data['Sex'] = data['Sex'].map({'male':0,'female':1})
data['Sex'] = data['Sex'].astype('category')

data['Embarked'] = data['Embarked'].map({'C':0,'Q':1,'S':2,})
data['Embarked'] = data['Embarked'].astype('category')

data['Title'] = data['Title'].map({'Master':0,'Miss':1,'Mr':2,'Mrs':3,'Rev':4,'Others':5})
data['Title'] = data['Title'].astype('category')

data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

data['Cabin'] = data['Cabin'].fillna('U')
data['Cabin'] = data['Cabin'].astype('category')

data['Pclass'] = data['Pclass'].astype('category')

data = data.drop(['Name','PassengerId','Ticket'],axis=1)

cat_data = data[['Sex','Embarked','Title','Pclass','Cabin']]

scale_data = data.drop(['Sex','Embarked','Title','Pclass','Cabin'],axis=1)

cat_data = pd.get_dummies(cat_data)

cat_data = cat_data.drop('Sex_1',axis=1)

identifier = list(scale_data['dataset'])

scale_data = scale_data.drop('dataset',axis=1)

scaler = StandardScaler()

scaler.fit(scale_data)

scale_data = scaler.transform(scale_data)

data_1 = np.concatenate((scale_data,cat_data),axis=1)

data_1 = pd.DataFrame(data_1)

data_1['dataset'] = identifier

train_1 = data_1[data_1['dataset'] == 'train']

test_data = data_1[data_1['dataset'] == 'test']

train_1 = train_1.drop('dataset',axis=1)
test_data = test_data.drop('dataset',axis=1)

data_2 =np.concatenate((train_1,test_data),axis=0)

data_2 = pd.DataFrame(data_2)
   
pca=PCA(n_components=36)

pca.fit(data_2)

data_3 = pca.transform(data_2)

data_3 = pd.DataFrame(data_3)

data_3['dataset'] = identifier

train_1 = data_3[data_3['dataset'] == 'train']

test_data = data_3[data_3['dataset'] == 'test']

train_1 = train_1.drop('dataset',axis=1)
test_data = test_data.drop('dataset',axis=1)

train_1 = train_1.values

scaler.fit(train_1)

train_1 = scaler.transform(train_1)

X_train,X_test,y_train,y_test = train_test_split(train_1,
                                             labels,
                                             test_size=0.25,
                                             random_state=21323)


    
logreg = SVC(kernel='linear')

logreg.fit(X_train,y_train)

y_pred = logreg.predict(X_test)

confusion_matrix(y_test,y_pred)

#y_pred_score = logreg.predict_proba(X_test)

accuracy = accuracy_score(y_test,y_pred)


print(classification_report(y_test,y_pred))

y_pred_final = logreg.predict(test_data)

passenger_id = passenger_id.ravel()

submission = pd.DataFrame({'PassengerId':passenger_id,
                           'Survived':y_pred_final})

submission.to_csv('submission_3.csv',index=False)
