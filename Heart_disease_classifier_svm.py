
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 20:00:10 2019
Title : Heart Disease Predictor 
@author: Samiran Mudgalkar
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report,roc_curve,auc
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt


data = pd.read_csv('Heart.csv', index_col=0)

data['Sex'] = data['Sex'].astype('category')

data['ChestPain'] = data['ChestPain'].map({'asymptomatic':1, 'nonanginal':2, 'nontypical':3, 'typical':4})
data['ChestPain'] = data['ChestPain'].astype('category')

data['Fbs'] = data['Fbs'].astype('category')
data['RestECG'] = data['RestECG'].astype('category')
data['ExAng'] = data['ExAng'].astype('category')
data['Ca'] = data['Ca'].astype('category')

data['Thal'] = data['Thal'].map({'fixed':1, 'normal':2, 'reversable':3})
data['Thal']  = data['Thal'].astype('category')


data['AHD']  = data['AHD'].map({'Yes':1,'No':0})
data['AHD']  = data['AHD'].astype('category')

col_to_scale = ['Age','RestBP','Chol','MaxHR','Oldpeak','Slope']
col_to_encode = ['ChestPain','Thal','Ca','RestECG']

data = data.dropna()

scaler = StandardScaler()

ohe = OneHotEncoder(sparse=False)
scaler = StandardScaler()

X = data.drop('AHD',axis=1)

X = pd.get_dummies(X)

X_toscale = X.iloc[:,0:6]

X_encoded = X.iloc[:,6:].values

X_scaled = scaler.fit_transform(X_toscale)

X_final = np.concatenate((X_scaled,X_encoded),axis=1)

y = data['AHD']

X_train,X_test,y_train,y_test = train_test_split(X_final,y,test_size =0.20,random_state=21323)

logreg = SVC(kernel='linear',C=100,probability=True)

logreg.fit(X_train,y_train)

y_pred_sv = logreg.predict(X_test)

y_pred_score_sv = logreg.predict_proba(X_test)

confusion_matrix(y_test,y_pred_sv)

fpr_sv,tpr_sv,threshold_sv = roc_curve(y_test,y_pred_score_sv[:,1])

auc = auc(fpr_sv,tpr_sv)

# Plotting ROC

plt.title('Reciever operating characteristics')
plt.plot(fpr_sv,tpr_sv,color = 'red',label = 'AUC = %0.2f' % auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

percent_auc = auc*100


print('AUC of the model is:' + str(percent_auc) + '%')
