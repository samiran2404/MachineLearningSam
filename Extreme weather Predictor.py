"""
Title: Prediction of Extreme weather conditions based on Rapid Intensification.

Developers:
    
    1. Pushpraj Patil
    2. Nihal Agarwal
    3. Arpana Chopde
    4. Samiran Mudgalkar
    
Algorithm: Logistic Regression

Predictors:
    
1.  Name: Name of the Hurricane
2.  Status: Status of the Hurricane
3.  Latitude
4.  Longitude
5.  Date Time
6.  Maximum Wind: Maximum velocity of the Hurrciane
7.  Diff: Difference between T time wind velocity vs 
         T-6 hour wind velocity
8.  i: Unknown Parameter
9.  n: Unknown Parameter
10. Persistence: change in wind strength over the previous 
                12 hours - between T-18 and T-6 assuming that 
                the strength at T+0 is not known

11. Product: Product of the Persistence and Initial Maximum wind velocity
12. Initial.Max: The strongest wind strength of the tropical cyclone up till that point
13. Speed: speed over a 12 hour period
14. Speed_z : Zonal Component
15. Speed_m : Meridonial Compoent
16. Jday : days from the average peak (Julian) day of the season.

"""

# Importing all required packages 

from flask import Flask
import os
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from imblearn.over_sampling import SMOTE 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report,roc_curve,auc,accuracy_score

#Initializing the Flask App
app = Flask(__name__)

#Setting the directory for the input file
os.chdir('C:\\Users\m.samiran.ashok\Downloads')

#Reading the data from CSV file 
data = pd.read_csv('HURdat_ExtremeWeatherEvents.csv',index_col =0).reset_index(drop=True)


#--Preprocessing---------------------------------------------------------------
#removing the records with NA values
data = data.dropna()

#Picking up unique values of hurricanes of Atlantic Basin for future usage
hurricanes = list(data['ID'].unique())

#Storing the ID column values for future usage
Idcols = data['ID'].copy()

#Creating new columns on the basis of DateTime variable given
data['date_time'] = pd.to_datetime(data['date_time'])
data['year'] = [a.year for a in data['date_time']]
data['month'] = [a.month for a in data['date_time']]
data['day'] = [a.day for a in data['date_time']]
data['hour'] = [a.hour for a in data['date_time']]

#Setting appropriate data types for the date varaibles
data['month'] = data['month'].astype('category')
data['day'] = data['day'].astype('category')
data['hour'] = data['hour'].astype('category')
data['year'] = data['year'].astype('category')
data['YearNum'] = [item[2:4] for item in data['ID']]

#Removing the unwated columns after processing
data = data.drop(['ID','Name','date_time',
                      'speed_m','day'],axis=1)
    

#Setting appropriate data types for the other varaibles
data['Status'] = data['Status'].astype('category')
data['diff'] = data['diff'].astype('int')
data['rapid_int'] = data['rapid_int'].astype('int')
data['YearNum'] = data['YearNum'].astype('category')
data['Latitude'] = data['Latitude'].astype('int')
data['Longitude'] = data['Longitude'].astype('int')
data['Latitude_p'] = data['Latitude_p'].astype('int')
data['Longitude_p'] = data['Longitude_p'].astype('int')

#Initializing the Scaling package
sc = StandardScaler()

#Segregating the columns for Scaling
scale_data = data[['Maximum.Wind','diff','i','n','persistence','product','Initial.Max',
                   'speed','speed_z','Jday','Maximum.Wind_p','Latitude','Longitude',
                   'Latitude_p','Longitude_p']]

#Segregating the columns for Encoding
cat_data = data[['Status','month','hour']]

#Target Variable Setting
y = data['rapid_int'].copy()

#Creating Dummy variables for Encoding purpose
cat_data = pd.get_dummies(cat_data)

#Removing incorrect values from Maximum.Wind variable
scale_data['Maximum.Wind'][scale_data['Maximum.Wind'] == -99] = np.nan
scale_data['Maximum.Wind'] = scale_data['Maximum.Wind'].fillna(np.mean(scale_data['Maximum.Wind'][scale_data['Maximum.Wind'] != np.nan]))


#Feature Importances----------------------------------------------------------

#Picking up column names from both data frames for future use
scale_columns = list(scale_data.columns)
cat_columns = list(cat_data.columns)

#fitting the data to scale
sc.fit(scale_data)

#Transforming the data to scale
scale_data = sc.transform(scale_data)

#making the catgeorical data numeric into array object
cat_data = cat_data.values

#Joining the scaled and encoded data for feature importance creation
X = np.concatenate((scale_data,cat_data),axis=1)

#fitting the data to ExtraTreesClassifier to get the feature importance
feat_imp = ExtraTreesClassifier(n_estimators = 250)
feat_imp.fit(X,y)

#Creating the Feature importance data for visualization
importances = list(feat_imp.feature_importances_)
final_cols = scale_columns + cat_columns
imp_dat = pd.DataFrame(importances)
imp_dat['Cols'] = final_cols
imp_dat[0] = [a*100 for a in imp_dat[0]]

#-------------------------------------------------
#Modelling

#Data preprocessing for Modelling

Idcols = np.array(Idcols).reshape(-1,1)
y = np.array(y).reshape(-1,1)
Proc_data = np.concatenate((scale_data,cat_data,Idcols,y),axis=1)
split_ratio = round(len(hurricanes)*(0.75))
train_hurricanes = hurricanes[0:381]
test_hurricanes = hurricanes[381:]
Proc_data  = pd.DataFrame(Proc_data)
train_data = Proc_data[Proc_data[32].isin(train_hurricanes)].reset_index(drop=True)
test_data = Proc_data[Proc_data[32].isin(test_hurricanes)].reset_index(drop=True)
train_data = train_data.drop(32,axis=1)
test_data = test_data.drop(32,axis=1)

"""Creating training set and test set based on seperate occurences of 
   Hurricanes in different years"""

X_train = train_data.drop(33,axis=1)
X_test = test_data.drop(33,axis=1)

y_train = np.array(train_data[33]).reshape(-1,1).astype('int')
y_test = np.array(test_data[33]).reshape(-1,1).astype('int')

#Data to be received from the UI
uidata = X_test.iloc[[21,70,90,136],:]

#Modelling using Logistic Regression

X_train = X_train.values
X_test = X_test.values

#Oversamling of the data because the Rapid Intensification is a rare occurence
sm = SMOTE()
X_train_sm,y_train_sm = sm.fit_sample(X_train,y_train)

#Fitting the algorithm to the data
lm = LogisticRegression()
lm.fit(X_train_sm,y_train_sm)

#Predicting the values for test data evauation
y_pred = lm.predict(X_test)
y_pred_prb = lm.predict_proba(X_test)

#Evaluation Metrics-----------------------------------------------------------
# =============================================================================

# print(confusion_matrix(y_test,y_pred))
# print(classification_report(y_test,y_pred))
# 
# =============================================================================
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
"""

Flask REST API:
    
The following API calls are made in order to ensure sample POC predictions
generated for Display purpsoes

"""
#Predicting the 
ui_pred = lm.predict(uidata)

@app.route('/LUIS')
def LUIS():
    
    return json.dumps({'Prediction': str(ui_pred[0])})

@app.route('/MARCO')
def MARCO():
    
    return json.dumps({'Prediction': str(ui_pred[1])})

@app.route('/MITCH')
def MITCH():
    
    return json.dumps({'Prediction': str(ui_pred[2])})

@app.route('/ISAAC')
def ISAAC():
    
    return json.dumps({'Prediction': str(ui_pred[3])})

if __name__ == '__main__':
    app.run()
    
    
#------------------------------------------------------------------------------
