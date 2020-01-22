# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 10:25:11 2019

@author: m.samiran.ashok
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date,datetime
from nltk.tokenize import RegexpTokenizer
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from nltk import word_tokenize
import itertools
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report,roc_curve,auc,accuracy_score
sns.set()

train = pd.read_csv('loan_train.csv',parse_dates = ['Date.of.Birth','DisbursalDate',])
test = pd.read_csv('loan_test.csv',parse_dates = ['Date.of.Birth','DisbursalDate',])

unique_id = test['UniqueID'].copy()

train['dataset'] = 'train'
test['dataset'] = 'test'

#train['loan_default'] = train['loan_default'].astype('category')

labels = train['loan_default']


def getDate(x):
    
    y = list(x)
    age_1 =[]
    
    today = date.today()
    
    for i in range(0, len(y)):
        age = today.year - y[i].year
        age_1.append(age)  
    return age_1 

def get_proc(x):
    acc_age = x

    tokens = [word.split() for word in acc_age]
    
    tok = tokens[0]
    
    final_tokens = []
    
    for i in range(0,len(tokens)):
        tok = tokens[i]
        token=[]
        for i in range(0,len(tok)):
            t1 = list(tok[i])
            token.append(t1)
        final_tokens.append(token)
        
    
    final_tokens_1 = [list(itertools.chain.from_iterable(word)) for word in final_tokens]
    
    final_tokens_2= []
    for i in range(0,len(final_tokens_1)):    
       tok_1= [s for s in final_tokens_1[i] if s.isdigit()]
       final_tokens_2.append(tok_1)
    
    final_parse = []
    for i in range(0,len(final_tokens_2)):  
        fin = [int(word) for word in final_tokens_2[i]]    
        final_parse.append(fin)
        
    fin_1 = [word[0]*12 for word in final_parse]
    fin_2 = [word[1]*10 if len(word) > 2 else word[1] for word in final_parse]
    fin_3 = [word[-1] if len(word) > 2 else 0 for word in final_parse]
    
    fin_1 = np.array(fin_1)
    fin_2 = np.array(fin_2)
    fin_3 = np.array(fin_3)
    
    fin_list = fin_1+fin_2+fin_3
    
    fin_list = list(fin_list)
    
    return fin_list

def getDays(x):
    y = list(x)
    days = []
    for i in range(0,len(y)):
        d1 = datetime.strptime(str(date.today()),"%Y-%m-%d")
        d2 = datetime.strptime(str(y[i]),"%Y-%m-%d %H:%M:%S")
        day = abs((d1-d2).days)
        days.append(day)
    
    return days

train = train.drop('loan_default',axis=1)

data = pd.concat((train,test),axis=0)

#--------------------------------------------------------------------

data['Age'] = getDate(data['Date.of.Birth'])
data['Acc_age'] = get_proc(data['AVERAGE.ACCT.AGE'])
data['credit_hist_len'] = get_proc(data['CREDIT.HISTORY.LENGTH'])
data['Time since disbursal'] = getDays(data['DisbursalDate'])

data['Employment.Type'] = data['Employment.Type'].fillna(method='ffill')


data = data.drop(['UniqueID','Date.of.Birth','DisbursalDate',
                  'supplier_id','Current_pincode_ID','Employee_code_ID','MobileNo_Avl_Flag',
                  'AVERAGE.ACCT.AGE','CREDIT.HISTORY.LENGTH','Time since disbursal',
                  'SEC.NO.OF.ACCTS','SEC.ACTIVE.ACCTS','SEC.OVERDUE.ACCTS','SEC.CURRENT.BALANCE',
                  'SEC.SANCTIONED.AMOUNT','SEC.DISBURSED.AMOUNT','PRI.NO.OF.ACCTS',
                  'PRI.ACTIVE.ACCTS','PRI.OVERDUE.ACCTS'],axis=1)
    

data['Age'] = [s+100 if s<0 else s for s in data['Age']]

data['branch_id'] = data['branch_id'].astype('category')
data['manufacturer_id'] = data['manufacturer_id'].astype('category')
data['Employment.Type'] = data['Employment.Type'].astype('category')
data['State_ID'] = data['State_ID'].astype('category')
data['Aadhar_flag'] = data['Aadhar_flag'].astype('category')
data['PAN_flag'] = data['PAN_flag'].astype('category')
data['VoterID_flag'] = data['VoterID_flag'].astype('category')
data['Driving_flag'] = data['Driving_flag'].astype('category')
data['Passport_flag'] = data['Passport_flag'].astype('category')
data['PERFORM_CNS.SCORE.DESCRIPTION'] = data['PERFORM_CNS.SCORE.DESCRIPTION'].astype('category')

#--------------------------------------------------------------------------------------

data = data.reset_index()
dataset = data['dataset']

data = data.drop('dataset',axis=1)


#---------------------------------------------------------------------------------------
label_enc = LabelEncoder()

cat_1 = data[['branch_id','manufacturer_id','Employment.Type',
                    'State_ID','Aadhar_flag','PAN_flag','VoterID_flag',
                    'Driving_flag','Passport_flag','PERFORM_CNS.SCORE.DESCRIPTION']]

scale_1 = data.drop(['branch_id','manufacturer_id','Employment.Type',
                           'State_ID','Aadhar_flag','PAN_flag','VoterID_flag',
                           'Driving_flag','Passport_flag','PERFORM_CNS.SCORE.DESCRIPTION'],axis=1)

cat_1['branch_id'] = label_enc.fit_transform(cat_1['branch_id'])
cat_1['manufacturer_id'] = label_enc.fit_transform(cat_1['manufacturer_id'])
cat_1['Employment.Type'] = label_enc.fit_transform(cat_1['Employment.Type'])
cat_1['State_ID'] = label_enc.fit_transform(cat_1['State_ID'])
cat_1['PERFORM_CNS.SCORE.DESCRIPTION'] = label_enc.fit_transform(cat_1['PERFORM_CNS.SCORE.DESCRIPTION'])
cat_1['branch_id'] = cat_1['branch_id'].astype('category') 
cat_1['manufacturer_id'] = cat_1['manufacturer_id'].astype('category')
cat_1['Employment.Type'] = cat_1['Employment.Type'].astype('category')
cat_1['State_ID'] = cat_1['State_ID'].astype('category')
cat_1['PERFORM_CNS.SCORE.DESCRIPTION'] = cat_1['PERFORM_CNS.SCORE.DESCRIPTION'].astype('category')

cat_1 = pd.get_dummies(cat_1)

cat_1 = cat_1.drop(['Aadhar_flag_1','PAN_flag_1','VoterID_flag_1','Driving_flag_1','Passport_flag_1'],axis=1)

scaler = StandardScaler()

scaler.fit(scale_1)

scale_1 = scaler.transform(scale_1)

scale_1 = pd.DataFrame(scale_1)

data_1 = pd.concat((cat_1,scale_1),axis=1)

data = data_1

data['dataset'] = dataset
#-------------------------
#PCA
data = data.drop('dataset',axis=1)

pca = PCA(n_components = 150)

pca.fit(data)

data = pca.transform(data)

data = pd.DataFrame(data)

data['dataset'] = dataset
#-------------------------
train_data = data[data['dataset'] == 'train']
test_data = data[data['dataset'] == 'test']


train_data = train_data.drop('dataset',axis=1)
test_data = test_data.drop('dataset',axis=1)

train_data = train_data.values
test_data = test_data.values

X_train,X_test,y_train,y_test = train_test_split(train_data,labels,test_size=0.25,random_state=23)

smote = RandomUnderSampler()

X_balanced,y_balanced = smote.fit_sample(X_train,y_train)

xgb = XGBClassifier(n_estimators=150,max_depth=3)

xgb.fit(X_balanced,y_balanced)

y_pred = xgb.predict(X_test)

y_pred_score  = xgb.predict_proba(X_test)

fpr,tpr,threshold = roc_curve(y_pred,y_pred_score[:,1])

auc = auc(fpr,tpr)

# Plotting ROC

plt.title('Reciever operating characteristics')
plt.plot(fpr,tpr,color = 'red',label = 'AUC = %0.2f' % auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

percent_auc = auc*100


print('AUC of you model is:' + str(percent_auc) + '%')

print(classification_report(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

y_final = xgb.predict(test_data)

submission = pd.DataFrame({'UniqueID':unique_id,'loan_default':y_final})

submission.to_csv('loan_submission.csv',index=False)