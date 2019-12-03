
# Underwriter Decision Report

from flask import Flask

from crossdomain import *

app = Flask(__name__)

@app.route('/ModelAnalyzer')
def ModelAnalyzer():
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler,LabelEncoder
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix,classification_report,roc_curve,auc,accuracy_score
    from sklearn.model_selection import cross_val_predict
    import json

    def data_load():

        data = pd.read_csv('Heart.csv', index_col=0)
    #    messagebox.showinfo('Data Load', 'Data Load Completed')
        return data


    def Clean_data():

        data = data_load()
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

        data = data.dropna()

        data_cleaned = data
    #    messagebox.showinfo('Data Cleaning', 'Data Cleaning Completed')

        return data_cleaned



    def Feature_engineer():

        data = Clean_data()
        scaler = StandardScaler()

        X = data.drop('AHD',axis=1)

        X = pd.get_dummies(X)

        X_toscale = X.iloc[:,0:6]

        X_encoded = X.iloc[:,6:].values

        X_scaled = scaler.fit_transform(X_toscale)

        X_final = np.concatenate((X_scaled,X_encoded),axis=1)

        y = data['AHD']

    #    messagebox.showinfo('Data Engineering', 'Data Processed and Engineered')

        return X_final,y



    def splitter():

        X_final,y = Feature_engineer()

        X_train,X_test,y_train,y_test = train_test_split(X_final,y,test_size =0.20,random_state=21323)

    #    messagebox.showinfo('Data Splitter', 'Data split into train and test')

        return X_train,X_test,y_train,y_test


    def train_analyze():

        X_train,X_test,y_train,y_test  = splitter()

        logreg = SVC(kernel='linear',probability=True)
        logreg.fit(X_train,y_train)

        y_pred_sv = logreg.predict(X_test)
        y_pred_score_sv = logreg.predict_proba(X_test)

        y_pred_sv = pd.Series(y_pred_sv)
        y_test = pd.Series(y_test)

        y_pred_sv = y_pred_sv.map({0:'Preferred',1:'Non-Preferred'})
        y_test = y_test.map({0:'Preferred',1:'Non-Preferred'})

        c = classification_report(y_test,y_pred_sv)

        x = confusion_matrix(y_test,y_pred_sv,labels=['Preferred','Non-Preferred'])

        x = pd.DataFrame(x,columns=['Preferred','Non-Preferred'],index= ['Preferred','Non-Preferred'])

        d = x

    # =============================================================================
    #     text_box.insert(END, str(c))
    #     text_box.insert(END, str(d))
    # =============================================================================

        accuracy = accuracy_score(y_test,y_pred_sv) * 100

    #    messagebox.showinfo('Prediction', 'Completed with an accuracy of %0.2f' % accuracy + '%')

        return c

    report = train_analyze()

    return json.dumps(report)


def train(Age,Sex,ChestPain,RestBP,Chol,
         Fbs,RestECG,MaxHR,ExAng,Oldpeak,
         Slope,Ca,Thal):

    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler,LabelEncoder
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix,classification_report,roc_curve,auc,accuracy_score
    from sklearn.model_selection import cross_val_predict
    import json

    data = pd.read_csv('Heart.csv',index_col=0)


    data = data.reset_index(drop=True)

    # Chest Pain Listbox dropdown
    onehot_1 = OneHotEncoder()
    onehot_1.fit(np.array(data['ChestPain']).reshape(-1,1))
    cp_array = onehot_1.transform(np.array(data['ChestPain']).reshape(-1,1)).toarray()
    cp_array = pd.DataFrame(cp_array)
    data = pd.concat((data,cp_array),axis=1)
    data = data.drop('ChestPain',axis=1)

    # Thal Radiobutton
    data['Thal']  = data['Thal'].fillna(method = 'ffill')
    onehot_2 = OneHotEncoder()
    onehot_2.fit(np.array(data['Thal']).reshape(-1,1))
    th_array = onehot_2.transform(np.array(data['Thal']).reshape(-1,1)).toarray()
    th_array = pd.DataFrame(th_array)
    data = pd.concat((data,th_array),axis=1)
    data = data.drop('Thal',axis=1)

    #RestECG Radiobutton
    onehot_3 = OneHotEncoder()
    onehot_3.fit(np.array(data['RestECG']).reshape(-1,1))
    ex_array = onehot_3.transform(np.array(data['RestECG']).reshape(-1,1)).toarray()
    ex_array = pd.DataFrame(ex_array)
    data = pd.concat((data,ex_array),axis=1)
    data = data.drop('RestECG',axis=1)

    data['AHD']  = data['AHD'].map({'Yes':1,'No':0})

    data['Ca'] = data['Ca'].fillna(method='ffill')


    scaler = StandardScaler()

    y = data['AHD']
    X = data.drop('AHD',axis=1)

    scaler.fit(X)
    X = scaler.transform(X)

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.20,random_state=21323)

    logreg = SVC(kernel='linear',probability=True)

    logreg.fit(X_train,y_train)

    y_pred_sv = logreg.predict(X_test)

    y_pred_score_sv = logreg.predict_proba(X_test)

    X_new = {'Age':[Age],'Sex':[Sex],'ChestPain':[ChestPain],
                'RestBP':[RestBP],'Chol':[Chol],'Fbs':[Fbs],
                'RestECG':[RestECG],'MaxHR':[MaxHR],'ExAng':[ExAng],
                'Oldpeak':[Oldpeak],'Slope':[Slope],'Ca':[Ca],'Thal':[Thal]}

    data_future = pd.read_csv('Growing_heart_data.csv')

    X_new = pd.DataFrame(X_new)

    X_imp = X_new.copy()

    data_future  = data_future.append(X_imp,ignore_index=True)

    cp_array = onehot_1.transform(np.array(X_new['ChestPain']).reshape(-1,1)).toarray()
    cp_array = pd.DataFrame(cp_array)
    X_new = pd.concat((X_new,cp_array),axis=1)
    X_new = X_new.drop('ChestPain',axis=1)
    th_array = onehot_2.transform(np.array(X_new['Thal']).reshape(-1,1)).toarray()
    th_array = pd.DataFrame(th_array)
    X_new = pd.concat((X_new,th_array),axis=1)
    X_new = X_new.drop('Thal',axis=1)
    ex_array = onehot_3.transform(np.array(X_new['RestECG']).reshape(-1,1)).toarray()
    ex_array = pd.DataFrame(ex_array)
    X_new = pd.concat((X_new,ex_array),axis=1)
    X_new = X_new.drop('RestECG',axis=1)

    Solo = logreg.predict(X_new)
    Solo_prob = logreg.predict_proba(X_new)
    predicted_prob = [x[1] if Solo == 1 else x[0] for x in Solo_prob]

    data_future.to_csv('Growing_heart_data.csv',index=False)

    return Solo,predicted_prob


@app.route('/Customer_1')
@crossdomain(origin='*')
def Customer_1():

    import json

    Age = 25

    Sex = 1

    ChestPain = 'typical'

    RestBP =120

    Chol = 90

    Fbs = 1

    RestECG = 1

    MaxHR =	90

    ExAng = 1

    Oldpeak = 3.4

    Slope = 2

    Ca = 2

    Thal = 'fixed'

    solo, pred_confidence = train(Age,Sex,ChestPain,RestBP,Chol,
                                        Fbs,RestECG,MaxHR,ExAng,Oldpeak,
                                        Slope,Ca,Thal)
    if solo[0] == 1:
        value = 'Non-Preferred'
    else:
        value = 'Preferred'

    x = pred_confidence[0]

    confidence = x*100

    confidence = round(confidence,2)

    result = 'This customer is '+ str(value) + ' and '+ 'with a confidence of '+ str(confidence) + '%'

    return json.dumps({'Prediction': value, 'Confidence': confidence})

@app.route('/Customer_2')
@crossdomain(origin='*')
def Customer_2():

    import json

    Age = 45

    Sex = 0

    ChestPain = 'nonanginal'

    RestBP =140

    Chol = 112

    Fbs = 0

    RestECG = 2

    MaxHR =	112

    ExAng = 1

    Oldpeak = 3.4

    Slope = 3

    Ca = 2

    Thal = 'fixed'

    solo, pred_confidence = train(Age,Sex,ChestPain,RestBP,Chol,
                                        Fbs,RestECG,MaxHR,ExAng,Oldpeak,
                                        Slope,Ca,Thal)
    if solo[0] == 1:
        value = 'Non-Preferred'
    else:
        value = 'Preferred'

    x = pred_confidence[0]

    confidence = x*100

    confidence = round(confidence,2)

    result = 'This customer is '+ str(value) + ' and '+ 'with a confidence of '+ str(confidence) + '%'

    return json.dumps({'Prediction': value, 'Confidence': confidence})


@app.route('/Customer_3')
@crossdomain(origin='*')
def Customer_3():

    import json

    Age = 25

    Sex = 1

    ChestPain = 'asymptomatic'

    RestBP =120

    Chol = 110

    Fbs = 0

    RestECG = 0

    MaxHR =	110

    ExAng = 0

    Oldpeak = 2

    Slope = 2

    Ca = 0

    Thal = 'normal'

    solo, pred_confidence = train(Age,Sex,ChestPain,RestBP,Chol,
                                        Fbs,RestECG,MaxHR,ExAng,Oldpeak,
                                        Slope,Ca,Thal)
    if solo[0] == 1:
        value = 'Non-Preferred'
    else:
        value = 'Preferred'

    x = pred_confidence[0]

    confidence = x*100

    confidence = round(confidence,2)

    result = 'This customer is '+ str(value) + ' and '+ 'with a confidence of '+ str(confidence) + '%'

    return json.dumps({'Prediction': value, 'Confidence': confidence})

@app.route('/Customer_4')
@crossdomain(origin='*')
def Customer_4():

    import json

    Age = 37

    Sex = 0

    ChestPain = 'asymptomatic'

    RestBP =120

    Chol = 100

    Fbs = 0

    RestECG = 0

    MaxHR =	100

    ExAng = 0

    Oldpeak = 2

    Slope = 2

    Ca = 0

    Thal = 'normal'

    solo, pred_confidence = train(Age,Sex,ChestPain,RestBP,Chol,
                                        Fbs,RestECG,MaxHR,ExAng,Oldpeak,
                                        Slope,Ca,Thal)
    if solo[0] == 1:
        value = 'Non-Preferred'
    else:
        value = 'Preferred'

    x = pred_confidence[0]

    confidence = x*100

    confidence = round(confidence,2)

    result = 'This customer is '+ str(value) + ' and '+ 'with a confidence of '+ str(confidence) + '%'

    return json.dumps({'Prediction': value, 'Confidence': confidence})
