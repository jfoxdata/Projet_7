import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

# Kernel de Kaggle disponible sur https://www.kaggle.com/jsaguiar/lightgbm-with-simple-features
from lightgbm_with_simple_features import *
import lightgbm
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

import pickle
import re

from flask import Flask, request, jsonify, render_template

train_test = application_train_test(num_rows = None, nan_as_category = False)
# #bureau_balance = bureau_and_balance(num_rows = None, nan_as_category = True)
# #prev_appli = previous_applications(num_rows = None, nan_as_category = True)
# #poscash = pos_cash(num_rows = None, nan_as_category = True)
# #install_pay = installments_payments(num_rows = None, nan_as_category = True)
# #cre_card_bal = credit_card_balance(num_rows = None, nan_as_category = True)
# home_cred = pd.read_csv('HomeCredit_columns_description.csv', encoding ='cp1258')

train_test = train_test.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

train = train_test[train_test['TARGET'].notna()].drop(columns=['index', 'SK_ID_CURR'])
test = train_test[train_test['TARGET'].isna()].drop(columns=['index', 'SK_ID_CURR', 'TARGET'])


X = train.drop(columns=['TARGET']) 
y = train['TARGET']

X_fill = X.fillna(X.mean())
X_t = test.fillna(X.mean())

scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_fill), columns= X.columns)
X_test = pd.DataFrame(scaler.transform(X_t), columns= test.columns)



# clf4 = LGBMClassifier(max_depth=24, n_estimators=836, num_leaves=23,
#                      learning_rate=0.02,
#                      min_child_weight= 95.7889530150502,
#                      min_split_gain= 0.5331652849730171,
#                      reg_alpha= 0.6918771139504734,
#                      reg_lambda= 0.31551563100606295,
#                      colsample_bytree= 0.20445224973151743,
#                      subsample= 0.8781174363909454, 
#                      is_unbalance=True, random_state=1, force_row_wise=True)

# model = clf4.fit(X_train, y)
# # y_pred1 = clf4.predict_proba(X_train)
# # y_pred2 = model.predict_proba(X_test)
# # y_pred = np.concatenate((y_pred1, y_pred2))

M = pd.concat([X_train, X_test], axis=0)


app=Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
# def predict():

# 	client = NumberInput(min=0, max=len(train_test))
# 	prediction = model.predict_proba(client)
# 	return render_template('index.html', prediction_text='Your Rating is: {}'.format(predict))

def predict():
    features = [int(x) for x in request.form.values()]
    final_features = M.iloc[features]
    prediction = model.predict_proba(final_features)

    output = prediction

    return render_template('index.html', prediction_text='Your Rating is: {}'.format(output))

if __name__=='__main__':
    app.run(debug=True)
#	app.run(host='0.0.0.0', port=8080)