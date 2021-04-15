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

train_test = application_train_test(num_rows = None, nan_as_category = False)

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

data = train_test.drop(columns=['index', 'SK_ID_CURR', 'TARGET']).fillna(X.mean())

clf4 = LGBMClassifier(max_depth=24, n_estimators=836, num_leaves=23,
                     learning_rate=0.02,
                     min_child_weight= 95.7889530150502,
                     min_split_gain= 0.5331652849730171,
                     reg_alpha= 0.6918771139504734,
                     reg_lambda= 0.31551563100606295,
                     colsample_bytree= 0.20445224973151743,
                     subsample= 0.8781174363909454, 
                     is_unbalance=True, random_state=1, force_row_wise=True)

model = clf4.fit(X_train, y)
y_pred1 = clf4.predict_proba(X_train)
y_pred2 = model.predict_proba(X_test)
y_pred = np.concatenate((y_pred1, y_pred2))

pickle.dump(model, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))
print(y_pred)