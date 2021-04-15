import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import matplotlib
# Kernel de Kaggle disponible sur https://www.kaggle.com/jsaguiar/lightgbm-with-simple-features
from lightgbm_with_simple_features import *
import lightgbm
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix
from PIL import Image
from sklearn.model_selection import train_test_split
import shap
import streamlit.components.v1 as components

import pickle
import plotly.graph_objects as go


import lime
import lime.lime_tabular

st.set_option('deprecation.showPyplotGlobalUse', False)

# import altair as alt
import re
#matplotlib.use('Agg')

# import io
# import requests
# url2 = 'https://www.kaggle.com/c/home-credit-default-risk/data?select=application_test.csv'
# url1 = 'https://www.kaggle.com/c/home-credit-default-risk/data?select=application_train.csv'
# s1=requests.get(url1)
# s2=requests.get(url2)
# df=pd.read_csv(io.StringIO(s1.text))
# test_df=pd.read_csv(io.StringIO(s2.text))
# # df=pd.read_csv(c2)
# # test_df=pd.read_csv(c1)

st.set_page_config(layout="wide")


# model = pickle.load(open('model.pkl', 'rb'))

st.title('Prêt à dépenser')

image = Image.open('pret-a-depenser.PNG')

@st.cache(suppress_st_warning=True)
def train():
    train_test = application_train_test(num_rows = None, nan_as_category = False)
    train_test = train_test.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    train = train_test[train_test['TARGET'].notna()].drop(columns=['index', 'SK_ID_CURR'])
    X = train.drop(columns=['TARGET'])
    y = train['TARGET']
    X_fill = X.fillna(X.mean())
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_fill), columns= X.columns)
    return X_train, y

X_train = train()[0]
y = train()[1]

# train_test = application_train_test(num_rows = None, nan_as_category = False)
# #bureau_balance = bureau_and_balance(num_rows = None, nan_as_category = True)
# #prev_appli = previous_applications(num_rows = None, nan_as_category = True)
# #poscash = pos_cash(num_rows = None, nan_as_category = True)
# #install_pay = installments_payments(num_rows = None, nan_as_category = True)
# #cre_card_bal = credit_card_balance(num_rows = None, nan_as_category = True)
# # home_cred = pd.read_csv('HomeCredit_columns_description.csv', encoding ='cp1258')

# train_test = train_test.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

# train = train_test[train_test['TARGET'].notna()].drop(columns=['index', 'SK_ID_CURR'])
# test = train_test[train_test['TARGET'].isna()].drop(columns=['index', 'SK_ID_CURR', 'TARGET'])


# X = train.drop(columns=['TARGET']) 
# y = train['TARGET']

# X_fill = X.fillna(X.mean())
# X_t = test.fillna(X.mean())

# scaler = StandardScaler()
# X_train = pd.DataFrame(scaler.fit_transform(X_fill), columns= X.columns)
# X_test = pd.DataFrame(scaler.transform(X_t), columns= test.columns)
# X2 = pd.concat([X_train, X_test])

# data = train_test.drop(columns=['index', 'SK_ID_CURR', 'TARGET']).fillna(X.mean())

# y_pred = model.predict_proba(data)
    

# ######### ------------------------ ###########
@st.cache(suppress_st_warning=True)
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# ######### ------------------------ ###########
@st.cache(suppress_st_warning=True)
def predict():
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
    y_pred = clf4.predict_proba(X_train)
#     y_pred2 = model.predict_proba(X_test)
#     y_pred = np.concatenate((y_pred1, y_pred2))
    return y_pred, model

y_pred = predict()[0]
model = predict()[1]

# ######### ------------------------ ###########
@st.cache(suppress_st_warning=True)
def seuil(k, y_predprob):
    classification = []
    for i in range(len(y_predprob)):
        if y_predprob[:,1][i]>k:
            classification.append(1)
        else:
            classification.append(0)
    return classification

k = 0.3
score = seuil(k, y_pred)

# ######### ------------------------ ###########
@st.cache(suppress_st_warning=True)
def prob(data):
    return np.array(list(zip(1-model.predict(data), model.predict(data))))

# ######### ------------------------ ###########


feat_importances = pd.Series(model.feature_importances_, index=X.columns)

# ########## ------------------- ############ 

# @st.cache(suppress_st_warning=True)
# def shap_explainer():
#     explainer = shap.TreeExplainer(model)
#     shap_values = explainer.shap_values(X2)
#     #fig_summary_shap = shap.summary_plot(shap_values, X2)
#     return explainer, shap_values #, fig_summary_shap



# @st.cache(suppress_st_warning=True)
# def lime_explainer():
#     explainer = lime.lime_tabular.LimeTabularExplainer(X_train.astype(int).values,  
#     mode='classification',training_labels=y,feature_names=X_train.columns)
#     return explainer


def main():
    
    

    st.sidebar.image(image, width=300)



    client=st.sidebar.number_input("Saisissez un numéro de clients", min_value=0, max_value=356251, value=0)




#     col1, col2 = st.beta_columns(2)


#     col1.text('Score :')
#     col1.write(y_pred[client][0])
#     if score[client] == 0:
#         col1.success('Prêt accepté !')
#     else : 
#         col1.warning('Prêt refusé !')


#     # fig = go.Figure(go.Indicator(
#     # mode = "number+gauge+delta", value = y_pred[client][0],
#     # domain = {'x': [0.1, 1], 'y': [0, 1]},
#     # title = {'text' :"Score du client "},
#     # delta = {'reference': 1},
#     # gauge = {
#     #     'shape': "bullet",
#     #     'axis': {'range': [0, 1]},
#     #     'threshold': {
#     #         'line': {'color': "red", 'width': 2},
#     #         'thickness': 0.75,
#     #         'value': 0.3},
#     #     'steps': [
#     #         {'range': [0, 0.2], 'color': "red"},
#     #         {'range': [0.2, 0.4], 'color': "orange"},
#     #         {'range': [0.4, 0.7], 'color': "lightgreen"},
#     #         {'range': [0.7, 1], 'color': "green"}
#     #         ],
#     #     'bar' : {'color': 'royalblue'}
#     #         }))
#     # fig.update_layout(height = 250)
#     # col2.plotly_chart(fig)

#     col2.write('Score du client')
#     col2.progress(y_pred[client][0])

#     col2.write('Seuil attendu')

#     col2.progress(0.4)


#     fig = go.Figure(go.Indicator(
#     mode = "gauge+number+delta",
#     value = y_pred[client][0],
#     domain = {'x': [0, 1], 'y': [0, 1]},
#     title = {'text': "Score du client", 'font': {'size': 24}},
#     delta = {'reference': 1, 'increasing': {'color': "RebeccaPurple"}},
#     gauge = {
#         'axis': {'range': [None, 1], 'tickwidth': 1, 'tickcolor': "royalblue"},
#         'bar': {'color': "darkblue"},
#         'bgcolor': "white",
#         'borderwidth': 2,
#         'bordercolor': "gray",
#         'steps': [
#             {'range': [0, 0.2], 'color': 'red'},
#             {'range': [0.2, 0.4], 'color': 'orange'},
#             {'range': [0.4, 0.7], 'color': 'lightgreen'},
#             {'range': [0.7, 1], 'color': 'green'}],
#         }))

#     fig.update_layout(paper_bgcolor = "lavender", font = {'color': "darkblue", 'family': "Arial"})
#     col1.plotly_chart(fig)



#     labels = ['Crédit Refusé', 'Crédit accepté']
#     S = sum(score)
#     values = [len(y_pred) -S , S ]

#     fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent',
#                              insidetextorientation='radial')])

#     plt.title('Répartition crédit accepté/ crédit refusé', fontsize=25)

#     col2.plotly_chart(fig, use_container_width=True) 


#     col1, col2 = st.beta_columns(2)


#     fig = plt.figure(figsize=(20, 15))
#     feat_importances.nlargest(10).plot(kind='barh')
#     plt.title('Importance global des Features', fontsize=25)
#     col2.pyplot(fig)



    
    

#         # # asking for explanation for LIME model
#     fig = plt.figure(figsize=(20, 20))
#     exp = lime_explainer().explain_instance(data.iloc[client].astype(int).values, prob, num_features=10)
#     plt.title('Importance locale des features')
#     exp.as_pyplot_figure()
#     col1.pyplot()

 



#     # explain the model's predictions using SHAP
#     # (same syntax works for LightGBM, CatBoost, scikit-learn and spark models)
#     st.write("Explication des points forts et points faibles du client", fontsize=25)
#     st_shap(shap.force_plot(shap_explainer()[0].expected_value[1], shap_explainer()[1][1][client,:], data.iloc[client,:]))

#     # st.write("Représentation générale", fontsize=25)
    
#     # st.pyplot(shap_explainer()[2])
#     # for name in X_train.columns:
#     # shap.dependence_plot(name, shap_values[1], X, display_features=X_display)

#     col1, col2 = st.beta_columns(2)
#     feat_imp_sort = feat_importances.sort_values(ascending=False)
#     feat_imp_sort2 = feat_imp_sort.copy()
#     var1 = col1.selectbox('Sélectionner la première variable', feat_imp_sort.index)
#     var2 = col2.selectbox('Sélectionner la deuxième variable', feat_imp_sort.index.drop(var1))

    
#     # fig = px.scatter(data, x=var1, y=var2,
#     #              color=y_pred[:,1], color_continuous_scale='Inferno')
#     # df = data.iloc[client]
#     # fig.add_trace(px.scatter(df, x=var1, y=var2,
#     #              color='red'))
     
#     # st.plotly_chart(fig)




#     fig = plt.figure(figsize=(15,10))
#     plt.scatter(data[var1], data[var2], c=y_pred[:,1])
#     plt.scatter(data[var1].iloc[client], data[var2].iloc[client], marker="8", s=200, c='red')
#     st.pyplot(fig)

if __name__ =='__main__':
    main()
