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
import matplotlib.patches as mpatches

st.set_option('deprecation.showPyplotGlobalUse', False)

# import altair as alt
import re
#matplotlib.use('Agg')



st.set_page_config(layout="wide")


model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# st.title('Prêt à dépenser')

image = Image.open('pret-a-depenser.PNG')

@st.cache(suppress_st_warning=True)
def train():
    train_test = application_train_test(num_rows = None, nan_as_category = False)
    train_test = train_test.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    train = train_test[train_test['TARGET'].notna()].drop(columns=['index', 'SK_ID_CURR'])
    X = train.drop(columns=['TARGET'])
    y = train['TARGET']
    X_fill = X.fillna(X.mean())
    X_train = pd.DataFrame(scaler.transform(X_fill), columns= X.columns).sample(frac=0.05, random_state=1)
    return X_train, y

X_train = train()[0]
y = train()[1]


    

# ######### ------------------------ ###########
@st.cache(suppress_st_warning=True)
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# ######### ------------------------ ###########
@st.cache(suppress_st_warning=True)
def predict():
    y_pred = model.predict_proba(X_train)
    return y_pred #, model

y_pred = predict()




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

k = 0.5
score = seuil(k, y_pred)

############ -------------- ########

def fonc():
    y = []
    for i in range(len(y_pred)):
            y.append(2* y_pred[i][0]-1)
    return y

y_scaled = fonc()


# ######### ------------------------ ###########
@st.cache(suppress_st_warning=True)
def prob(data):
    return np.array(list(zip(1-model.predict(data), model.predict(data))))

# ######### ------------------------ ###########


feat_importances = pd.Series(model.feature_importances_, index=X_train.columns)

# ########## ------------------- ############ 

# @st.cache(suppress_st_warning=True)
# def shap_explainer():
#     explainer = shap.TreeExplainer(model)
#     shap_values = explainer.shap_values(X_train)
# #     #fig_summary_shap = shap.summary_plot(shap_values, X_train)
#     return explainer, shap_values #, fig_summary_shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)




# @st.cache(suppress_st_warning=True)
@st.cache(allow_output_mutation=True)
def lime_explainer():
    explainer = lime.lime_tabular.LimeTabularExplainer(X_train.astype(int).values,  
    mode='classification',training_labels=y,feature_names=X_train.columns)
    return explainer


def main():
    st.markdown("<h1 style='text-align: center; color: red;'>Prêt à dépenser</h1>", unsafe_allow_html=True)
    

    st.sidebar.image(image, width=300)



    client=st.sidebar.number_input("Saisissez un numéro de clients", min_value=0, max_value=X_train.shape[0], value=0)



    col1, col2 = st.beta_columns(2)


    col1.text('Score :')
    col1.write(y_pred[client][0])
    if score[client] == 0:
        col1.success('Prêt accepté !')
    else : 
        col1.warning('Prêt refusé !')



    col2.write('Score du client')
    col2.progress(y_pred[client][0])

    col2.write('Seuil attendu')

    col2.progress(0.5)



    fig = go.Figure(go.Indicator(
    mode = "gauge+delta",
    value = y_scaled[client], #y_pred[client][0],
    domain = {'x': [0, 1], 'y': [0, 0.75]},
    title = {'text': "Un autre indicateur de score du client <br><span style='font-size:0.8em;color:red'>Insatisfaisant</span> \
    <span style='font-size:0.8em;color:orange'>Passable</span> \
    <span style='font-size:0.8em;color:lightgreen'>Correct</span>\
    <span style='font-size:0.8em;color:green'>Très bon</span><br> "
    , 'font': {'size': 24, 'color':'black'}},
    delta = {'reference': 0, 'increasing': {'color': "RebeccaPurple"}},
    gauge = {
        'axis': {'range': [-1, 1], 'tickwidth': 0.1, 'tickcolor': "royalblue"},
        'bar': {'color': "darkblue"},
        'bgcolor': "white",
        'borderwidth': 2,
        'bordercolor': "gray",
        'steps': [
            {'range': [-1, -0.5], 'color': 'red', 'name':'Mauvais'},
            {'range': [-0.5, 0], 'color': 'orange', 'name':'Passable'},
            {'range': [0, 0.5], 'color': 'lightgreen', 'name':'Correct'},
            {'range': [0.5, 1], 'color': 'green', 'name':'Bon'}],
        }))
    fig.update_layout(paper_bgcolor = "white", font = {'color': "darkblue", 'family': "Arial"})



    col1.plotly_chart(fig, use_container_width=True)


    labels = ['Crédit Refusé', 'Crédit accepté']
    S = sum(score)
    values = [len(y_pred) -S , S ]

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent',
                             insidetextorientation='radial')])

    col2.write('** Répartition crédits acceptés / crédits refusés sur notre échantillon** ', fontsize=40)

    col2.plotly_chart(fig, use_container_width=True) 
# ---------------------------------------


    col1, col2 = st.beta_columns(2)


    fig = plt.figure(figsize=(15, 10))
    feat_importances.nlargest(10).plot(kind='barh')
    col2.write('**Importance globale des caractéristiques**', fontsize=40)
    col2.pyplot(fig)


        # asking for explanation for LIME model
    col1.write('**Descriptions des principales caractéristiques du client pour la classe 1 (crédit refusé)**', fontsize=40) 
    fig = plt.figure(figsize=(20, 20))
    exp = lime_explainer().explain_instance(X_train.iloc[client].astype(int).values, prob, num_features=10)
    plt.title('Importance locale des features')
#     html = exp.as_html()
    exp.as_pyplot_figure()
    col1.pyplot()
    
    

        # asking for explanation for LIME model
    # fig = plt.figure(figsize=(20, 20))
    # exp = lime_explainer().explain_instance(X_train.iloc[client].astype(int).values, prob, num_features=10)
    # plt.title('Importance locale des features')
#     html = exp.as_html()
    # components.html(exp.as_html(), height=500)
    # exp.as_pyplot_figure(title='Importance locale des features')
    # col1.pyplot()
 



    # explain the model's predictions using SHAP
    # (same syntax works for LightGBM, CatBoost, scikit-learn and spark models)
    st.write("**Explication des points forts et points faibles du client**", fontsize=40)
    st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1][client,:], X_train.iloc[client,:]), height=200)

    # st.write("**Comparatif sur un échantillon de 100 clients**", fontsize=40)
    # st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1][:100,:], X_train.iloc[:100,:]), height=500)


#-----------------------------------------------

    st.write("**Position du client par rapport à l'échantillon**", fontsize=40)

    col1, col2 = st.beta_columns(2)
    feat_imp_sort = feat_importances.sort_values(ascending=False)
    feat_imp_sort2 = feat_imp_sort.copy()
    var1 = col1.selectbox('Sélectionner la première variable', feat_imp_sort.index)
    var2 = col2.selectbox('Sélectionner la deuxième variable', feat_imp_sort.index.drop(var1))


    fig = plt.figure(figsize=(15,10))
    plt.scatter(X_train[var1], X_train[var2], c=y_pred[:,0])
    plt.colorbar()
    plt.title("Couleur par score", fontsize=20 )
    # plt.annotate("Client observé", [X_train[var1].iloc[client], X_train[var2].iloc[client]], marker="8", s=200, c='red' )
    plt.scatter(X_train[var1].iloc[client], X_train[var2].iloc[client], marker="8", s=200, c='red', label='Client')
    plt.xlabel(var1, fontweight ='bold',
               fontsize=16)
    plt.ylabel(var2, fontweight ='bold',
               fontsize=16)
    col1.pyplot(fig)

    colormap = np.array(['#0b559f', '#89bedc'])

    fig = plt.figure(figsize=(15,11))
    plt.title("Couleur par classification", fontsize=20)
    plt.scatter(X_train[var1], X_train[var2], c=colormap[score])
    pop_a = mpatches.Patch(color='#0b559f', label='crédit refusé')
    pop_b = mpatches.Patch(color='#89bedc', label='crédit accepté')
    plt.legend(handles=[pop_a,pop_b])
    plt.scatter(X_train[var1].iloc[client], X_train[var2].iloc[client], marker="8", s=200, c='red', label='Client')
    plt.xlabel(var1, fontweight ='bold',
               fontsize=16)
    plt.tick_params(axis='y', which='both', labelleft=False, labelright=True)
    plt.ylabel(var2, fontweight ='bold',
               fontsize=16)
    col2.pyplot(fig)







if __name__ =='__main__':
    main()
