import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score


st.title("Application de Machine Learning Financial Inclusion in AFRICA")
st.subheader("Auteur: BIRAMA")

    # Définition de la fonction pour télécharger les données

data = pd.read_csv("Financial_inclusion_dataset.csv")


# Afficher des informations générales sur l'ensemble de données
data.info()


data.isnull().sum()

# Encoder les variables catégorielles en variables bnaires (encodage one-hot)
encoder=LabelEncoder()
data["bank_account"]=encoder.fit_transform(data["bank_account"])
data["country"]=encoder.fit_transform(data["country"])
data["uniqueid"]=encoder.fit_transform(data["uniqueid"])
data["location_type"]=encoder.fit_transform(data["location_type"])
data["cellphone_access"]=encoder.fit_transform(data["cellphone_access"])
data["gender_of_respondent"]=encoder.fit_transform(data["gender_of_respondent"])
data["relationship_with_head"]=encoder.fit_transform(data["relationship_with_head"])
data["marital_status"]=encoder.fit_transform(data["marital_status"])
data["education_level"]=encoder.fit_transform(data["education_level"])
data["job_type"]=encoder.fit_transform(data["job_type"])



if st.sidebar.checkbox('Afficher la base de données', False):
        st.subheader("Quelques données du dataset")
        st.write(data.head())
        st.subheader("Description")
        st.write(data.describe())
        st.subheader("valeurs manquantes")
        st.write(data.isnull().sum())


# Séparer les variables prédictives (X) et la variable cible (y)
x = data.drop(['bank_account'], axis=1)
y = data['bank_account']

# Diviser les données en ensembles d'entraînement et de test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

st.sidebar.subheader("Les hyperparamètres du modèle")
n_arbres = st.sidebar.number_input("Nombre d'arbres pour le modèle de forêt", 100, 1000, step=10)
profondeur_arbre = st.sidebar.number_input("La profondeur max du modèle de forêt", 1, 20, step=1)
bootstrap = st.sidebar.radio("Échantillons bootstrap lors de la création d'arbres", (True, False))


#Prédiction de la forêt aléatoire
if st.sidebar.button("Exécuter", key="classify"):
        st.subheader("Random Forest Résultat")
        model = RandomForestClassifier(n_estimators=n_arbres, max_depth=profondeur_arbre, bootstrap=bootstrap)
        model.fit(x_train, y_train)
        y_pred=model.predict(x_test)  #testing our model
        Accuracy = metrics.accuracy_score(y_test, y_pred)
        st.write("Accuracy :", Accuracy)
        #st.write('F1-Score: %.2f%%' % (f1_score(y_test, y_pred)))
       # st.write(classification_report(y_test, y_pred))
