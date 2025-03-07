import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px

# Titre de l'application
st.title("Prédiction de risque de crédit")

# Sous-titre et description
st.subheader("Application réalisée par jean-samuel HAUDIE")
st.markdown("*Cette application utilise un modèle de Machine Learning pour prédire si une personne est éligible pour un prêt ou pas un client solvable*")

# Chargement du modèle de Machine Learning pré-entraîné
model = joblib.load("decision_tree_model.joblib")

# Fonction d'inférence pour effectuer la prédiction
def inference(person_age, person_income, person_home_ownership, person_emp_length, loan_intent, loan_grade, loan_amnt, loan_int_rate, loan_percent_income, cb_person_default_on_file, cb_person_cred_hist_length):
    """
    Cette fonction prend les caractéristiques / informations de la personne en entrée 
    et utilise le modèle de Machine Learning pour prédire si la personne est éligible au crédit.
    """
    
    # Encodage des variables catégorielles
    home_ownership_dict = {'RENT': 0, 'OWN': 1, 'MORTGAGE': 2}
    intent_dict = {'PERSONAL': 0, 'EDUCATION': 1, 'MEDICAL': 2, 'VENTURE': 3, 'HOMEIMPROVEMENT': 4, 'DEBTCONSOLIDATION': 5}
    grade_dict = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
    default_dict = {'Y': 1, 'N': 0}

    # Transformation des variables catégorielles
    person_home_ownership = home_ownership_dict.get(person_home_ownership, -1)
    loan_intent = intent_dict.get(loan_intent, -1)
    loan_grade = grade_dict.get(loan_grade, -1)
    cb_person_default_on_file = default_dict.get(cb_person_default_on_file, -1)

    # Création du tableau de données avec les caractéristiques fournies par l'utilisateur
    new_data = np.array([person_age, person_income, person_home_ownership, person_emp_length,
                         loan_intent, loan_grade, loan_amnt, loan_int_rate, loan_percent_income,
                         cb_person_default_on_file, cb_person_cred_hist_length])

    # Réorganisation du tableau pour le modèle
    pred = model.predict(new_data.reshape(1, -1))

    return pred

# Interface utilisateur pour la saisie des caractéristiques
person_age = st.number_input("Age de la personne:", min_value=18, max_value=100, value=30)
person_income = st.number_input('Revenu annuel de la personne:', min_value=1000, max_value=1000000, value=50000)
person_home_ownership = st.selectbox('Propriété du logement (RENT, OWN, MORTGAGE):', ['RENT', 'OWN', 'MORTGAGE'])
person_emp_length = st.number_input('Durée de l\'emploi (en années):', min_value=0, max_value=50, value=5)
loan_intent = st.selectbox('Intention du prêt (PERSONAL, EDUCATION, MEDICAL, VENTURE, HOMEIMPROVEMENT, DEBTCONSOLIDATION):', 
                           ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'])
loan_grade = st.selectbox('Qualité du prêt (A, B, C, D, E, F, G):', ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
loan_amnt = st.number_input('Montant du prêt:', min_value=1000, max_value=500000, value=10000)
loan_int_rate = st.number_input('Taux d\'intérêt du prêt (%):', min_value=1.0, max_value=50.0, value=5.0)
loan_percent_income = st.number_input('Pourcentage du revenu à allouer au prêt:', min_value=1, max_value=100, value=20)
cb_person_default_on_file = st.selectbox('Défaut historique de crédit (Y/N):', ['Y', 'N'])
cb_person_cred_hist_length = st.number_input('Longueur de l\'historique de crédit (en années):', min_value=0, max_value=50, value=5)

# Bouton pour prédire le risque de crédit
if st.button('Faire une prédiction'):
    prediction = inference(person_age, person_income, person_home_ownership, person_emp_length, 
                           loan_intent, loan_grade, loan_amnt, loan_int_rate, loan_percent_income, 
                           cb_person_default_on_file, cb_person_cred_hist_length)

    # Affichage du résultat
    if prediction == 1:
        st.subheader("Risque élevé de crédit")
    else:
        st.subheader("Risque faible de crédit")

# --------------- Visualisations interactives ------------------

# Affichage des caractéristiques sous forme de graphique Radar
if st.checkbox("Voir les caractéristiques du client sur un radar chart"):
    fig = go.Figure(data=go.Scatterpolar(
        r=[person_age, person_income, person_home_ownership, person_emp_length, loan_intent,
           loan_grade, loan_amnt, loan_int_rate, loan_percent_income, cb_person_default_on_file, 
           cb_person_cred_hist_length],
        theta=['Age', 'Revenu', 'Propriété du logement', 'Durée de l\'emploi', 'Intention du prêt', 
               'Qualité du prêt', 'Montant du prêt', 'Taux d\'intérêt', 'Pourcentage revenu', 
               'Défaut historique', 'Historique crédit'],
        fill='toself', 
        name="Caractéristiques du client"
    ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False)
    st.plotly_chart(fig)

# Affichage des distributions des caractéristiques d'entraînement sous forme de graphique interactif
if st.checkbox("Voir les distributions des caractéristiques d'entraînement"):
    # Simuler un jeu de données d'entraînement pour les visualisations
    data = {
        "Age": np.random.randint(18, 70, 500),
        "Revenu": np.random.randint(1000, 100000, 500),
        "Propriété du logement": np.random.choice(['RENT', 'OWN', 'MORTGAGE'], 500),
        "Durée de l'emploi": np.random.randint(0, 50, 500),
        "Intention du prêt": np.random.choice(['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'], 500),
        "Qualité du prêt": np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G'], 500),
        "Montant du prêt": np.random.randint(1000, 500000, 500)
    }
    df = pd.DataFrame(data)
    fig = px.histogram(df, x="Revenu", nbins=30, title="Distribution du revenu des clients")
    st.plotly_chart(fig)
