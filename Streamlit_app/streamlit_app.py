import streamlit as st

from pages import page_introduction, page_conclusion, page_jeu_de_donnees, page_predict_raintomorrow, \
    page_predict_rain_long_horizon, page_predict_maxtemps, page_preprocessing_feature_engineering

st.title("Prévision météo en Australie")

st.sidebar.title("Sommaire")

pages = ["Introduction",
         "Le jeu de données",
         "Pre-processing",
         "Prédiction de RainTomorrow",
         "Prédiction de la pluie",
         "Prédiction de MaxTemps",
         "Conclusion"]
page = st.sidebar.radio("Aller vers ", pages)

# Introduction à la page 1
if page == pages[0]:
    page_introduction.app()

if page == pages[1]:
    page_jeu_de_donnees.app()

if page == pages[2]:
    page_preprocessing_feature_engineering.app()

if page == pages[3]:
    page_predict_raintomorrow.app()

if page == pages[4]:
    page_predict_rain_long_horizon.app()

if page == pages[5]:
    page_predict_maxtemps.app()

if page == pages[6]:
    page_conclusion.app()

