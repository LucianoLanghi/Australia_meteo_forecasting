# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 14:36:52 2024

@author: Sophie
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_backend_sb import ProjetAustralieSoutenance


def affiche_page_introduction(titre:str):
    st.title(titre)
    
    # ajouter lien Kaggle
    st.write("lien kaggle")
    
    # evolution de RainToday => peu de cohérence géo et temporelle
    fig = pas.animation_variable("RainToday")
    st.plotly_chart(fig)
    
    # l'Australie, c'est grand !
    st.image('img/carte-australie-europe.jpg', caption="l'Australie, c'est grand!", width=768) #, use_column_width=True)
    
def affiche_page_dataset(titre:str):
    st.title(titre)
    
    st.write(df.shape)
    st.dataframe(df.describe())

    # géographie - raintomorrow / MaxTemp
    fig = pas.synthetise_villes()
    st.plotly_chart(fig)


    # Histogramme temp / pluviometrie
    option = st.selectbox(
        'Sélectionnez une Location',
        (df.Location.unique())
    )
    
    pas.histogramme_temperatures_precipitations(option)
    st.pyplot(plt.gcf())  
    
    pas.graphe_vent("")
    st.pyplot(plt.gcf())  
    
    pas.matrice_corr_quyen(pas.df, "Corrélations entre les variables du dataset initial")
    st.pyplot(plt.gcf())  

    
def affiche_page_na(titre:str):
    st.title(titre)
    
    pas.graphe_taux_na_location_feature()
    st.pyplot(plt.gcf())
            

def affichage_page_feature_engineering(titre:str):
    st.title(titre)

    affichage_clusterisation()

    pas.matrice_corr_quyen(pas.data)
    st.pyplot(plt.gcf())  
    
    
def affichage_clusterisation():
    # Clusterisation
    fig = pas.clusterisation_groupee()
    st.pyplot(plt.gcf())
    st.caption("Dendrogramme de clusterisation")

    st.plotly_chart(fig)
    st.caption("Résulat de la clusterisation en 7 zones climatiques")

    st.image('img/climats_wiki.png', caption="Climats australiens (Wikipédia)", width=512)


# affichage large
st.set_page_config(layout="wide")

# Titre de l'application
st.title('Prévisions Météorologiques en Australie')

# Titre menu
st.sidebar.title('Prévisions Météorologiques en Australie')
points_menu = ['Introduction', 'le Jeu de Données', 'Valeurs manquantes', 'Feature Engineering', 'Modélisation ML', 'Interprétabilité', 'Horizon de temps', 'MaxTemp', 'Conclusion']

# chargement des données et instanciation des classes de backend
df=pd.read_csv("datasets/data_process3_knnim_resample_J2.csv", index_col=0)
pas = ProjetAustralieSoutenance(df)

# Options de menu
option = st.sidebar.radio(
    'Choisissez une option',
    points_menu
)

# Redirections des points de menu
if option == points_menu[0]:
    affiche_page_introduction(option)
elif option == points_menu[1]:
    affiche_page_dataset(option)
elif option == points_menu[2]:
    affiche_page_na(option)
elif option == points_menu[3]:
    affichage_clusterisation()