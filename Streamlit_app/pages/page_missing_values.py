# -*- encoding: utf-8 -*-

import streamlit as st
import matplotlib.pyplot as plt

def app(pas):
    
    st.markdown(
        """
        <style>
     .big-font, .big-font ul, .big-font li {
         font-size:20px !important; 
     }
     </style>
    
    <div class='big-font'>
        L'identification des valeurs manquantes est cruciale dans l'analyse de données, car elle impacte l'exactitude des résultats. 
        Dans notre projet, nous constatons un nombre significatif de données absentes pour certaines variables, comme le montre le graphique suivant
    </div>""", unsafe_allow_html=True
    )
    pas.graphe_taux_na_location_feature()
    st.pyplot(plt.gcf())
    
    
    # ajoute valeur manqantes vs temp
    
    
    #Conclution 
    st.markdown(
        """
            
    <div class='big-font'>
        Pour compléter les valeurs manquantes, il a été décidé d'adopter deux mesures : 
        <ul>
                <li>Suppression des lignes avec des données manquantes pour la variable cible (représentant 2.2% de l’ensemble de données).</li>
                <li>Suppression des lignes avec une forte proportion de données manquantes.</li> 
        </ul>        
    </div>""", unsafe_allow_html=True
    )
    
        
    st.markdown("<br><br>", unsafe_allow_html=True)    
    st.markdown(
        """
        <style>
    .big-font2 {
        font-size:24px !important; 
        }
    </style>
    
    <div class='big-font2'>
       Traitement des valeurs manquantes avec KNN_Imputer.
    </div>
    
    <div class='big-font'>
        Les valeurs manquantes ont été traitées avec la méthode KNN Imputer, basée sur l'approximation des k plus proches voisins. Ce processus remplace chaque donnée absente par une moyenne pondérée de ses voisins les plus proches, assurant ainsi une imputation adaptée et efficace.
        La image suivante illustre les distributions des variables avant et après l'imputation via la méthode KNN, tout en les comparant aux méthodes traditionnelles de moyenne et médiane, lesquelles, bien que plus simples, se révèlent être moins précises.
    </div>
    
    """, unsafe_allow_html=True)
    
    st.image('img/knn_imputer2.png', width=750)
    st.image('img/knn_imputer.png', width=750)
    
    ##Traitement KNN_imputer