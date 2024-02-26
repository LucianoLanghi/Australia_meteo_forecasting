# -*- encoding: utf-8 -*-

import streamlit as st

def app(pas):
    
    st.markdown(
        """
        Ce projet consiste à prédire des variables météorologiques à partir d’un jeu de données contenant dix ans de
         relevés sur de nombreuses stations météo australienne.
        """
    )

    # ajouter lien Kaggle
    st.write("lien kaggle")
    
    # evolution de RainToday => peu de cohérence géo et temporelle
    fig = pas.animation_variable("RainToday")
    st.plotly_chart(fig)
    
    # l'Australie, c'est grand !
    st.image('img/carte-australie-europe.jpg', caption="l'Australie, c'est grand!", width=768) #, use_column_width=True)
