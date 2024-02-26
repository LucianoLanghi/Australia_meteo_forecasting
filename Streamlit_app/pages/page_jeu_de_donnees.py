# -*- encoding: utf-8 -*-

import streamlit as st
import matplotlib.pyplot as plt

def app(pas):
    st.write(pas.df.shape)
    st.dataframe(pas.df.describe())

    # géographie - raintomorrow / MaxTemp
    fig = pas.synthetise_villes()
    st.plotly_chart(fig)


    # Histogramme temp / pluviometrie
    option = st.selectbox(
        'Sélectionnez une Location',
        (pas.df.Location.unique())
    )
    
    pas.histogramme_temperatures_precipitations(option)
    st.pyplot(plt.gcf())  
    
    pas.graphe_vent("")
    st.pyplot(plt.gcf())  
    
    pas.matrice_corr_quyen(pas.df, "Corrélations entre les variables du dataset initial")
    st.pyplot(plt.gcf())  




