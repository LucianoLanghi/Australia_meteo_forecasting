# -*- encoding: utf-8 -*-

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def app(pas):
    st.write(pas.df.shape)
    st.dataframe(pas.df.describe())

    # géographie - raintomorrow / MaxTemp
    fig = pas.synthetise_villes()
    st.plotly_chart(fig)

    locations = pas.df.Location.unique()
    
    # Histogramme temp / pluviometrie
    option = st.selectbox(
        'Sélectionnez une Location',
        (locations)
    )
    
    pas.histogramme_temperatures_precipitations(option)
    st.pyplot(plt.gcf())  
    
    location_aus = np.append('', locations)
    option = st.selectbox(
        'Sélectionnez une Location',
        (location_aus)
    )   
    pas.graphe_vent(option)
    st.pyplot(plt.gcf())  
    
    pas.matrice_corr_quyen(pas.df, "Corrélations entre les variables du dataset initial")
    st.pyplot(plt.gcf())  




