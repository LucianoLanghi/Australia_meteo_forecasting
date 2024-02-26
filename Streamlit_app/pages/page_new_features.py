# -*- encoding: utf-8 -*-

import streamlit as st
import matplotlib.pyplot as plt


def app(pas):
    affichage_clusterisation(pas)

    pas.matrice_corr_quyen(pas.data)
    st.pyplot(plt.gcf())  
    
    
def affichage_clusterisation(pas):
    # Clusterisation
    fig = pas.clusterisation_groupee()
    st.pyplot(plt.gcf())
    st.caption("Dendrogramme de clusterisation")

    st.plotly_chart(fig)
    st.caption("Résulat de la clusterisation en 7 zones climatiques")

    st.image('img/climats_wiki.png', caption="Climats australiens (Wikipédia)", width=512)

