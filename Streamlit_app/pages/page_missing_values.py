# -*- encoding: utf-8 -*-

import streamlit as st
import matplotlib.pyplot as plt

def app(pas):
    pas.graphe_taux_na_location_feature()
    st.pyplot(plt.gcf())
    