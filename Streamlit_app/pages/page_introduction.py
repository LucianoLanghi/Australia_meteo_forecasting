# -*- encoding: utf-8 -*-

import streamlit as st


def app():
    st.write("Introduction")
    st.sidebar.markdown("Introduction")
    st.markdown(
        """
        Ce projet consiste à prédire des variables météorologiques à partir d’un jeu de données contenant dix ans de
         relevés sur de nombreuses stations météo australienne.
        """
    )
