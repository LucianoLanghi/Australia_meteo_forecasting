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

st.set_page_config(layout="wide")

df=pd.read_csv("datasets/data_process3_knnim_resample_J2.csv", index_col=0)

st.write(df.shape)
st.dataframe(df.describe())


pas = ProjetAustralieSoutenance(df)

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

# evolution de RainToday => peu de cohérence géo et temporelle
fig = pas.animation_variable("RainToday")
st.plotly_chart(fig)

# l'Australie, c'est grand !
st.image('img/carte-australie-europe.jpg', caption="l'Australie, c'est grand!", width=768) #, use_column_width=True)


# Clusterisation
fig = pas.clusterisation_groupee()
st.pyplot(plt.gcf())
st.plotly_chart(fig)

st.image('img/climats_wiki.png', caption="Climats australiens (Wikipédia)", width=512)
