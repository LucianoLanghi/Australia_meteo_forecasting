# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 14:44:46 2024

@author: Sophie
"""
import pandas as pd
import numpy as np

# dataviz
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# pour couleurs
import plotly.express as px
import plotly.colors as pc

# scaler
from sklearn.preprocessing import MinMaxScaler


class ProjetAustralieSoutenance:

    def __init__(self, data:pd.DataFrame):
        
        self.disp_width = 1200
        self.disp_height = 800

        # donnees originelles
        self.df = pd.read_csv("datasets/weatherAUS.csv")
        self._preprocessing()
        self.is_preprocessing_apres_analyse=False

        # donnees preprocessees
        
        self.X=None
        self.y=None
        #self.data = data.dropna()
        self.data = data
        
        self.data.index = pd.to_datetime(self.data.index)
        
        #self.data = self.data.drop(columns=["SaisonCos"])
        
        # si SaisonCos existe, alors on la renomme en 4pi et on ajoute SaisonCos2pi
        if hasattr(self.data, "SaisonCos"):
            self.data = self.data.rename(columns={'SaisonCos':'SaisonCos4pi'})
            self.data["SaisonCos2pi"] = np.cos(2*np.pi*(self.data.index.day_of_year-1)/365)
            
        #self.data = self.data.drop(columns=["SaisonCos2pi", "SaisonCos4pi"])

        # s'il n'y a que mount ginini en climat 5, on degage
        if (self.data[self.data.Climat==5].Location.nunique()==1):
            self.data = self.data[self.data.Climat!=5]
            
        # palette
        palette_set1 = px.colors.qualitative.Set1
        self.palette=[]
        for i in range(7):
            self.palette.append(pc.unconvert_from_RGB_255(pc.unlabel_rgb(palette_set1[i])))
            
        # libelle des climats
        self.lib_climats = {0:"Côte Est", 1:"Nord", 2:"Centre", 3:"Sud-Est", 4:"Intermédiaire", 5:"Mount Ginini", 6:"Côte Sud"}
        
        # evite de reclaculer au vol
        self.df_resample = self.data
        self._ajoute_colonnes_dates()

    def _preprocessing(self):
        # remplace Yes/No par 0/1
        self.df = self.df.replace({"No":0, "Yes":1})
        self.df.Date = pd.to_datetime(self.df.Date)
        self.df = self.df.set_index("Date", drop=False)

    # ajoute les proprietes des villes au DF
    def _ajoute_prop_locations(self):
        # si la colonne de la latitude a déja été ajoutée, on sort
        if 'lat' in self.df.columns:
            return
        # sinon, on charge les props des villes et on les ajoute au df
        self._charge_villes()
        self.df = pd.merge(self.df, self.df_cities, on='Location')
        self.df.index = self.df.Date
               

    
    def _ajoute_colonnes_dates(self):
        self.df_resample["Date"] = self.df_resample.index
        self.df_resample["Annee"] = self.df_resample.Date.dt.year
        self.df_resample["Mois"] = self.df_resample.Date.dt.month
        self.df_resample["AAMM"] = self.df_resample.Annee.astype(str)+"-"+self.df_resample.Mois.astype(str)

    # remplace les variables categorielles de direction de vent par les composantes x et y
    def remplace_direction_vent(self):
        # si deja substitué: on sort
        if hasattr(self.df, "WindGustDir_X"):
            return
        
        print (self.df.columns)
        self._remplace_direction_vent("WindGustDir", "WindGustSpeed")
        print (self.df.columns)
        self._remplace_direction_vent("WindDir3pm", "WindSpeed3pm")
        print (self.df.columns)
        self._remplace_direction_vent("WindDir9am", "WindSpeed9am")       
        print (self.df.columns)
    
    # remplace une colonne categorielle de direction de vent par deux colonnes numeriques
    def _remplace_direction_vent(self, nom_colonne_dir: str, nom_colonne_speed: str):
        df_direction = pd.DataFrame()
        df_direction["dir"]=["E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW", "N", "NNE", "NE", "ENE"]
        
        # increment pour chacune des directions (sens horaire)
        increment=-np.pi/8
        
        df_direction["rad"]=increment*df_direction.index
        df_direction["sin"]=np.sin(df_direction.rad)
        df_direction["cos"]=np.cos(df_direction.rad)
        
        df_direction.loc[len(df_direction)]=[None, 0, 0, 0]
        
        self.df_dir_vent = df_direction
        
        # jointure pour deduire les cos et sin multipliés par la vitesse
        df_temp = self.df.merge(df_direction, left_on=nom_colonne_dir, right_on="dir")
        df_temp.index = df_temp.Date
        df_temp = df_temp.sort_index()
        
        df_temp[nom_colonne_dir+"_X"]=df_temp.cos*df_temp[nom_colonne_speed]
        df_temp[nom_colonne_dir+"_Y"]=df_temp.sin*df_temp[nom_colonne_speed]
        df_temp[nom_colonne_dir+"_RAD"]=df_temp["rad"]  # on garde rad pour les graphes
        
        self.df = df_temp.drop(columns=["cos", "sin", "rad", "dir", nom_colonne_dir])


    # histogramme des temperatures / precipitations pour une location
    def histogramme_temperatures_precipitations(self, location:str):
        # si l'attribut n'a pas encore été créé, alors on fait la reindexation temporelle
        #if not hasattr(self, "df_resample"):
        #    self.reindexation_temporelle()
        
        
        # retrait des données < 1/1/2009 et >31/12/2016 pour avoir des années complètes
#        df_resample = self.df_resample.loc['2009-01-01':'2016-12-31']

        df_filtre = self.df_resample[self.df_resample.Location == location]
        
        # retrait des données < 1/1/2009 et >31/12/2016 pour avoir des années complètes
        df_filtre = df_filtre.loc['2009-01-01':'2016-12-31']
        
        
        self.da = df_filtre.groupby('AAMM').agg({'MaxTemp': 'mean', 'Rainfall': 'sum', 'Annee':'max', 'Mois':'max'}).reset_index()
        self.da2 = self.da.groupby("Mois").agg({'MaxTemp': 'mean', 'Rainfall': 'mean'}).reset_index()
        
        fig, ax1 = plt.subplots(figsize=(12, 8))
        
        ax1.plot(np.array(self.da2['Mois']), np.array(self.da2['MaxTemp']), label='Température (°C)', color='r')
        ax1.set_yticks(np.arange(0,41,5))
        ax1.set_ylabel("Température (°C)")

        ax2 = ax1.twinx()
        ax2.bar(self.da2['Mois'], self.da2['Rainfall'], label='Précipitations (mm)', color='#06F', alpha=.5)
        ax2.set_yticks(np.arange(0,201,10))
        ax2.set_ylabel("Précipitations (mm)")

        
        nom_mois = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin', 'Juil', 'Août', 'Sep', 'Oct', 'Nov', 'Déc']
        plt.xticks(ticks=np.arange(len(nom_mois))+1, labels=nom_mois)
        nom_titre = 'Températures moyennes mensuelles et cumul des précipations\nAnnées 2009 à 2016'
        nom_titre = nom_titre+" - "+location
        
        plt.title(nom_titre)
        plt.show()

    # anime une variable sur avril 2014
    def animation_variable(self, variable:str="RainToday", discrete:bool=False):
        
        data = self.data.loc[(self.data.index>='2014-04-01')&(self.data.index<='2015-04-30'),:].copy()
        data["Date"] = data.index
        
        if discrete:
            cible = data[variable].astype(str)
        else:
            cible = data[variable]
        
        fig = px.scatter_mapbox(data, 
                                lat='lat', 
                                lon='lng', 
                                hover_name='Location', 
                                color=cible, 
                                #text='Location', 
                                #labels=modele.labels_, 
                                animation_frame="Date",
#                                animation_group="Location",
                                size_max=30, 
                                opacity=.8,
                                #color_continuous_scale=px.colors.qualitative.Plotly
                                color_discrete_sequence=px.colors.qualitative.Set1,
                                #color_discrete_sequence=px.colors.qualitative.T10,
                                range_color=[data[variable].min(), data[variable].max()]
                                ).update_traces(marker=dict(size=30))
                
        fig.update_layout(mapbox_style='open-street-map', width=self.disp_width, height=self.disp_height)
        #fig.show(renderer='browser')      
        return fig

    # affiche climats
    def affiche_climats(self):
        df = self.data.drop("Date", axis=1).reset_index()[['Location', 'Climat', 'lng', 'lat']].drop_duplicates()
        
        fig = px.scatter_mapbox(self.df_moyenne.sort_values(by="Climat"), 
                            lat='lat', 
                            lon='lng', 
                            hover_name='Location', 
                            #text='Location', 
                            size_max=30, 
                            opacity=.8,
                            #color_continuous_scale=px.colors.qualitative.Plotly
                            color_discrete_sequence=px.colors.qualitative.Set1
                            #color_discrete_sequence=px.colors.qualitative.T10
                            ).update_traces(marker=dict(size=30))
    
        fig.update_layout(mapbox_style='open-street-map', width=self.disp_width, height=self.disp_height)
        #fig.show(renderer='browser')      
        return fig


    # représentation geographique sur une carte    
    def synthetise_villes(self):
        self._charge_villes()
        
        #s_rt = self.df.groupby("Location")["RainTomorrow"].mean()
        df_rt = self.df.groupby("Location").agg({"RainTomorrow":'mean', "MaxTemp":'mean', "Pressure9am":'mean', 'Location':'count'})
        df_rt = df_rt.rename(columns={"Location":"Nb"})
        
        df_rt = df_rt.merge(self.df_cities, left_index=True, right_on='Location')
        
        fig = px.scatter_mapbox(df_rt, lat='lat', lon='lng', hover_name='Location', color='MaxTemp', size='RainTomorrow', color_continuous_scale='thermal')
#        fig = px.scatter_geo(df_rt, lat='lat', lon='lng', hover_name='Location', color='MaxTemp', size='RainTomorrow')
        fig.update_layout(mapbox_style='open-street-map', width=self.disp_width, height=self.disp_height)
        #fig.update_layout(margin={'r':0, 't':0, 'l':0, 'b':0})
        #fig.update_geos(projection_type='mercator')

        #fig.show(renderer='browser')             
        return fig

    # charge infos sur villes
    def _charge_villes(self):
        # si deja chargé, on sort
        if hasattr(self, "df_cities"):
            return
        # Créer un DataFrame avec les coordonnées de la ville de Paris        
        self.df_cities = pd.read_csv("datasets/villes_coordonnees.csv",sep=";")     

    # clusterisation des villes en 7 zones climatiques, basées sur la moyenne des variables sur les 10 ans de relevés
    def clusterisation_groupee(self):
        from sklearn.cluster import AgglomerativeClustering, MeanShift, estimate_bandwidth
        from scipy.cluster.hierarchy import linkage, dendrogram

        self._ajoute_prop_locations()

        self.remplace_direction_vent() # besoin de supprimer les variables categorielles
                       
        self.df_moyenne = self.df.groupby("Location").agg(["mean", "std"])
        self.df_moyenne.columns = ['{}_{}'.format(col[0], col[1]) for col in self.df_moyenne.columns]
        
        self.df_moyenne = self.df_moyenne.dropna(axis=1)

        # sur la suite, la normalisation et la clusterization ne se font pas sur les deux dernières colonnes,
        # c'est à dire les coordonnées geographiques, pour que celles-ci n'influence pas l'appartenance à un cluster
        # enleve egalement la premiere colonne, cad la date
        scaler=MinMaxScaler()
        
        mask_df_moyenne = self.df_moyenne.columns.drop(["lat_mean", "lng_mean", "Date_mean", 
                                                        "lat_std", "lng_std", "Date_std"])
                                                        #])

        mask_df_moyenne = mask_df_moyenne.drop(['WindSpeed9am_mean', 'WindSpeed9am_std', 
                                                'WindSpeed3pm_mean', 'WindSpeed3pm_std', 
                                                'WindGustDir_RAD_mean', 'WindGustDir_RAD_std', 
                                                'WindDir3pm_X_mean', 'WindDir3pm_X_std', 
                                                'WindDir3pm_Y_mean', 'WindDir3pm_Y_std', 
                                                'WindDir3pm_RAD_mean', 'WindDir3pm_RAD_std', 
                                                'WindDir9am_X_mean', 'WindDir9am_X_std', 
                                                'WindDir9am_Y_mean', 'WindDir9am_Y_std', 
                                                'WindDir9am_RAD_mean', 'WindDir9am_RAD_std'])

        """
        mask_df_moyenne = mask_df_moyenne.drop(['WindSpeed9am_mean', 
                                                'WindSpeed3pm_mean', 
                                                'WindGustDir_RAD_mean', 
                                                'WindDir3pm_X_mean', 
                                                'WindDir3pm_Y_mean',
                                                'WindDir3pm_RAD_mean', 
                                                'WindDir9am_X_mean', 
                                                'WindDir9am_Y_mean', 
                                                'WindDir9am_RAD_mean'])

        """        
        """
        # retrait des variables bo-quotidiennes
        mask_df_moyenne = mask_df_moyenne.drop(['Temp9am_mean', 'Temp9am_std', 
                                                'Temp3pm_mean', 'Temp3pm_std', 
                                                'Humidity9am_mean', 'Humidity9am_std'
                                                ])
        """
        # retrait de raintomorrow, qui ne concerne pas le relevé météo du jour
        mask_df_moyenne = mask_df_moyenne.drop(['RainTomorrow_mean', 'RainTomorrow_std'
                                                #'RainToday_mean', 'RainToday_std'
                                                ])

        
        
        print ("Variables utilsées pour la clusterisation:", mask_df_moyenne,"\n")
        
        self.df_moyenne[mask_df_moyenne] = scaler.fit_transform(self.df_moyenne[mask_df_moyenne])        

        Z = linkage(self.df_moyenne[mask_df_moyenne], method='ward', metric = 'euclidean')
        
        plt.figure(figsize=(12,8))
        dendrogram(Z, labels=self.df_moyenne.index, leaf_rotation=90., color_threshold=1.5)
        plt.show()

        # 7 clusters (pas optimal, juste pour voir)        
        clf = AgglomerativeClustering(n_clusters=7)
        clf.fit(self.df_moyenne[mask_df_moyenne])
        clust_lab = clf.labels_.astype(str)

        #self.charge_villes()
        self.df_moyenne = self.df_moyenne.reset_index()
        self.df_moyenne["Climat"] = clf.labels_[self.df_moyenne.index]
        
        self.df_climat = self.df_moyenne[["Location", "Climat"]]
        
        fig = px.scatter_mapbox(self.df_moyenne.sort_values(by="Climat"), 
                                lat='lat_mean', 
                                lon='lng_mean', 
                                hover_name='Location', 
                                color=np.sort(clust_lab), 
                                #text='Location', 
                                labels=clf.labels_, 
                                size_max=30, 
                                opacity=.8,
                                #color_continuous_scale=px.colors.qualitative.Plotly
                                color_discrete_sequence=px.colors.qualitative.Set1
                                #color_discrete_sequence=px.colors.qualitative.T10
                                ).update_traces(marker=dict(size=30))
        
        fig.update_layout(mapbox_style='open-street-map', width=self.disp_width, height=self.disp_height)
        #fig.show(renderer='browser')      
        return fig
