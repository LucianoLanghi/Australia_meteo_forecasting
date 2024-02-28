# -*- encoding: utf-8 -*-
import shap
import streamlit as st

import os

import pandas as pd
import xgboost as xgb
from joblib import load
from matplotlib import pyplot as plt
from sklearn import ensemble, tree, metrics
from sklearn import neighbors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, recall_score, precision_score
from sklearn.preprocessing import LabelEncoder
from streamlit import components

print(os.getcwd())

path_data = "./data/RainTomorrow"
list_metric_considered = ['accuracy', 'recall', 'precision', 'f1', 'roc_auc']
list_model = ["LogisticRegression", "TreeDecision", "RandomForest", "XGBoost"]

k = 4
path_classification = path_data
data_original = f"{path_data}/data_original_pre_processing_without_location_knn_imputed_{k}.joblib"
data_scaled = f"{path_data}/data_scaled_pre_processing_without_location_knn_imputed_{k}.joblib"


class MachineLearningModels:
    def __init__(self, path_data):
        self.path_classification = path_data
        self.data_original = f"{path_data}/data_original_pre_processing_without_location_knn_imputed_{k}.joblib"
        self.data_scaled = f"{path_data}/data_scaled_pre_processing_without_location_knn_imputed_{k}.joblib"

    def get_results(self, metric_name):
        file_scores = f"{self.path_classification}/Scores_of_modeles_best_{metric_name}.csv"
        file_rocauc_rain = f"{self.path_classification}/ROCCurve_of_Rain_of_modeles_best_{metric_name}.png"
        file_rocauc_no_rain = f"{self.path_classification}/ROCCurve_of_No_Rain_of_modeles_best_{metric_name}.png"

        df_scores = pd.read_csv(file_scores, sep=";", index_col=0)
        del df_scores["Model"]
        return df_scores, file_rocauc_rain, file_rocauc_no_rain

    def graph_mean_shap_xgboost(self, metric_name):
        X_train, X_test, y_train, y_test = load(data_original)

        file_best_model = f"{path_classification}/xgboost/best_model_{metric_name}.joblib"
        xgb_best = load(file_best_model)

        shap_explainer = shap.TreeExplainer(xgb_best)
        shap_values = shap_explainer(X_test)

        # =============================================================================================
        # SHAP - Mean SHAP Plot
        # =============================================================================================

        fig_mean_shap = plt.figure(figsize=(8, 12))
        shap.plots.bar(shap_values, max_display=20, show=False)
        plt.title(f'Mean Shap XGBoost for {metric_name}', fontsize=20)
        fig_mean_shap.tight_layout()

        # =============================================================================================
        # SHAP - Beeswarm Plot
        # =============================================================================================
        # print("bees")
        fig_beeswarm = plt.figure(figsize=(8, 20))
        shap.plots.beeswarm(shap_values, max_display=20, show=False)
        plt.title(f'Beeswarm XGBoost for {metric_name}', fontsize=20)
        fig_beeswarm.tight_layout()

        # =============================================================================================
        # SHAP - Dependence Plots
        # =============================================================================================

        list_variables = ["Humidity3pm", "Pressure3pm", "WindGustSpeed", "Sunshine"]
        fig_dependences, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 11))
        axes = axes.ravel()
        for i, variable in enumerate(list_variables):
            shap.plots.scatter(shap_values[:, variable], ax=axes[i])
            axes[i].set_title(variable)
        fig_dependences.suptitle("Dépendence de la valeur SHAP à une seule variable", fontsize=20)
        plt.tight_layout()

        return fig_mean_shap, fig_beeswarm, fig_dependences


sous_pages = ["***Approche de Machine Learning***",
              "***Approche de Deep Learning***"]


def set_sous_pages():
    page = st.sidebar.radio("", sous_pages)
    st.header(page)
    st.sidebar.markdown(page)
    return page


class ApprocheMachineLearning:
    @staticmethod
    def show_different_ML_models(metric_name):
        mlm = MachineLearningModels(path_data)
        df_scores, file_rocauc_rain, file_rocauc_no_rain = mlm.get_results(metric_name)

        col1, col2 = st.columns(2)

        col1.write("\n\n\n\n\n\n\n\n\n")
        col1.write(f"Les scores des modèles en optimisant le {metric_name}")
        col1.dataframe(df_scores.style.highlight_max(axis=0))
        col2.image(file_rocauc_rain, width=500)

    @staticmethod
    def show_SHAP_XGBoost(metric_name):
        st.markdown("### Interprétabilité du modèle XGBoost")

        col1, col2, col3 = st.columns(3)

        # file_mean_shap = f"{path_classification}/mean_SHAP_barplot.png"
        # col1.image(file_mean_shap, width=300)
        #
        # file_beeswarm = f"{path_classification}/beeswarm.png"
        # col2.image(file_beeswarm, width=300)
        mlm = MachineLearningModels(path_data)
        fig_mean_shap, fig_beeswarm, fig_dependences = mlm.graph_mean_shap_xgboost(metric_name)

        col1.pyplot(fig_mean_shap)
        col2.pyplot(fig_beeswarm)
        col3.pyplot(fig_dependences)

    @staticmethod
    def show_XGBoost_different_levels():
        st.markdown(
            '''
            ### Modèle XGBoost avec trois niveau de finesse
            -	Un niveau macro, avec des modèles portant sur l’ensemble des données australiennes du jeu de données
            -	Un niveau micro, où nous génèrerons des modèles spécifiques pour chaque Location
            -	Un niveau intermédiaire, dans lequel nous aurons clusterisé l’Australie en plusieurs zones climatiques
            
            Comparons maintenant les performances d’un XGBoost entraîné en optimisant l’AUC-ROC sur l’ensemble du jeu de 
            données avec des modélisations ciblant chaque station météo d’une part (filtrage par la variable Location), 
            et chaque zone climatique d’autre part (filtrage via la variable Climat issue de la clusterisation). 
            
            '''
        )

    @staticmethod
    def show_prediction_with_XGBoost(metric_name):
        st.markdown(
            '''
            ### Prédiction example avec XGBoost
            
            '''
        )

    def show(self):
        st.markdown(
            '''
            ### Modèles classiques de classification du Machine Learning
            Tout d'abord nous utilisons tous les observations du dataset issues du feature engineering que nous avons 
            effectués dans les parties précédentes et appliquons des modèles de classification tels que
            **Logistic Regression**, **Decision Tree**, **Random Forest** et **XGBoost**.
    
            Les hyperparamètres de chaque modèle seront optimisés via des tests manuels, des GridSearch, mais aussi à 
            l’aide de la bibliothèque Hyperopt, en cherchant à maximiser diverses métriques telles que l’accuracy, la 
            précision, le recall, le score F1 et le ROC AUC. Nous expliquerons les enjeux portant sur le choix d’une 
            métrique adaptée.
    
            '''
        )

        metric = st.selectbox(
            "***Quelle métrique de classification que vous voulez optimiser?***",
            list_metric_considered,
            index=None,
            placeholder="Choisir une métrique ..."
        )

        if metric is not None:
            # Résultats des différents modèles de Machine Learning
            self.show_different_ML_models(metric)
            # Résultats du modèle XGBoost aux 3 niveaux de finesse
            self.show_XGBoost_different_levels()
            # Interprétabilité du modèle XGBoost
            self.show_SHAP_XGBoost(metric)
            # Prédiction example
            self.show_prediction_with_XGBoost(metric)


def approche_deep_learning():
    st.markdown(
        '''
        ### Modèle RNN
        '''
    )


# La fonction principale qui est appellée dans streamlit_app.py pour afficher la page
def app():
    sous_page = set_sous_pages()
    st.write("Prédiction de Raintomorrow")

    if sous_page == sous_pages[0]:
        ApprocheMachineLearning().show()

    if sous_page == sous_pages[1]:
        approche_deep_learning()
