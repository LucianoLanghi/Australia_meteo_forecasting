# -*- encoding: utf-8 -*-

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

    def get_results_bis(self, metric_name):
        X_train, X_test, y_train, y_test = load(self.data_scaled)

        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)
        classes = ["Rain" if c == "Yes" else "No Rain" for c in label_encoder.classes_]

        # LogisticRegression
        data_file = f"{self.path_classification}/logistic_regression/best_params_{metric_name}.joblib"
        best_params = load(data_file)
        clf_lr = LogisticRegression(**best_params, max_iter=10, solver='saga')

        # # KNN
        # data_file = f"{path_classification}/knn/best_params_{metric_name}.joblib"
        # best_params = load(data_file)
        # clf_knn = neighbors.KNeighborsClassifier(**best_params)

        # TreeDecision
        data_file = f"{self.path_classification}/tree_decision/best_params_{metric_name}.joblib"
        best_params = load(data_file)
        clf_tree = tree.DecisionTreeClassifier(**best_params)

        # RandomForest
        data_file = f"{self.path_classification}/random_forest/best_params_{metric_name}.joblib"
        best_params = load(data_file)
        clf_rf = ensemble.RandomForestClassifier(**best_params)

        # XGBoost
        data_file = f"{self.path_classification}/xg_boost/best_params_{metric_name}.joblib"
        best_params = load(data_file)
        clf_xgboost = xgb.XGBClassifier(**best_params, random_state=42)

        # =================================================================================================
        # Calculer les scores
        # =================================================================================================
        df_scores = pd.DataFrame(columns=list_metric_considered, index=list_model)
        fig_rain, ax_rain = plt.subplots()
        fig_no_rain, ax_no_rain = plt.subplots()

        for clf, label in zip([clf_lr, clf_tree, clf_rf, clf_xgboost], list_model):
            print(label)
            clf.fit(X_train, y_train_encoded)

            y_pred = clf.predict(X_test.values)

            accuracy = accuracy_score(y_test_encoded, y_pred)
            recall = recall_score(y_test_encoded, y_pred)
            precision = precision_score(y_test_encoded, y_pred)
            f1 = f1_score(y_test_encoded, y_pred)
            rocauc_rain = round(roc_auc_score(y_test_encoded, clf.predict_proba(X_test)[:, 1]), 4)
            rocauc_no_rain = round(roc_auc_score(y_test_encoded, clf.predict_proba(X_test)[:, 0]), 4)

            df_scores.loc[label, :] = [accuracy, recall, precision, f1, rocauc_rain]

            metrics.RocCurveDisplay.from_estimator(clf, X_test, y_test_encoded,
                                                   ax=ax_rain, label=f'{label} - AUC Rain = {rocauc_rain}')
            metrics.RocCurveDisplay.from_estimator(clf, X_test, y_test_encoded, pos_label=0,
                                                   ax=ax_no_rain, label=f'{label} - AUC Rain = {rocauc_no_rain}')

        df_scores["Model"] = df_scores.index
        ax_rain.set_title("Comparaison des ROC Curve pour Rain")
        ax_no_rain.set_title("Comparaison des ROC Curve pour No Rain")

        return df_scores, fig_rain, fig_no_rain

    def get_results(self, metric_name):
        file_scores = f"{self.path_classification}/Scores_of_modeles_best_{metric_name}.csv"
        file_rocauc_rain = f"{self.path_classification}/ROCCurve_of_Rain_of_modeles_best_{metric_name}.png"
        file_rocauc_no_rain = f"{self.path_classification}/ROCCurve_of_No_Rain_of_modeles_best_{metric_name}.png"

        df_scores = pd.read_csv(file_scores, sep=";", index_col=0)
        del df_scores["Model"]
        return df_scores, file_rocauc_rain, file_rocauc_no_rain


def app():
    st.write("Prédiction de Raintomorrow")
    # st.sidebar.markdown("Prédiction de Raintomorrow")
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
        "Quelle métrique de classification que vous voulez optimiser?",
        list_metric_considered,
        placeholder="Choisir une métrique ..."
    )

    mlm = MachineLearningModels(path_data)
    df_scores, file_rocauc_rain, file_rocauc_no_rain = mlm.get_results(metric)

    col1, col2 = st.columns(2)

    col1.write("\n\n\n\n\n\n\n\n\n")
    col1.write(f"Les scores des modèles en optimisant le {metric}")
    col1.dataframe(df_scores.style.highlight_max(axis=0))
    col2.image(file_rocauc_rain, width=500)


