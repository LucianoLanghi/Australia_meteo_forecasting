# -*- encoding: utf-8 -*-

import streamlit as st


def app():
    st.write("Conclusion")
    st.markdown(
        """
        ### Constats
        Nos modélisations nous ont réservé plusieurs surprises lors de nos explorations. La première d’entre elle a été 
        la robustesse et l’intérêt du XGBoost, qu’il s’agisse de prédire RainTomorrow ou MaxTemp : pour notre 
        problématique, ce modèle, très rapide à entraîner, rivalise y compris avec des réseaux de neurones bien plus 
        complexes. 
        La seconde surprise a été que malgré le déséquilibre de nos classes sur la variable cible, la plupart des 
        modèles entraînés nous ont offert de très belles performances.
        La troisième surprise est l’apport limité du feature engineering dans notre problématique. Même s’il présente 
        bel et bien un intérêt, en particulier pour effectuer des prédictions plus éloignées dans le futur, nous avons 
        été particulièrement surpris par le faible écart de performances une modélisation effectuée avec les données 
        initiales et celles obtenues après plusieurs mois de feature engineering. C’est une vraie leçon en termes de 
        gestion de projet sur le temps à budgéter sur cet aspect. Il est en effet assez aisé de se perdre dans
         d’innombrables conjectures pour un intérêt potentiellement infime. 
        La quatrième surprise a été la possibilité de prédire la pluie sur une année à partir de l’observation d’une 
        seule journée ! Encore une fois, les performances sont faibles, mais cette possibilité est particulièrement 
        intrigante. Il s’agit là d’un aspect qu’il serait intéressant d’approfondir avec le regard d’un météorologue 
        australien, qui pourrait potentiellement comprendre ce phénomène à partir de son expertise métier.

        ### Limites et perspectives Pour autant, nous restons avec une satisfaction mitigée sur les performances 
        finales de nos notre meilleur modèle. Nous espérions en effet obtenir des prédictions quasiment fiables à 
        100%. Or, nous en sommes très loin avec notre accuracy de 86,6% sur l’ensemble de l’Australie, en particulier 
        au regard du taux de journées non pluvieuses de 77,6%. Nous avons toutefois pu identifier au fil du projet 
        plusieurs pistes qui nous permettraient d’améliorer potentiellement les performances. Tout d’abord, 
        nous avons vu dès la phase exploratoire que la feature Sunshine était absente sur environ la moitié des 
        observations alors qu’elle présentait une forte corrélation avec la variable cible. L’importance de Sunshine 
        a d’ailleurs été confirmée dans les analyses d’interprétabilité des différents modèles. Par conséquent, 
        la première action à mener pourrait être de récolter les valeurs de Sunshine pour le maximum d’observations. 
        Deux autres variables sont très importantes : Humidity3pm et Pressure3pm. Une autre piste pourrait être de 
        récolter le taux d’humidité et la pression atmosphérique à d’autres heures de la journée. Celles, existantes, 
        de 9h du matin sont bien moins importantes, mais peut-être que les modèles gagneraient à ajouter ces taux à 
        14h ou 16h, par exemple. D’ailleurs, n’oublions pas que l’Australie s’étale sur plusieurs fuseaux horaires, 
        allant de GMT+8 à GMT+11 : même si Pressure3pm est relevée à 15h pour chaque Location sur tout le pays, 
        il ne s’agit en réalité pas de la même heure par rapport au soleil. Rainfall est une variable dont la qualité 
        de renseignement est discutable, alors même qu’elle est la source de notre variable cible RainTomorrow : 
        contrairement aux autres variables, qui peuvent simplement être inexistantes pour une journée, 
        Rainfall s’accumule en réalité dans le pluviomètre pendant plusieurs jours jusqu’à ce que le niveau soit 
        relevé par un des bénévoles en charge. Il est probable que les modèles gagneraient en précision si les 
        relevés étaient réellement quotidiens et, a minima, que les valeurs ne se cumulent pas en cas d’absence de 
        relevé. Au-delà des features elles-mêmes, il conviendrait aussi de disposer d’un dataset avec le plus de 
        dates possibles renseignées. Pour rappel, il nous manque pour l’intégralité des lieux trois mois complets, 
        et, pour certains lieux tels Melbourne, il manque plus d’un an et demi en cumulé sur la plage de dates. Ces 
        trous sont de nature à perturber les modélisations par série temporelle. Il serait donc bénéfique de disposer 
        de dates intégralement observées. Notre dataset porte sur dix années, ce qui peut sembler conséquent, 
        mais qui est en réalité assez peu à l’échelle des possibilités du machine learning. Il serait intéressant de 
        pouvoir disposer de relevés sur une période de plusieurs décennies. Etant donnée l’immensité de l’Australie, 
        il pourrait être profitable de disposer de relevés d’autres stations météorologiques afin d’avoir d’une part 
        plus de données, mais également un meilleur maillage géographique, qui permettrait peut-être de toutes 
        nouvelles approches de prédictions basées sur les villes voisines. Il serait aussi peut-être possible 
        d’obtenir de meilleurs résultats avec les RNN en disposant de machines nettement plus puissantes que les 
        nôtres afin de pouvoir multiplier les affinement d’hyperparamètres. Enfin, nous n’avons pas eu le temps 
        d’explorer les transformer, qui pourraient potentiellement proposer des résultats encore meilleurs que les RNN.

        
        """
    )