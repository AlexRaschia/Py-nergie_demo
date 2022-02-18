# -*- coding: utf-8 -*-
"""
Projet Py-nergie
DS Bootcamp Juin 2021

"""

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#from sklearn.preprocessing import StandardScaler, MinMaxScaler

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout="centered")

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', None)


# Importation du dataset
@st.cache(suppress_st_warning=True)
def importandtreat_dfs():
    Ener = pd.read_csv('eco2mix-regional-cons-def.csv', sep=';', low_memory=False)


    # Tri par date et heure puis ordre alphabétique des Régions
    Ener_sorted = Ener.sort_values(by = ['Date - Heure', 'Région'], ascending = True)
    Ener_sorted = Ener_sorted.reset_index()
    Ener_sorted = Ener_sorted.drop(['index'], axis=1)

    # Suppression des colonnes de flux, code INSEE et Nature
    flux=[col for col in Ener_sorted if col.startswith('Flux')]
    Ener_sorted.drop(flux, axis=1, inplace=True)
    Ener_sorted.drop(['Code INSEE région'], axis=1, inplace=True)
    Ener_sorted.drop(['Nature'], axis=1, inplace=True)

    Ener_sorted['Date - Heure'] = pd.to_datetime(Ener_sorted['Date - Heure'])

    # Remplacement des fillna pour les colonnes de Prod et Conso
    colnam_NaNtoSup = [col for col in Ener_sorted if '(MW)' in col]
    for name in colnam_NaNtoSup:
        Ener_sorted[name] = Ener_sorted[name].fillna(0)

    # creation de la colonne Production qui représente la somme de toutes les productions
    Ener_sorted.insert(5, 'Production (MW)', 0)
    # calcul de la Production pour chaque ligne
    Ener_sorted['Production (MW)'] = Ener_sorted['Thermique (MW)'] + Ener_sorted['Nucléaire (MW)'] \
                                    + Ener_sorted['Eolien (MW)'] + Ener_sorted['Solaire (MW)'] \
                                    + Ener_sorted['Hydraulique (MW)'] + Ener_sorted['Pompage (MW)'] \
                                    + Ener_sorted['Bioénergies (MW)']

    # creation des colonnes Année, Jour et Mois
    Ener_sorted.insert(2, 'Jour', pd.DatetimeIndex(Ener_sorted['Date']).day)
    Ener_sorted.insert(3, 'Mois', pd.DatetimeIndex(Ener_sorted['Date']).month)
    Ener_sorted.insert(4, 'Année', pd.DatetimeIndex(Ener_sorted['Date']).year)
    
    
    
    # insérer l'import et le traitement des dataframes complémentaires 
    
    # Importation des données de Températures par Région et par jour (ODRE)
    Temp_per_day = pd.read_csv('temperature-quotidienne-regionale.csv', sep=';')
    Temp_per_day = Temp_per_day.sort_values(by = ['Date', 'Région'], ascending = True)
    Temp_per_day = Temp_per_day.reset_index()
    Temp_per_day = Temp_per_day.drop(['index', 'Code INSEE région'], axis=1)
    
    # Importation des données de population par an (INSEE)
    Pop_per_year = pd.read_csv('INSEE_pop_2016-2021.csv', sep=';')
    Pop_per_year = Pop_per_year.melt(id_vars="Région", var_name="Année", value_name="Population")
    Pop_per_year = Pop_per_year.sort_values(by = ['Année', 'Région'], ascending = True)
    Pop_per_year = Pop_per_year.reset_index()
    Pop_per_year = Pop_per_year.drop(['index'], axis=1)
    
    # Importation des données entreprises (INSEE) par secteur (primaire, secondaire, tertiaire) et taille (micro, PME, ETI, GE)
    Ent_2016 = pd.read_csv('stat_ent_2016_final.csv', sep=';')
    Ent_2017 = pd.read_csv('stat_ent_2017_final.csv', sep=';')
    Ent_2018 = pd.read_csv('stat_ent_2018_final.csv', sep=';')
    Ent_2016.insert(1, 'Année', 2016)
    Ent_2017.insert(1, 'Année', 2017)
    Ent_2018.insert(1, 'Année', 2018)
    # Création du dataframe Ent à partir des dataframes annuels par concatenation
    Ent_int = pd.concat([Ent_2016,Ent_2017], axis=0)
    Ent = pd.concat([Ent_int, Ent_2018], axis=0)
    Ent = Ent.sort_values(by = ['Année', 'Région'], ascending = True)
    Ent = Ent.reset_index()
    Ent = Ent.drop(['index'], axis=1)
    
    # prétrtaitement pour ML
    Col_todel = Ener_sorted.columns[8:]
    Ener_light = Ener_sorted.copy()
    Ener_light.drop(Col_todel, axis=1, inplace=True)
    Ener_light.drop(['Jour', 'Mois', 'Année'], axis=1, inplace=True)
    # Récupération des pics journaliers sur la Puissance Consommée
    Energy = Ener_light.groupby(["Date","Région"])["Consommation (MW)"].max().reset_index()
    # insertion de l'année dans le datafreame Energy
    Energy.insert(1, 'Année', pd.DatetimeIndex(Energy['Date']).year)
    # Selection des data sur les années 2016 à 2018
    Energy_selection = Energy[(Energy["Année"]>2015) & (Energy["Année"]<2019)].reset_index()
    Energy_selection = Energy_selection.drop(['index'], axis=1)
    # insertion de la colonne pour le merge Date-Région
    Energy_selection.insert(0, 'Date-Reg', Energy_selection['Date'] + ' ' + Energy_selection['Région'])
    Temp_per_day.insert(0, 'Date-Reg', Temp_per_day['Date'] + ' ' + Temp_per_day['Région'])
    Temp_per_day.drop(['Date', 'Région'], axis=1, inplace=True)  
    Pop_per_year.insert(0, 'An-Reg', Pop_per_year['Année'] + ' ' + Pop_per_year['Région'])
    Pop_per_year.drop(['Année', 'Région'], axis=1, inplace=True)
    # Fusion des data
    data_fusion = Energy_selection.merge(right = Temp_per_day, on = 'Date-Reg', how = 'left')
    data_fusion.drop('Date-Reg', axis=1, inplace=True)
    data_fusion['Année'] = data_fusion['Année'].astype(str)
    # insertion de la colonne pour le merge Année-Région
    data_fusion.insert(0, 'An-Reg', data_fusion['Année'] + ' ' + data_fusion['Région'])
    # Fusion des data avec la population
    data_fusion_2 = data_fusion.merge(right = Pop_per_year, on = 'An-Reg', how = 'left')
    # insertion de la colonne pour le merge Année-Région
    Ent['Année']=Ent['Année'].astype(str)
    Ent.insert(0, 'An-Reg', Ent['Année'] + ' ' + Ent['Région'])
    Ent.drop(['Année', 'Région'], axis=1, inplace=True)
    # Fusion des data avec les données entreprises
    data = data_fusion_2.merge(right = Ent, on = 'An-Reg', how = 'left')
    data.drop(['An-Reg', 'Année', 'TMin (°C)', 'TMax (°C)'], axis=1, inplace=True)
    
        
    return Ener, Ener_sorted, Temp_per_day, Pop_per_year, Ent, data



Ener, Ener_sorted, Temp_per_day, Pop_per_year, Ent, data = importandtreat_dfs()


# Création d'un dataframe par année
Ener_sorted_2013 = Ener_sorted[Ener_sorted['Année'] == 2013]
Ener_sorted_2014 = Ener_sorted[Ener_sorted['Année'] == 2014]
Ener_sorted_2015 = Ener_sorted[Ener_sorted['Année'] == 2015]
Ener_sorted_2016 = Ener_sorted[Ener_sorted['Année'] == 2016]
Ener_sorted_2017 = Ener_sorted[Ener_sorted['Année'] == 2017]
Ener_sorted_2018 = Ener_sorted[Ener_sorted['Année'] == 2018]
Ener_sorted_2019 = Ener_sorted[Ener_sorted['Année'] == 2019]
Ener_sorted_2020 = Ener_sorted[Ener_sorted['Année'] == 2020]
Ener_sorted_2021 = Ener_sorted[Ener_sorted['Année'] == 2021]




# Définition du menu de navigation
st.sidebar.header("Projet Py-Nergie")
choix = st.sidebar.radio("Menu de Navigation",
                         ('Présentation du Projet',
                          'Aspects Techniques',
                          'Data Sets',
                          'Dataviz Consommation',
                          'Dataviz Production',
                          'Modèle SARIMA',
                          'Modèles de Régression'))


# Définition page Présentation du projet
if choix == 'Présentation du Projet':
    
    st.title("Projet Py-Nergie - DS Juin à Sept 2021")
    st.write("")
    st.write("")
    
    st.image("pylone.jpg")
    st.write("")
    
    st.header("Présentation du Projet")
    st.write('Le sujet de ce projet porte sur l’analyse des données de production\n'
    'et de consommation du réseau électrique français.  \n' 
    'La source du jeu de données est celle de l’ODRE (Open Data Réseaux Energies)\n'
    'avec un accès à toutes les informations de consommation et de production\n'
    'par Région et par filière jour par jour (toutes les 1/2 heures) depuis 2013.')
    
    st.write("")
    st.image("ODRE.jpg")
    st.image("ODRE2.jpg")
    st.write("")
    
    st.header("Objectifs du projet")
    st.write('Les objectifs du projet s’expriment à travers trois questionnements\n'
    'que nous nous sommes posés et auxquels nous avons voulu répondre \n'
    'dans un analyse détaillée présentée dans ce rapport.  \n'
    'Les trois problématiques se résument ainsi :\n'
    ' - Comment est assuré l’équilibre entre consommation et production\n'
    '   au niveau national et régional ? \n'
    ' - Quelles sont les sources d’énergies au niveau national et régional qui contribuent\n'
    '   à satisfaire les besoins d’électricité et dans quelles proportions ?\n'
    ' - Sommes-nous capables de prédire correctement la consommation avec un/des\n'
    '   modèle(s) de machine learning afin de prévoir les besoins de production ?\n')

    
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.subheader("Crédits")
    st.write("Projet de fin de formation Data Scientist")
    st.write("Auteurs : Samira FLICI, Joëlda KPODJA, Alexandre RASCHIA")
    st.write("Mentor projet : Frédéric FRANCINE")
    st.write("Organisme formateur : DataScientest")
    



# Définition page Aspects Techniques
elif choix == 'Aspects Techniques':
    st.title("Aspects techniques du réseau électrique")
    
    st.write("Le développement des usages électriques depuis le milieu du XXe siècle a abouti \n"
             "à la construction d’un système de production centralisé, associé à un réseau \n"
             "électrique interconnecté et maillé à l’échelle nationale et continentale.  \n"
             "Ce réseau électrique est constitué de 3 types de réseaux :")
    
    st.subheader("Le réseau de transport")
    st.write(" - le réseau de transport est basé sur une structure de réseau maillée \n"
             "(«autoroutes de l’énergie»). Ils est à haute tension (225kV et 400 kV) et a pour \n"
             "but de transporter l'énergie des grands centres de production vers les régions \n"
             "consommatrices d'électricité. Les grandes puissances transitées imposent des lignes \n"
             "électriques de forte capacité de transit, ainsi qu'une structure maillée (ou interconnectée)")
    st.image("réseau transport 400 a 225 kV.jpg", width=350)

    link1 = '[source : RTE](https://assets.rte-france.com/prod/public/2020-07/SDDR%202019%20Chapitre%2002%20-%20Le%20renouvellement%20du%20r%C3%A9seau%20existant.pdf)'
    st.markdown(link1, unsafe_allow_html=True)

    
    st.subheader("Le réseau de répartition")
    st.write(" - Le réseaux de répartition (haute tension de l'ordre de 63kV et 90 kV) a pour but \n"
             "d'assurer à l'échelle régionale la fourniture d'électricité. L'énergie y est injectée \n"
             "essentiellement par le réseau de transport via des transformateurs, mais également par \n"
             "des centrales électriques de moyennes puissances (inférieures à environ 100 MW). \n"
             "Le réseau de répartition est distribué de manière assez homogène sur le territoire d'une région.")  
    st.image("réseau repartition 90 a 63kV.jpg", width=350)
    
    st.markdown(link1, unsafe_allow_html=True)
    
    st.subheader("Les réseaux publics de distribution")
    st.write(" - Les réseaux publics de distribution d'électricité desservent en moyenne et basse tension \n"
             "(20 kV et 400 V), selon une architecture en arborescence, les consommateurs finaux et \n"
             "les clients domestiques et professionnels (commerçants, artisans, petites industries). \n"
             "Leur longueur cumulée dépasse 1,3 million de kilomètres. L’interface entre les réseaux \n"
             "moyenne et basse tension est assurée par quelque 700 000 « postes de distribution ».\n"
             "Le développement de la production d’énergie décentralisée (éolien, photovoltaïque, etc.) \n"
             "et de nouveaux usages (autoproduction, électromobilité, etc.) modifient le rôle des réseaux \n"
             "de distribution qui deviennent collecteurs de l'énergie produite par les plus petites \n"
             "installations de production.")
    
    st.subheader("Synthèse du réseau électrique de la production à la consommation")
    st.image("synoptique réseaux.jpg")
    
    link2 = "[source : Commmision de régulation de l'énergie](http://modules-pedagogiques.cre.fr/m1/index.html)"
    st.markdown(link2, unsafe_allow_html=True)
    



# Définition page Data Sets
elif choix == 'Data Sets':
    st.title("Descriptif des Data Sets")
    st.subheader("Jeu de données initial éco2mix")
    st.write("Données éCO2mix régionales consolidées et définitives (janvier 2013 à juin 2021)")
    st.write("https://opendata.reseaux-energies.fr/explore/dataset/eco2mix-regional")
    st.write("Ce jeu de données, présente les données régionales consolidées depuis janvier 2020 \n"
             "et définitives (de janvier 2013 à décembre 2019) issues de l'application éCO2mix.  \n"
             "Elles sont élaborées à partir des comptages et complétées par des forfaits.")
    
    st.markdown("__Extrait Data Set initial après import__")
    st.dataframe(Ener.head())
    
    st.write("")
    st.write("")
    st.markdown("____")
    
    st.subheader("Température quotidienne régionale")
    st.write("https://opendata.reseaux-energies.fr/explore/dataset/temperature-quotidienne-regionale")
    st.write("Ce jeu de données présente les températures minimales, maximales et moyennes quotidiennes\n"
             "(en degré Celsius), par région administrative française, du 1er janvier 2016 à aujourd'hui.  \n"
             "Il est basé sur les mesures officielles du réseau de stations météorologiques françaises.")
    
    st.markdown("__Extrait Data Set Température après import__")
    st.dataframe(Temp_per_day.head())

    st.write("")
    st.write("")
    st.markdown("____")
    
    st.subheader("Estimation de population par région, sexe et grande classe d'âge - Années 1975 à 2021")
    st.write("https://www.insee.fr/fr/statistiques/1893198")
    st.write("Ce jeu de données brutes présente l’estimation de population par région, sexe et classe d’âge.  \n"
             "Les données sont organisées par années (1 onglet excel / 1 année).  \n"
             "Pour des facilités d’utilisation et les besoins du projet, nous avons regroupé toutes\n"
             "les données dont nous avions besoin dans un fichier csv pour plus de commodité.  \n"
             "Les données s’échelonnent de 2016 à 2021, pour une synchronisation avec les données de températures.")
    
    st.markdown("__Extrait Data Set Population après import__")
    st.dataframe(Pop_per_year.head())

    st.write("")
    st.write("")
    st.markdown("____")
    
    st.subheader("Démographie des entreprises et des établissements pour les années 2016, 2017 et 2018")
    st.write("https://www.insee.fr/fr/statistiques?taille=100&debut=0&theme=38&categorie=3")
    st.write("Démographie des entreprises et des établissements pour l'année 2016")
    st.write("https://www.insee.fr/fr/statistiques/3650551?sommaire=2665485")
    st.write("Fichier Stocks d'établissements au format dbf")
    st.write("Ces fichiers portent sur les établissements et les organismes en activité au 31 décembre 2015\n"
             "dont le siège est situé en France métropolitaine et dans les départements d'outre-mer.  \n"
             "Les fichiers de stocks d'établissements en activité au 31 décembre 2016 contiennent \n"
             "3 107 674 enregistrements regroupant les caractéristiques des 6 067 881 établissements concernés.")
    st.write("La quantité d’information comprise dans ce fichier ne permet pas d’être traité avec Excel.  \n"
             "En effet, Excel est limité à 1 048 576 lignes. Après quelques recherches, nous avons finalement \n"
             "réussi à traiter les données brutes avec Access en formulant des requêtes ensuite exploitables sous Excel afin de préparer un fichier csv.")
    
    st.markdown("__Extrait Data Set Entreprises après import__")
    st.dataframe(Ent.head())



    


# Définition page Dataviz Consommation
elif choix == 'Dataviz Consommation':
    st.title("Analyse de la Consommation")
    
    # Affichage de la Consommation Instantannée moyenne mensuelle par Région / année au choix
    st.subheader("Observation de la Consommation par Région sur une année")
    st.sidebar.subheader("Puissance Consommée")
    year_choice = st.sidebar.selectbox("Choisir une année",['2013','2014','2015','2016','2017','2018','2019','2020'])
    if year_choice == '2013' :
        df_yr = Ener_sorted_2013
    elif year_choice == '2014' :     
        df_yr = Ener_sorted_2014
    elif year_choice == '2015' :     
        df_yr = Ener_sorted_2015
    elif year_choice == '2016' :     
        df_yr = Ener_sorted_2016
    elif year_choice == '2017' :     
        df_yr = Ener_sorted_2017
    elif year_choice == '2018' :     
        df_yr = Ener_sorted_2018
    elif year_choice == '2019' :     
        df_yr = Ener_sorted_2019
    elif year_choice == '2020' :     
        df_yr = Ener_sorted_2020
    
       
    fig = plt.figure(figsize=(14,8))
    ax = plt.subplot(111)
    sns.lineplot(data=df_yr, x='Mois', y='Consommation (MW)', hue='Région', palette="Paired", ci=None)
    plt.ylabel("Puissance (MW)")
    str_title1 = "Moyennes mensuelles de Consommation Instantannée par Région en"+" "+year_choice
    plt.title(str_title1)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    st.pyplot()
    
    st.write("On note sur ces graphiques des variations saisonnières avec des valeurs hautes en hiver et basses en été.  \n"
             "Ces variations de consommation s'expliquent ainsi :\n"
             " - le besoin de chauffage en hiver chez les particuliers et dans les entreprises augmente la consommation d'électricité (en 2017, environ 40% de la population se chauffe avec de l'électricité : convecteurs, radiateurs à inertie, panneaux rayonnants, planchers chauffants, pompes à chaleur),\n"
             " - la baisse de l'activité des entreprises (Industries et Tertiaires) en été pour les congés; cumulée à l'absence de chauffage créent le creux caractéristique de la période d'été.\n")

    st.write("")
    st.write("")
    
    # Affichage de la Consommation instantannée par Région sur une semaine en janvier 2013
    st.subheader("Consommation Instantannée sur une période au choix")
    st.text("Recommandation : 1 ou 8 jours")
    month1 = st.sidebar.slider("Choisir le mois", 1, 12)
    j_min1 = st.sidebar.slider("jour début période", 1, 31)
    j_max1 = st.sidebar.slider("jour fin période", 1, 31)
    if j_min1 >= j_max1 : 
        st.error("Mauvaise définition de la période de temps")
    else : 
        fig = plt.figure(figsize=(14,8))
        ax = plt.subplot(111)
        df_M = df_yr[df_yr['Mois']==month1]
        df_week = df_M[(df_M['Jour']>=j_min1) & (df_M['Jour']<j_max1)]
        sns.lineplot(data=df_week, x='Date - Heure', y='Consommation (MW)', hue='Région', palette="Paired", ci=None)
        plt.ylabel("Puissance (MW)")
        j_min1 = str(j_min1)
        j_max1 = str(j_max1)
        month1 = str(month1)
        str_title2 = "Consommation instantannée par Région du "+j_min1+" au "+j_max1+" du mois "+month1+" de l'année "+year_choice
        plt.title(str_title2)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        st.pyplot()

    st.write("Observations : Quelle que soit la semaine observée dans l'année (janvier, avril, juillet et octobre) et quelque soit la Région, on observe un schéma récurrent de consommation quotidienne.\n"
             " - le minimum de consommation se situe aux alentours de 04h00 lorsque la majorité de la population dort.\n"
             " - Ensuite la consommation croît fortement jusqu'à 08h00 ce qui correspond à la montée en charge du matin : le réveil et l'allumage des premiers appareils électriques et chauffages/chauffe-eau.\n"
             " - Par la suite, la consommation croît plus modérément avec la mise en route des activités industrielles et de bureau jusqu'à un pic entre 12h00 et 13h00.\n"
             " - Pour l'après-midi, la consommation décroit légèrement jusqu'aux environs de 18h00.\n"
             " - Par la suite, le pic de consommation quotidien est observé entre 18h00 et 22h00 lorsque les français rentrent chez eux et allument alors simultanément le chauffage, la lumière et les autres appareils électroménagers (lave-linge, lave-vaisselle, télévision, …)\n"
             " - Un dernier pic apparait aux alentours de 23h00 et correspond probablement au lancement d’appareils de type lave-linge, sèche-linge et lave-vaisselle pendant les heures « creuses ».\n")


    st.write("")
    st.write("")
    st.write("")
    st.subheader("Observation de la Consommation Nationale entre 2013 et 2020")

    # Création du DataFrame de Consommation Nationale
    Ener_Conso_Nat = Ener_sorted.groupby(["Année","Mois","Jour","Heure"])["Consommation (MW)"].sum().reset_index()
    
    # Affichage de la Consommation par Année entre 2013 et 2020
    fig = plt.figure(figsize=(14,8))
    ax = plt.subplot(111)
    sns.lineplot(data=Ener_Conso_Nat, x='Mois', y='Consommation (MW)', hue='Année', palette="Paired", ci=None)
    plt.ylabel("Puissance (MW)")
    plt.title("Moyennes mensuelles des Consommations instantannées pour les Années entre 2013 et 2020")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    st.pyplot()
    
    st.write("On note une certaine stabilité dans la consommation instantanée moyennée par mois.  \n"
             "On retrouve cependant l'effet du confinement qui apparait clairement à partir de mars sur la courbe de l'année 2020.\n")


    st.write("")
    st.write("")
    st.write("")
    st.subheader("Observation de la Quantité d'énergie consommée par mois entre 2013 et 2020")

    # Calcul de QEner_Conso_Nat
    QEner_Conso_Nat_M = Ener_sorted.groupby(["Année","Mois"])["Consommation (MW)"].sum()/2/1000
    QEner_Conso_Nat_M = QEner_Conso_Nat_M.reset_index()
    QEner_Conso_Nat_M = QEner_Conso_Nat_M.rename(columns={"Consommation (MW)": "Consommation (GW.h)"})
    # Pour avoir la quantité en GigaWatt.heure par an : 
    # il faut intégrer la production instantée en sommant et divisant par 2 (mesure toute les 0.5 heure)
    # diviser par 1000 pour passer de MW;h en GW.h

    # Affichage de la Quantité d'Energie Consommée par mois entre 2013 et 2020
    fig = plt.figure(figsize=(14,8))
    ax = plt.subplot(111)
    sns.barplot(data=QEner_Conso_Nat_M, x='Mois', y='Consommation (GW.h)', hue='Année', palette="Paired", ci=None)
    plt.ylabel("Quantité d'Energie (GW.h)")
    plt.ylim(25000, 60000)
    plt.title("Consommation par mois entre 2013 et 2020")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    st.pyplot()

    
    st.write("")
    st.write("Zoom sur la période de Janvier à Mars")
    st.image("zoom_conso.png")
    
    st.write("")
    st.markdown("__Effet Confinement COVID-19 :__ ")
    st.write("On note un écart important à la baisse sur les mois d'avril et mai 2020 par rapport aux mêmes mois des autres années.")
    
    st.write("")
    st.markdown("__Effet Température :__ ")
    st.write("Pic sur Janvier 2017  \n"
             "http://www.meteofrance.fr/actualites/45600575-janvier-2017-froid-et-peu-arrose  \n"
             "Ce mois de janvier a été froid avec une température moyenne inférieure de 1.9 °C aux normales.")
    
    st.write("")
    st.write("Pic sur Février 2018  \n"
             "https://www.meteocontact.fr/climatologie/france/fevrier-2018  \n"
             "Ce mois de février a été bien froid ... De telles températures font de ce mois de février le plus froid depuis 2012.")
    
    st.write("")
    st.write("Pic sur Mars 2013  \n"
             "https://meteofrance.com/magazine/meteo-histoire/les-grands-evenements/11-au-15-mars-2013-des-chutes-de-neige-exceptionnelles  \n"
             "Un froid record : un épisode hivernal tardif est survenu du 11 au 15 mars et a concerné la quasi-totalité du pays.")
    
    st.write("")
    st.write("Pic sur Mars 2018  \n"
             "http://www.meteofrance.fr/actualites/61051700-mars-2018-agite  \n"
             "La température, en moyenne de 8,2 °C sur le mois et sur le pays, a été inférieure à la normale* de 0,5 °C. \n"
             "Ce mois de mars a été plus froid que janvier, qui avait bénéficié d'une douceur exceptionnelle avec 8,4 °C en moyenne sur le pays. \n"
             "Voir consommation de janvier exceptionnellement basse.")




# Définition Page Dataviz Production
elif choix == 'Dataviz Production':
    st.title("Analyse de la Production")


    # Création du dataframe Production Nationale
    Ener_Prod_Nat = Ener_sorted.groupby(["Date","Heure"]).sum().reset_index()
    # sommation des Région pour trouver le Puissance Instantannée Produite au niveau National en MW
    
    # Création des colonnes Jour, Mois, Année
    Ener_Prod_Nat['Jour'] = pd.DatetimeIndex(Ener_Prod_Nat['Date']).day
    Ener_Prod_Nat['Mois'] = pd.DatetimeIndex(Ener_Prod_Nat['Date']).month
    Ener_Prod_Nat['Année'] = pd.DatetimeIndex(Ener_Prod_Nat['Date']).year
    
    # Drop des TCO
    Ener_Prod_Nat = Ener_Prod_Nat.drop(['TCO Thermique (%)', 
                                    'TCO Nucléaire (%)', 
                                    'TCO Eolien (%)', 
                                    'TCO Solaire (%)', 
                                    'TCO Hydraulique (%)', 
                                    'TCO Bioénergies (%)'], axis=1)
    
    Ener_Prod_Nat.insert(2, 'Date - Heure', Ener_Prod_Nat['Date']+'T'+Ener_Prod_Nat['Heure'])
    Ener_Prod_Nat['Date - Heure'] = pd.to_datetime(Ener_Prod_Nat['Date - Heure'])
    Ener_Prod_Nat.insert(7, 'Prod_and_EchPhy', Ener_Prod_Nat['Production (MW)']+Ener_Prod_Nat['Ech. physiques (MW)'])
    Ener_Prod_Nat.insert(8, 'Diff_Prod_Conso', Ener_Prod_Nat['Prod_and_EchPhy']-Ener_Prod_Nat['Consommation (MW)'])

    # Création des dataframes de prod par année 
    Ener_Prod_Nat_2013 = Ener_Prod_Nat[Ener_Prod_Nat['Année'] == 2013]
    Ener_Prod_Nat_2014 = Ener_Prod_Nat[Ener_Prod_Nat['Année'] == 2014]
    Ener_Prod_Nat_2015 = Ener_Prod_Nat[Ener_Prod_Nat['Année'] == 2015]
    Ener_Prod_Nat_2016 = Ener_Prod_Nat[Ener_Prod_Nat['Année'] == 2016]
    Ener_Prod_Nat_2017 = Ener_Prod_Nat[Ener_Prod_Nat['Année'] == 2017]
    Ener_Prod_Nat_2018 = Ener_Prod_Nat[Ener_Prod_Nat['Année'] == 2018]
    Ener_Prod_Nat_2019 = Ener_Prod_Nat[Ener_Prod_Nat['Année'] == 2019]
    Ener_Prod_Nat_2020 = Ener_Prod_Nat[Ener_Prod_Nat['Année'] == 2020]
    
    
    
    st.subheader("Production vs Consommation sur une période au choix")
    st.text("Recommandation : 1 ou 8 jours")
    
    st.sidebar.subheader("Observation Puissances")
    year_prod = st.sidebar.selectbox("Choisir une année",['2013','2014','2015','2016','2017','2018','2019','2020'])
    if year_prod == '2013' :
        df_yr_prod = Ener_Prod_Nat_2013.copy()
    elif year_prod == '2014' :     
        df_yr_prod = Ener_Prod_Nat_2014.copy()
    elif year_prod == '2015' :     
        df_yr_prod = Ener_Prod_Nat_2015.copy()
    elif year_prod == '2016' :     
        df_yr_prod = Ener_Prod_Nat_2016.copy()
    elif year_prod == '2017' :     
        df_yr_prod = Ener_Prod_Nat_2017.copy()
    elif year_prod == '2018' :     
        df_yr_prod = Ener_Prod_Nat_2018.copy()
    elif year_prod == '2019' :     
        df_yr_prod = Ener_Prod_Nat_2019.copy()
    elif year_prod == '2020' :     
        df_yr_prod = Ener_Prod_Nat_2020.copy()
    
    month2 = st.sidebar.slider("Choisir le mois", 1, 12)
    j_min2 = st.sidebar.slider("jour début période", 1, 31)
    j_max2 = st.sidebar.slider("jour fin période", 1, 31)


    # Affichage de la Production et de la consommation instantannée sur une période    
    if j_min2 >= j_max2 : 
        st.error("Mauvaise définition de la période de temps")
    else :
        fig = plt.figure(figsize=(14,18))
        df_yr_prod_M = df_yr_prod[df_yr_prod['Mois']==month2]
        df_yr_prod_week = df_yr_prod_M[(df_yr_prod_M['Jour']>=j_min2) & (df_yr_prod_M['Jour']<j_max2)]
        
        plt.subplot(311)
        sns.lineplot(data=df_yr_prod_week, x='Date - Heure', y='Prod_and_EchPhy', color='blue', ci=None, alpha=0.7)
        plt.ylabel("Puissance (MW)")
        plt.xlabel("Date")
        j_min2_str = str(j_min2)
        j_max2_str = str(j_max2)
        month2 = str(month2)
        str_title3 = "Production Nationale +ou- Echanges physiques instantannés du "+j_min2_str+" au "+j_max2_str+" du mois "+month2+" de l'année "+year_prod
        plt.title(str_title3)
        
        plt.subplot(312)
        sns.lineplot(data=df_yr_prod_week, x='Date - Heure', y='Consommation (MW)', color='red', ci=None, alpha=0.7)
        plt.ylabel("Puissance (MW)")
        plt.xlabel("Date")
        str_title4 = "Consommation Nationale instantannée du "+j_min2_str+" au "+j_max2_str+" du mois "+month2+" de l'année "+year_prod
        plt.title(str_title4)
        
        plt.subplot(313)
        sns.lineplot(data=df_yr_prod_week, x='Date - Heure', y='Diff_Prod_Conso', color='black', ci=None, alpha=0.7)
        plt.ylabel("Puissance (MW)")
        plt.xlabel("Date")
        str_title5 = "Différentiel entre la Production +- Echanges physiques et la Consommation du "+j_min2_str+" au "+j_max2_str+" du mois "+month2+" de l'année "+year_prod
        plt.title(str_title5)
        st.pyplot()
    
    
    st.write("Observations : On note des écarts maximum de quelques dizaines de MW en équilibrage sur l'ensemble du réseau national. \n"
             "Etant donné l'ordre de grandeur de la Production et de la Consommation, cela représente moins de 0,1% de variation, soit une synchronisation parfaite.  \n"
             "On observe donc un équilibrage très fin et totalement flexible du réseau électrique.")
    
    st.write("")
    st.write("")
    
    
    # Affichage de la Production par type de source sur une période
    st.subheader("Production par type de source sur la période choisie")
    if j_min2 >= j_max2 : 
        st.error("Mauvaise définition de la période de temps")
    else :
        df_yr_prod_week.plot(x='Date - Heure', 
                         y=['Thermique (MW)','Nucléaire (MW)','Eolien (MW)',
                            'Solaire (MW)','Hydraulique (MW)','Bioénergies (MW)','Pompage (MW)'], 
                             style=['r','k','m','y','b','g','c'], figsize=(14,8))
        plt.ylabel("Puissance (MW)")
        plt.xlabel("Date")
        str_title6 = "Production instantannée par type de source du "+j_min2_str+" au "+j_max2_str+" du mois "+month2+" de l'année "+year_prod
        plt.title(str_title6)
        st.pyplot()
    
    st.write("Observations : On note des productions avec des comportements et des utilités distinctes selon le type de source : \n"
             " - Les sources non contrôlables (Eolienne et Solaire) \n"
             " - Les sources contrôlables et peu flexibles (Nucléaire, Thermique et Bioénergies ) \n"
             " - Les sources contrôlables et capable de flexibilité extrême (Hydraulique et pompage STEP permettant de stocker l’énergie )")
 
    
    st.write("")
    st.write("")
    
    # Quantité d'energie produite par mois
    # Quantité d'energie produite par mois en GW.h
    
    st.sidebar.subheader("Observation Quantité d'Energie")
    year_choice_Q = st.sidebar.selectbox("Choisir une année",['année 2013','année 2014','année 2015','année 2016','année 2017','année 2018','année 2019','année 2020'])
    if year_choice_Q == 'année 2013' :
        df_yrQ = Ener_sorted_2013.copy()
        df_yr_prodb = Ener_Prod_Nat_2013.copy()
    elif year_choice_Q == 'année 2014' :     
        df_yrQ = Ener_sorted_2014.copy()
        df_yr_prodb = Ener_Prod_Nat_2014.copy()
    elif year_choice_Q == 'année 2015' :     
        df_yrQ = Ener_sorted_2015.copy()
        df_yr_prodb = Ener_Prod_Nat_2015.copy()
    elif year_choice_Q == 'année 2016' :     
        df_yrQ = Ener_sorted_2016.copy()
        df_yr_prodb = Ener_Prod_Nat_2016.copy()
    elif year_choice_Q == 'année 2017' :     
        df_yrQ = Ener_sorted_2017.copy()
        df_yr_prodb = Ener_Prod_Nat_2017.copy()
    elif year_choice_Q == 'année 2018' :     
        df_yrQ = Ener_sorted_2018.copy()
        df_yr_prodb = Ener_Prod_Nat_2018.copy()
    elif year_choice_Q == 'année 2019' :     
        df_yrQ = Ener_sorted_2019.copy()
        df_yr_prodb = Ener_Prod_Nat_2019.copy()
    elif year_choice_Q == 'année 2020' :     
        df_yrQ = Ener_sorted_2020.copy()
        df_yr_prodb = Ener_Prod_Nat_2020.copy()
    
    # Création dataframe Quantité d'énergie par mois
    QEner_Prod_Nat_yr_M = df_yr_prodb.groupby(['Mois']).sum()/2/1000
    QEner_Prod_Nat_yr_M = QEner_Prod_Nat_yr_M.reset_index()
    
    # Renommage en GW
    QEner_Prod_Nat_yr_M.rename(columns={"Production (MW)": "Production (GW.h)", 
                               "Thermique (MW)": "Thermique (GW.h)", 
                               "Nucléaire (MW)": "Nucléaire (GW.h)",
                               "Eolien (MW)": "Eolien (GW.h)",
                               "Solaire (MW)": "Solaire (GW.h)",
                               "Hydraulique (MW)": "Hydraulique (GW.h)",
                               "Pompage (MW)": "Pompage (GW.h)",
                               "Bioénergies (MW)": "Bioénergies (GW.h)"},
                               inplace=True)
    
    QEner_Prod_Nat_yr_M = QEner_Prod_Nat_yr_M.drop(['Jour',
                                                    'Année',
                                                    'Production (GW.h)',
                                                    'Diff_Prod_Conso',
                                                    'Pompage (GW.h)', 
                                                    'Consommation (MW)', 
                                                    'Prod_and_EchPhy', 
                                                    'Ech. physiques (MW)'], axis=1)
    
    # Utilisation de df.melt pour transposer les colonne de production en ligne
    QEner_Prod_Nat_yr_M_melted = QEner_Prod_Nat_yr_M.melt(id_vars="Mois",var_name="Source_Type", value_name="Q_Energy (GW.h)")
    
    # Affichage bargraphe
    fig = plt.figure(figsize=(14,8))
    ax = plt.subplot(111)
    sns.barplot(data=QEner_Prod_Nat_yr_M_melted, x="Mois", y="Q_Energy (GW.h)", hue="Source_Type")
    plt.ylabel("Quantité d'Energie (GW.h)")
    title_6 = "Quantité d'énergie produite par type de source par mois sur l'"+year_choice_Q
    plt.title(title_6)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    st.pyplot()
    
    st.write("Observations : On retrouve la prépondérance du Nucléaire sur toutes les autres énergies \n"
             "avec une production variant de 27000 à 40000 GW.h selon les besoins. Au global, on retrouve \n"
             "la saisonnalité pour répondre à la demande de consommation déjà observée précédemment.")
    
    
    st.write("")
    st.write("")
    
    # Affichage de la répartition sur l'année
    title_7 = "Répartition de la Quantité d'Energie produite par type de source sur l'"+year_choice_Q
    st.subheader(title_7)
    Q_Therm_yr = QEner_Prod_Nat_yr_M['Thermique (GW.h)'].sum()
    Q_Nuke_yr = QEner_Prod_Nat_yr_M['Nucléaire (GW.h)'].sum()
    Q_Eole_yr = QEner_Prod_Nat_yr_M['Eolien (GW.h)'].sum()
    Q_Sol_yr = QEner_Prod_Nat_yr_M['Solaire (GW.h)'].sum()
    Q_Hydro_yr = QEner_Prod_Nat_yr_M['Hydraulique (GW.h)'].sum()
    Q_BioEn_yr = QEner_Prod_Nat_yr_M['Bioénergies (GW.h)'].sum()
    Q_Total_yr = Q_Therm_yr + Q_Nuke_yr + Q_Eole_yr + Q_Sol_yr + Q_Hydro_yr + Q_BioEn_yr

    R_Th = Q_Therm_yr/Q_Total_yr
    R_Nu = Q_Nuke_yr/Q_Total_yr
    R_Eo = Q_Eole_yr/Q_Total_yr
    R_So = Q_Sol_yr/Q_Total_yr
    R_Hy = Q_Hydro_yr/Q_Total_yr
    R_Bi = Q_BioEn_yr/Q_Total_yr

    fig = plt.figure(figsize=(6,6))
    ax = plt.subplot(111)
    labels = 'Thermique', 'Nucléaire', 'Eolien', 'Solaire', 'Hydraulique', 'Bioénergies'
    sizes = [R_Th, R_Nu, R_Eo, R_So, R_Hy, R_Bi]
    explode = (0.1, 0.1, 0.1, 0.1, 0.1, 0.1)
    plt.pie(sizes, explode=explode, labels=labels, labeldistance=1.22, autopct='%1.1f%%', pctdistance=1.1, startangle=90)
    plt.axis('equal')
    st.pyplot()  
    
    
    
    st.write("")
    st.write("")
    
    # Production et Consommation par région sur une année
    st.subheader("Observation de la Production vs Consommation par Région sur une année au choix")    
    
    # liste des colonnes à supprimer
    Cln_QE_todel = df_yrQ.columns[17:]
    QEner_yr_temp = df_yrQ.copy()
    QEner_yr_temp.drop(Cln_QE_todel, axis=1, inplace=True)
    
    QEner_yr_som = QEner_yr_temp.groupby('Région').sum()/2/1000
    # Pour avoir la quantité en GigaWatt.heure par an : 
    # il faut intégrer la production instantée en sommant et divisant par 2 (mesure toute les 0.5 heure)
    # diviser par 1000 pour passer de MW;h en GW.h
    
    QEner_yr_som = QEner_yr_som.round(2)
    QEner_yr_som = QEner_yr_som.drop(["Jour", "Mois", "Année"],axis=1)
    
    QEner_yr_som.rename(columns={"Consommation (MW)": "Consommation (GW.h)", 
                                 "Production (MW)": "Production (GW.h)",
                                 "Ech. physiques (MW)": "Ech. physiques (GW.h)"}, inplace=True)
    
    fig = plt.figure()
    title_fig = "Consommation, Production et Echanges Inter-régionnaux "+year_choice_Q
    QEner_yr_som.plot.bar(y=['Consommation (GW.h)', 'Production (GW.h)', 'Ech. physiques (GW.h)'], 
                        ylabel = "Quantité d'énergie en GW.h", 
                        title = title_fig, 
                        color={"Consommation (GW.h)": "red", "Production (GW.h)": "blue", "Ech. physiques (GW.h)": "green"}, 
                        alpha=0.6, rot=70, figsize=(14,10))
    st.pyplot()
    
    st.subheader("Observations spécifiques pour l'année 2019")
    st.write("On identifie 3 grandes classes de Région en France :  \n"
             " - Les régions « consommatrices » produisant peu ou très peu comme Bourgogne-Franche-Comté, Bretagne, Pays de la Loire, Provence-Alpes-Côte d'Azur, et surtout Île-de-France. \n"
             " - Les régions « équilibrées » comme Hauts-de-France et Occitanie. \n"
             " - Les régions « Productrices » distribuant leurs excédents comme Auvergne-Rhône-Alpes, Centre-Val de Loire, Grand Est, Normandie et Nouvelle-Aquitaine.")
    
    
    
    st.write("")
    st.write("")
    
    # Affichage carte de production par source d'énergie en 2019
    st.subheader("Observation de la Production en 2019 par source d'énergie et par Région en TW.h")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image("Ener_Sol.png")
    with col2:
        st.image("Ener_Eol.png")
    
    with col1:
        st.image("Ener_Hydro.png")
    with col2:
        st.image("Ener_Bio.png")
    
    with col1:
        st.image("Ener_Ther.png")
    with col2:
        st.image("Ener_Nuke.png")
    
    
    st.write("Observations : \n"
             " - Hydraulique : 1ère source d’électricité produite à partir d’énergie renouvelable, \n"
             " - 3 grandes régions productrices d’énergies renouvelables : Auvergne-Rhône-Alpes, Grand-Est, Occitanie, \n"
             " - 60% du Nucléaire produit par 3 grandes régions : Auvergne Rhône-Alpes, le Centre-Val de Loire, Grand Est.")
    
 



# Définition Page Modèle SARIMA
elif choix == 'Modèle SARIMA':
    st.title("Application d'un modèle SARIMA")
    
    st.header("Pourquoi un SARIMA ?")
    st.write("Nous avons choisi de travailler sur un modèle de séries temporelles pour pouvoir analyser \n"
             "dans notre série la dépendance dans le temps et le comportement saisonnier qui peut apparaitre.  \n"
             "Cela fait suite aux éléments de datavisualisation que nous avons vu précédemment. \n"
             "Nous avons clairement constaté un effet saisonnier dans notre série.  \n"
             "  \n"
             "On peut grapher ci-dessous nos données brutes de consommation entre 2016 et 2021. \n"
             "Nous avons choisi de restreindre le jeu de données en partant de l’année 2016.  \n"
             "Plus précisément il s’agit, ici de la représentation de la consommation nationale maximale mensuelle \n"
             "en méga watt, plus pertinente que la consommation moyenne mensuelle si l’on souhaite établir une prédiction \n"
             "utile pour anticiper les besoins et la disponibilité des installations. \n"
             "  \n")
    
    #On se concentre sur les données à partir de 2016
    Ener_model=Ener_sorted[Ener_sorted['Année']>2015].copy()
    Ener_model=Ener_model[['Date','Consommation (MW)']]
    #on change le type de Date pour les séries temporelles
    Ener_model['Date'] = pd.to_datetime(Ener_model['Date'])
    #On resample pour avoir des données par MOIS sur la PUISSANCE max
    Ener_model=Ener_model.resample('MS', on='Date').max()
    #on transforme le df en "series" pandas pour appliquer un modèle de séries temporelles
    Ener_model_series = Ener_model['Consommation (MW)'].squeeze()
    
    import warnings
    import itertools
    import numpy as np
    warnings.filterwarnings("ignore")
    import pandas as pd
    import statsmodels.api as sm
    import matplotlib
    matplotlib.rcParams['axes.labelsize'] = 14
    matplotlib.rcParams['xtick.labelsize'] = 12
    matplotlib.rcParams['ytick.labelsize'] = 12
    
    #on graphe la série brute de notre jeu de données
    st.subheader("Etude de la série chronologique")
    Ener_model_series.plot(figsize=(19, 4))
    plt.xlabel('Date')
    plt.ylabel('Puissance(MW)')
    plt.title('Consommation nationale - Pmax mensuelle 2016-2021')
    st.pyplot()
    
    st.write("Observations :  \n"
             "En analysant le graphique, nous pouvons observer que la série chronologique présente une saisonnalité annuelle. \n"
             "Les pics s'observent en début et fin d'année (période hivernale).  \n"
             "Nous pouvons décomposer la série chronologique en trois composantes distinctes : la tendance, la saisonnalité \n"
             "et le bruit à l’aide de la commande « sm.tsa.seasonal_decompose » de la bibliothèque pylab.")
    
    st.write("")
    st.write("")
    
    # En utilisant la commande « sm.tsa.seasonal_decompose » de la bibliothèque pylab, 
    # nous pouvons décomposer la série chronologique en trois composants distincts : tendance, saisonnalité et bruit
    from pylab import rcParams
    #from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
    rcParams['figure.figsize'] = 18, 12
    decomposition = sm.tsa.seasonal_decompose(Ener_model_series, model='additive')
    fig = decomposition.plot()
    st.pyplot()
    
    st.write("Observations :  \n"
             "Ici, la décomposition est additive, en effet on n’observe pas de tendance claire et continue dans le temps, à la hausse ou à la baisse.  \n"
             "Aussi, on confirme la présence d’une saisonnalité annuelle.  \n"
             "Le choix du modèle SARIMA ( Seasonal Autoregressive Integrated Moving Average) permet de contrôler la saisonnalité \n"
             "en l’incluant directement en tant que caractéristique (variable X) dans notre modèle au lieu de transformer notre variable Y en changement annuel.")
    
    st.write("")

    
    st.subheader("Définition du modèle SARIMA")
    
    # Définition des p,d et q
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    
    #on simule ici différents modèles SARIMA, avec différents paramètres 
    #et on choisit la meilleure suivant le critère AIC en sortie
    
    st.write(" Nous utilisons l'implémentation de SARIMA par statsmodels et nous simulons différents modèles SARIMA avec \n"
             "différents paramètres afin de choisir la meilleur configuration suivant le critères AIC en sortie.")
    
    
    st.image("SARIMA_AIC.png")
    
    st.image("results_AIC.jpg")
    
    st.write("")
    
    st.write("Le critère AIC (critère d'information d'Akaike) est un estimateur de la qualité relative des modèles \n"
             "statistiques pour un ensemble de données. Le critère AIC estime la qualité de chaque modèle, par rapport \n"
             "à chacun des autres modèles. Plus la valeur de l'AIC est faible plus le modèle sera pertinent.  \n"
             "Notre sortie suggère que SARIMAX (0, 1, 1)x(0, 1, 1, 12) avec une valeur AIC de 604.79 est la meilleure \n"
             "combinaison, nous considérerons donc cette option comme optimale.")
    
    st.write("")
    
    with st.echo():
        # On définit le modèle SARIMAX (0,1, 1)x(0, 1, 1, 12) 
        mod = sm.tsa.SARIMAX(Ener_model_series,
                             order=(0, 1, 1),
                             seasonal_order=(0, 1, 1, 12),
                             enforce_stationarity=False,
                             enforce_invertibility=False)
        # On ajuste le modèle
        results = mod.fit()
        # On affiche les résultats du summary
        st.write(results.summary())
    
    st.write("")
    
    st.write("Tous les paramètres semblent être significatifs dans le modèle. Nous devons maintenant vérifier que le résidu \n"
             "est un bruit blanc et qu'il est distribué normalement grâce aux test statistiques qui nous indiquent les propriétés sur le résidu notamment.")
    
    st.write("")
    
    st.write("Les tests statistiques nous indiquent les propriétés des résidus :  \n"
             "  \n" 
             "Le test de Ljung-Box est un test de blancheur. C'est un test statistique qui vise à rejeter \n"
             "ou non l'hypothèse H0 : Le résidu est un bruit blanc. Ici on lit sur la ligne Prob(Q) que la p-valeur \n"
             "de ce test est de 1, donc on ne rejette pas l'hypothèse.  \n"
             "  \n"
             "Le test de Jarque-Bera est un test de normalité. C'est un test statistique qui vise à rejeter \n"
             "ou non l'hypothèse H0 : Le résidu suit une distribution normale. Ici on lit sur la ligne Prob (JB) \n"
             "que la p-valeur du test est de 0.89. On ne rejette donc pas l'hypothèse.  \n"
             "  \n"
             "Le résidu vérifie les hypothèses que l'on a faites à priori. On peut donc conclure que le modèle SARIMA(0,1,1)(0,1,1)12 est satisfaisant.")
    
    st.write("")
    
    st.write("Nous allons maintenant utiliser ce modèle pour faire une prédiction sur la consommation d’électricité. \n"
             "Les prédictions s'effectuent à l'aide de la méthode predict appliquée à un modèle ajusté.  \n"
             "Cette étape consiste à comparer les valeurs réelles avec les prédictions prévues.")
    
    st.write("")
    
    st.subheader("Prédiction sur la Consommation d'électricité à l'aide du modèle SARIMA")
    
    pred = results.get_prediction(start=pd.to_datetime('2019-01-01'), dynamic=False)
    pred_ci = pred.conf_int()
    ax = Ener_model_series['2019':].plot(label='observed')
    pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(18, 6))
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='g', alpha=.2)
    ax.set_xlabel('Date')
    ax.set_ylabel('Consommation (MW)')
    plt.title('Prédiction de la Consommation nationale à partir de Janvier 2019 (Puissance maximale mensuelle)')
    plt.legend()
    st.pyplot()
    
    st.write("")
    
    st.markdown("__Conclusion__")
    st.write("En prenant en compte l'intervalle de confiance représentée en vert clair, nous pouvons \n"
             "considérer que les prédictions sont globalement fiables.  \n"
             "Le modèle peut être considéré comme performant.")
    


# Définition Page Modèles de Régression
elif choix == 'Modèles de Régression':
    
    import numpy as np
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import  LassoCV, RidgeCV, ElasticNetCV
    from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
        
    st.title("Application de modèles de Régression régularisée")
    
    st.header("Pourquoi une Régression régularisée ?")
    st.write("L’approche par Régression nous semble être une deuxième approche possible au regard \n"
             "des data que nous avons pu récupérer en jeux de données complémentaires.  \n"
             "En effet, nous avons pu voir l’impact de la température sur des variations de consommation.  \n"
             "Nous supposons également que les données de population influencent la consommation dans la \n"
             "mesure ou l’électricité est consommé pour des besoins humains.  \n"
             "A cela s’ajoute enfin le profil économique d’une région qui peut se traduire par la représentation \n"
             "des établissements d’entreprises en fonction de leur taille et leur secteur d’activité dans le sens \n"
             "où l’activité économique est très probablement liée à la consommation d’électricité.")
    
    st.write("")
    
    st.subheader("Descriptif du jeux de données fusionnés")
    st.write(data.head(12))
    
    st.write(data.tail(12))
    
    st.write("")
    
    st.write("Nous disposons d'un enregistrement par jour et par Région du 01/01/2016 au 31/12/2018  \n"
             "Explication des variables : \n"
             " - Consommation : valeur max quotidienne de consommation électrique pour une Région \n"
             " - Tmoy (°C) : température moyenne sur la Région au jour indiqué \n"
             " - Population : nombre d'habitants estimés par Région (chaque année) \n"
             " - Démographie des Entreprises dans chaque Région par secteur (Primaire, Secondaire, Tertiaire) et par taille (Micro, PME, ETI GE)")
    
    st.write("")
    st.write("Nous cherchons à prédire la Consommation, ce sera donc notre variable cible, les autres variables seront considérées comme features.")
    
    st.write("")
    
    st.markdown("__Visualisation de la distribution de la variable cible__")
    # distribution de la variable cible "Consommation"
    plt.figure(figsize=(8,8))
    sns.displot(data['Consommation (MW)'])
    st.pyplot()
    
    st.write("")
    
    st.markdown("__Analyse préliminaire des corrélations par heatmap__")
    # Analyse préliminaire par heatmap pour observer les correlations entre chaque colonne de data
    plt.figure(figsize=(16,15))
    sns.heatmap(data.corr(), annot=True, cmap='RdBu_r', center=0)
    st.pyplot()
    
    st.write("")
    
    st.write("On note que la variable Sect_Prim_GE est vide et qu'elle peut par conséquent être supprimée.  \n"
             "En effet, il n'existe pas d'entreprises de 5000 salariés et plus dans le secteur primaire représentant \n"
             "l'agriculture, la sylviculture et la pêche.  \n"
             "  \n"
             "Par ailleurs, on observe des corrélations importantes avec la variable cible concernant la Population, \n"
             "le secteur secondaire et le secteur tertiaire.  \n"
             "On retrouve également une corrélation inverse sur la Température moyenne qui est logique dans la mesure \n"
             "où plus il fait froid et plus la consommation électrique augmente pour le chauffage.")
    
    st.write("")
    st.write("")
    
    st.write("Nous allons donc tester plusieurs modèles de Régression régularisée : \n"
             " - un modèle RIDGE \n"
             " - un modèle LASSO \n"
             " - un modèle Elastic Net")
    
    st.write("")
    st.write("")
    
    # Finalisation jeux de données

    data_4Reg = data.copy()
    data_4Reg.index = data['Date'] + ' - ' + data['Région']
    data_4Reg.drop(['Date', 'Région', 'Sect_Prim_GE'], axis=1, inplace=True)
    
    
    st.markdown("____")
        
    st.subheader("Le modèle Ridge")
    
    st.markdown("__Pourquoi tester un Ridge ?__")
    
    st.write("La régression Ridge est une méthode d'ajustement de modèle qui est utilisée pour analyser \n"
             "les données qui souffrent de multicollinéarité, corrélations entre les variables prédictives.  \n"
             "Cette méthode effectue une régularisation L2, norme euclidienne. Lorsque le problème de la \n"
             "multicollinéarité se pose, les moindres carrés ne sont pas biaisés et les variances sont importantes, \n"
             "ce qui fait que les valeurs prédites sont très éloignées des valeurs réelles.  \n"
             "C’est donc un moyen de créer un modèle parcimonieux en instaurant un partage des poids entre variables.")
    
    st.write("")
    
    st.write("__Etapes :__ \n"
             " - la sélection des features et de la target \n"
             " - la normalisation des variables avec un Standard scaler \n"
             " - le split des échantillons \n"
             " - la création du modèle Ridge avec Cross Validation \n"
             " - l'entrainement du modèle")
    
    st.write("")
    
    with st.echo():       
        # Selection des features et de la target
        features1 = data_4Reg.drop(['Consommation (MW)'], axis=1)
        target1 = pd.DataFrame(np.c_[data_4Reg['Consommation (MW)']], columns = ['Consommation (MW)'], index=data_4Reg.index)
    
        # Normalisation
        scaler1 = preprocessing.StandardScaler()
        features1[features1.columns] = pd.DataFrame(scaler1.fit_transform(features1), index=features1.index)
        target1_scaled = scaler1.fit_transform(target1)
    
        # Séparation des échantillons train et test avec option Shuffle=False
        X_train1, X_test1, y_train1, y_test1 = train_test_split(features1, target1_scaled, 
                                                            test_size=0.2, random_state=44,
                                                            shuffle=False)
    
        # Création du modèle Ridge
        model_Ridge = RidgeCV(alphas= (0.001, 0.01, 0.1, 0.3, 0.7, 1, 10, 50, 100))
    
        # Entrainement du modèle
        model_Ridge.fit(X_train1, y_train1)
    
    st.write("")
    st.write("")
    
    st.write("__Affichage du coefficient alpha sélectionné par le modèle__")
    st.write("Coefficient alpha sélectionné par la Cross Validation du modèle : ", model_Ridge.alpha_)
    
    st.write("")
    
    st.subheader("Evaluation du modèle avec les métriques")
    
    # predictions du modèle Ridge
    pred_train1 = model_Ridge.predict(X_train1)
    pred_test1 = model_Ridge.predict(X_test1)
    
    # MSE (Mean Squarred Error) pour train et test
    st.write("MSE train modèle Ridge : ",mean_squared_error(y_train1, pred_train1).round(5))
    st.write("MSE test modèle Ridge : ",mean_squared_error(y_test1, pred_test1).round(5))
    st.write("La  MSE est excellente sur les 2 échantillons (valeur à minimiser).")
    
    st.write("")
    
    # score R² (coefficient de détermination) pour les 2 échantillons
    st.write("R² score train modèle Ridge : ",r2_score(y_train1, pred_train1).round(5))
    st.write("R² score test modèle Ridge : ",r2_score(y_test1, pred_test1).round(5))
    st.write("Le score R² est très bon sur les 2 échantillons (valeur à maximiser , entre 0 et 1).")
    
    st.write("")
    
    # MAPE (Mean Absolute Percentage Error) pour les 2 échantillons
    st.write("MAPE train modèle Ridge : ", mean_absolute_percentage_error(y_train1, pred_train1).round(5))
    st.write("MAPE test modèle Ridge : ", mean_absolute_percentage_error(y_test1, pred_test1).round(5))
    st.write("La MAPE est très bonne sur les 2 échantillons (valeur à minimiser). \n")
    
    st.write("")
    
    # reshape des predictions et valeurs réelles
    pred_test1_resh = np.reshape(pred_test1, 2631)
    y_test1_resh = np.reshape(y_test1, 2631)  
    
    
    st.write("")
    
    st.subheader("Evaluation du modèle Ridge par contrôle des Consommation réelles vs Consommation prédites sur l'échantillon de test")
    
    # création des variables moyenne et ecart-type
    st.write("Création des variables moyenne et écart type à partir du scaler")
    moy1 = scaler1.mean_[-1]
    ec1 = scaler1.scale_[-1]
    st.write("moyenne :", moy1.round(3))
    st.write("ecart-type :", ec1.round(3))
    
    st.write("")
    
    # Affichage des consommation observées et des consommations prédites par le modèle Elastic Net.
    df_results1 = pd.DataFrame({'Consommations_observées_(MW)': (y_test1_resh*ec1)+moy1, 
                               'Consommations_prédites_(MW)' : (pred_test1_resh*ec1)+moy1},
                               index=X_test1.index)
    
    st.write(df_results1.head(50))
    
    st.write("")
    
    st.write("On observe des écarts importants entre les valeurs réelles et les valeurs prédites (ex : 3eme lign Ile-de-France)  \n"
             "Il semble que le modèle ne soit pas aussi performant qu'observé avec les métriques.")
    
     
    
    st.write("")
    st.write("")    
    
    st.markdown("____")
    
    st.subheader("Le modèle Lasso")
    
    st.markdown("__Pourquoi tester un Lasso ?__")
    
    st.write("La régression Lasso est une technique de régularisation utilisée pour une prédiction \n"
             "plus précise en appliquant le principe de rétrécissement. \n"
             "Cela consiste à réduire les valeurs des données vers un point central comme la moyenne.  \n"
             "Ce type particulier de régression est bien adapté aux modèles présentant des niveaux élevés de multicollinéarité \n"
             "ou lorsque l’on souhaite automatiser certaines parties de la sélection du modèle, comme la sélection des variables/élimination des paramètres.  \n"
             "La régression Lasso utilise la technique de régularisation L1.  \n"
             "Elle est utilisée lorsque nous avons un plus grand nombre de variables car elle effectue automatiquement la sélection des variables.")
    
    st.write("")
    
    st.write("__Etapes :__ \n"
             " - la sélection des features et de la target \n"
             " - la normalisation des variables avec un Standard scaler \n"
             " - le split des échantillons \n"
             " - la création du modèle Lasso avec Cross Validation \n"
             " - l'entrainement du modèle")
    
    st.write("")
    
    with st.echo():       
        # Selection des features et de la target
        features2 = data_4Reg.drop(['Consommation (MW)'], axis=1)
        target2 = pd.DataFrame(np.c_[data_4Reg['Consommation (MW)']], columns = ['Consommation (MW)'], index=data_4Reg.index)
    
        # Normalisation
        scaler2 = preprocessing.StandardScaler()
        features2[features2.columns] = pd.DataFrame(scaler2.fit_transform(features2), index=features2.index)
        target2_scaled = scaler2.fit_transform(target2)
    
        # Séparation des échantillons train et test avec option Shuffle=False
        X_train2, X_test2, y_train2, y_test2 = train_test_split(features2, target1_scaled, 
                                                            test_size=0.2, random_state=44,
                                                            shuffle=False)
    
        # Création du modèle Lasso
        model_Lasso = LassoCV(cv=10)
    
        # Entrainement du modèle
        model_Lasso.fit(X_train2, y_train2)
    
    st.write("")
    st.write("")
    
    st.write("__Affichage du coefficient alpha sélectionné par le modèle__")
    st.write("Coefficient alpha sélectionné par la Cross Validation du modèle : ", model_Lasso.alpha_.round(5))
    
    st.write("")
    
    st.subheader("Evaluation du modèle avec les métriques")
    
    # predictions du modèle Ridge
    pred_train2 = model_Lasso.predict(X_train2)
    pred_test2 = model_Lasso.predict(X_test2)
    
    # MSE (Mean Squarred Error) pour train et test
    st.write("MSE train modèle Lasso : ",mean_squared_error(y_train2, pred_train2).round(5))
    st.write("MSE test modèle Lasso : ",mean_squared_error(y_test2, pred_test2).round(5))
    st.write("La  MSE est excellente sur les 2 échantillons (valeur à minimiser).")
    
    st.write("")
    
    # score R² (coefficient de détermination) pour les 2 échantillons
    st.write("R² score train modèle Lasso : ",r2_score(y_train2, pred_train2).round(5))
    st.write("R² score test modèle Lasso : ",r2_score(y_test2, pred_test2).round(5))
    st.write("Le score R² est très bon sur les 2 échantillons (valeur à maximiser , entre 0 et 1).")
    
    st.write("")
    
    # MAPE (Mean Absolute Percentage Error) pour les 2 échantillons
    st.write("MAPE train modèle Lasso : ", mean_absolute_percentage_error(y_train2, pred_train2).round(5))
    st.write("MAPE test modèle Lasso : ", mean_absolute_percentage_error(y_test2, pred_test2).round(5))
    st.write("La MAPE est très bonne sur les 2 échantillons (valeur à minimiser). \n")
    
    st.write("")
    
    # reshape des predictions et valeurs réelles
    pred_test2_resh = np.reshape(pred_test2, 2631)
    y_test2_resh = np.reshape(y_test2, 2631)  
    
    
    st.write("")
    
    st.subheader("Evaluation du modèle Lasso par contrôle des Consommation réelles vs Consommation prédites sur l'échantillon de test")
    
    # création des variables moyenne et ecart-type
    st.write("Création des variables moyenne et écart type à partir du scaler")
    moy2 = scaler2.mean_[-1]
    ec2 = scaler2.scale_[-1]
    st.write("moyenne :", moy2.round(3))
    st.write("ecart-type :", ec2.round(3))
    
    st.write("")
    
    # Affichage des consommation observées et des consommations prédites par le modèle Elastic Net.
    df_results2 = pd.DataFrame({'Consommations_observées_(MW)': (y_test2_resh*ec2)+moy2, 
                               'Consommations_prédites_(MW)' : (pred_test2_resh*ec2)+moy2},
                               index=X_test2.index)
    
    st.write(df_results2.head(50))
    
    st.write("")
    
    st.write("Comme pour le modèle Ridge, On observe des écarts importants entre les valeurs réelles et les valeurs prédites.  \n"
             "Il semble que le modèle ne soit pas aussi performant qu'observé avec les métriques.")
    
    
    st.write("")
    st.write("")
    
    st.markdown("____")
    
    
    st.subheader("Le modèle Elastic Net")
    
    st.markdown("__Pourquoi tester un Elastic Net ?__")
    st.write("Dans les paragraphes ci-dessus, nous avons vu que la régression Ridge utilise la pénalité L2 et la régression Lasso utilise la pénalité L1.  \n"
             "Le modèle ElasticNet offre pour sa part les avantages d'une combinaison linéaire des pénalités L1 et L2.")
    
    st.write("")
    
    st.write("__Etapes :__ \n"
             " - la normalisation du dataframe avec un MinMax scaler \n"
             " - la sélection des features et de la target \n"
             " - le split des échantillons \n"
             " - la création du modèle Elastic Net avec Cross Validation \n"
             " - l'entrainement du modèle")
    
    st.write("")
    
    with st.echo():
        # Normalisation avec le scaler MinMax 
        # pas de loi normale sur les features en dehors d'une distribution bimodale sur Tmoy
        scaler3 = preprocessing.MinMaxScaler()
        data_scaled3 = pd.DataFrame(scaler3.fit_transform(data_4Reg), index=data_4Reg.index, columns=data_4Reg.columns)
    
        # Selection des features et de la target
        features3 = data_scaled3.drop(['Consommation (MW)'], axis=1)
        target3 = data_scaled3['Consommation (MW)']
    
        # Séparation des échantillons train et test avec option Shuffle=False
        X_train3, X_test3, y_train3, y_test3 = train_test_split(features3, target3, 
                                                    test_size=0.2, random_state=44, 
                                                    shuffle=False)
        
        # Création du modèle ElasticNet
        model_ElNet = ElasticNetCV(cv=10, l1_ratio=(0.1, 0.25, 0.5, 0.7, 0.75, 0.8, 0.85, 0.9, 0.99), 
                                   alphas=(0.001, 0.01, 0.02, 0.025, 0.05, 0.1, 0.25, 0.5, 0.8, 1.0),
                                   max_iter=1000000000)
    
        # Entrainement du modèle
        model_ElNet.fit(X_train3, y_train3)
    
    st.write("")
    st.write("")
    
    st.write("__Affichage de l'intercept et des coefficients estimés pour chaque variable__")
    
    st.write("")
    
    # Affichage de l'intercept et des coeff estimés pour chaque variable
    coeffs3 = list(model_ElNet.coef_)
    coeffs3.insert(0, model_ElNet.intercept_)
    feats3 = list(features3.columns)
    feats3.insert(0, 'intercept')

    df_coeffs = pd.DataFrame({'valeur estimée': coeffs3}, index=feats3)
    st.write(df_coeffs)
    
    st.write("Observations : la population, la température et les ETI du secteur tertiaire influencent le plus le modèle au regard des coefficients renvoyés.")
    
    st.write("")
    
    st.write("__Affichage du coefficient alpha sélectionné par le modèle__")
    
    #alpha3 = model_ElNet.alpha_
    #alpha3 = str(alpha3)
    #str_alpha = "Coefficient alpha sélectionné par la Cross Validation du modèle : "+alpha3
    st.write("Coefficient alpha sélectionné par la Cross Validation du modèle : ", model_ElNet.alpha_)
    
    # format Dataframe
    y_train3 = pd.DataFrame(y_train3)
    y_test3 = pd.DataFrame(y_test3)
    
    # predictions et format dataframe pour les preds
    pred_train3 = model_ElNet.predict(X_train3)
    pred_test3 = model_ElNet.predict(X_test3)

    pred_train3 = pd.DataFrame(pred_train3, columns = y_train3.columns, index=y_train3.index)
    pred_test3 = pd.DataFrame(pred_test3, columns = y_test3.columns, index=y_test3.index)
    
    st.write("")
    
    st.subheader("Evaluation du modèle avec les métriques")
    
    # MSE (Mean Squarred Error) pour train et test
    st.write("MSE train modèle Elastic Net: ",mean_squared_error(y_train3, pred_train3).round(5))
    st.write("MSE test modèle Elastic Net : ",mean_squared_error(y_test3, pred_test3).round(5))
    st.write("La  MSE est excellente sur les 2 échantillons (valeur à minimiser).")
    
    st.write("")
    
    # score R² (coefficient de détermination) pour les 2 échantillons
    st.write("R² score train modèle Elastic Net : ",r2_score(y_train3, pred_train3).round(5))
    st.write("R² score test modèle Elastic Net : ",r2_score(y_test3, pred_test3).round(5))
    st.write("Le score R² est très bon sur les 2 échantillons (valeur à maximiser , entre 0 et 1).")
    
    st.write("")
    
    # MAPE (Mean Absolute Percentage Error) pour les 2 échantillons
    st.write("MAPE train modèle Elastic Net : ", mean_absolute_percentage_error(y_train3, pred_train3).round(5))
    st.write("MAPE test modèle Elastic Net : ", mean_absolute_percentage_error(y_test3, pred_test3).round(5))
    st.write("la  MAPE est anormalement haute sur l'échantillon train et très bonne sur l'échantillon de test (valeur à minimiser).  \n"
             "L'anomalie sur la MAPE train s'explique par le mode de calul et la présence d'un 0 dans l'échantillon train.")
    
    
    # création des variables min et max
    min_ElNet = scaler3.data_min_[0]
    max_ElNet = scaler3.data_max_[0]
    
    st.write("")
    st.write("")
    
    st.subheader("Evaluation du modèle Elastic Net par contrôle des Consommation réelles vs Consommation prédites sur l'échantillon de test")
    
    # Affichage des consommation observées et des consommations prédites par le modèle Elastic Net.
    df_results3 = pd.DataFrame({'Consommations_observées_(MW)': (y_test3['Consommation (MW)']*(max_ElNet-min_ElNet))+min_ElNet, 
                               'Consommations_prédites_(MW)' : (pred_test3['Consommation (MW)']*(max_ElNet-min_ElNet))+min_ElNet},
                               index=y_test3.index)
    
    st.write(df_results3.head(50))
    
    st.write("")
    
    st.write("Comme pour le Ridge et le Lasso, on observe des écarts importants entre les valeurs réelles et les valeurs prédites  \n"
             "Il semble que le modèle ne soit pas aussi performant qu'observé avec les métriques.")
    
    st.write("")
    
    df_results3["pct_error"] = (df_results3["Consommations_prédites_(MW)"] - df_results3["Consommations_observées_(MW)"]) / df_results3["Consommations_observées_(MW)"] * 100
    df_results3.insert(0, "Région", df_results3.index)
    df_results3["Région"] = df_results3["Région"].str[13:]
    df_results3.insert(1, "Date", df_results3.index)
    df_results3["Date"] = df_results3["Date"].str[0:11]

    df_results3 = df_results3.reset_index()
    df_results3 = df_results3.drop(['index'], axis=1)
    
    
    st.subheader("Analyse statistique des erreurs de prédiction")
    # Observation des satistiques d'erreur par Région sur l'échantillon de test
    p = sns.catplot(x='Région', y='pct_error', kind='violin',inner=None, data=df_results3, height=8, aspect=12/8)
    sns.swarmplot(x='Région', y='pct_error', size=3, color='black', alpha=0.7, data=df_results3, ax=p.ax)
    plt.xticks(rotation=70)
    plt.ylabel("Erreur de prédiction en %")
    plt.ylim(-80, 100)
    plt.title("Analyse statistique de l'erreur de prédiction en fonction de la Région")
    st.pyplot()
    
    st.write("Cette analyse statistique laisse apparaitre plusieurs profils de régions : \n"
             " - Des régions avec une distribution « compactes » telles que l’Auvergne-Rhône-Alpes ou les Hauts-de-France. \n"
             " - Des régions avec une distribution « étendue » comme la Bourgogne-Franche-Comté ou le Centre-Val de Loire. \n"
             " - Des régions avec une distribution « intermédiaire » comme la Normandie ou l’Île-de-France.  \n"
             "De plus, on observe que quelque soit le profil, les erreurs de prédiction sont nombreuses et loin d’être faibles (supérieur à 5%).")
    
    st.write("")
    
    
    st.subheader("Observation de l'erreur de prédiction sur 2 Régions")
    
    # préparation dataframe et affichage région Bourgogne-Franche-Comté
    df_results3_BFC = df_results3[df_results3['Région'] == "Bourgogne-Franche-Comté"]
    df_results3_BFC_chrono = df_results3_BFC.sort_values(by = ['Date'], ascending = True)
    
    df_results3_BFC_chrono.plot(x='Date', y=['Consommations_observées_(MW)', 'Consommations_prédites_(MW)'],
                                style = ["b-d", "g-h"], 
                                title = "Consommation observé vs prédite - Région Bourgogne-Franche-Comté", figsize = (16,10))
    st.pyplot()
    
    
    st.write("")
    
    # préparation dataframe et affichage région Hauts-de-France
    df_results3_HDF = df_results3[df_results3['Région'] == "Hauts-de-France"]
    df_results3_HDF_chrono = df_results3_HDF.sort_values(by = ['Date'], ascending = True)
    
    df_results3_HDF_chrono.plot(x='Date', y=['Consommations_observées_(MW)', 'Consommations_prédites_(MW)'],
                                style = ["b-d", "g-h"], 
                                title = "Consommation observé vs prédite - Région Hauts-de-France", figsize = (16,10))
    st.pyplot()
    
    st.write("On note que quelque soit le profil, les prédictions ne sont pas bonnes avec des écarts importants entre les prédictions et le réel.")
    
    st.write("")
    st.write("")
    st.write("")
    
    st.subheader("Conclusion sur les modèles de Régression régularisée")
    st.write("")
    st.write("Malgré des métriques exceptionnellement bonnes et qui laissent penser que les modèles sont performants, \n"
             "il apparait que les prédictions sont totalement en écart avec les valeurs observées.  \n"
             "Cette divergence entre métriques et performance pose une problématique importante sur laquelle nous ne pouvons émettre que des hypothèses.  \n")
    
    st.write("Hypothèse 1 : le type de modèle serait inadapté à la relation entre la variable cible et les variables explicatives. Il nous faudrait donc choisir d’autres modèles à tester.  \n"
             "  \n"
             "Hypothèse 2 : les variables explicatives utilisées pour les modèles sont insuffisamment représentatives ou portent des biais.  \n")
    
    st.write("La divergence observée entre les métriques et les résultats de prédiction démontre qu'il ne faut jamais partir du principe \n"
             "que les métriques sont suffisantes à qualifier un modèle mais qu'une vérification par comparaison prédiction vs observée est \n"
             "absolument nécessaire afin de valider le modèle.")
    


