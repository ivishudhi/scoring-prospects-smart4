# Scoring Prospects Smart4 Engineering

Ce projet porte sur un outil de scoring commercial basé sur l'historique des missions Smart4 Engineering (plateforme Whoz). 
Le modèle estime la probabilité de succès d'un prospect à partir de ses caractéristiques (entité, délais, priorité, contexte de l'entité).

J'ai réalisé ce projet dans le cadre de mon mémoire de fin d'études M2 Data & IA à Nexa Digital School, en alternance en tant que BI Analyst chez Smart4 Engineering.

**Application déployée** : https://smart4-scoring-prospects.streamlit.app/

## Confidentialité

Les données utilisées pour ce projet appartiennent à Smart4 Engineering et sont extraites de sa base Whoz (Azure SQL Data Lake). Dans le cadre de ce mémoire, l'entreprise n'autorise pas la diffusion de cette base, des données brutes ou traitées, ni du code d'extraction (requêtes SQL et accès au Data Lake).

Ce dépôt contient donc uniquement :
- le code de modélisation (entraînement, comparaison des modèles, évaluation)
- le code de l'application Streamlit

La partie extraction (connexion Azure SQL, requêtes de jointure et d'agrégation) n'est pas incluse. 
Les dossiers de données ne sont pas fournis non plus. 
Pour rappel, dès l'extraction, les identifiants sensibles (mission, workspace) sont systématiquement hashés en SHA-256, et aucune donnée nominative (nom de consultant, nom de client) n'entre dans le pipeline.

## Structure du dépôt

├── app.py application Streamlit
├── scoring.py fonctions de scoring et de calcul SHAP
├── model_scoring_prospects.joblib modèle entraîné
├── mappings.json mappings des variables catégorielles (entités, priorité...)
├── win_ratio_par_entite.csv taux de succès historique par entité
├── style.css style et esthétique de l'application
├── requirements.txt
└── README.md

## Pipeline

Le code d'extraction (SQL, connexion Azure) n'étant pas inclus dans ce dépôt, le pipeline présenté ici démarre à partir du jeu de données déjà extrait :
1. Nettoyage : exclusion des missions issues d'imports en masse (biais de survie)
2. Feature engineering : écarts de dates, priorité encodée, agrégats de certifications par entité
3. Entraînement et comparaison de plusieurs modèles (régression logistique, arbre de décision, random forest, XGBoost, SVM)
Modèle retenu : Random Forest (AUC-ROC 0.9614, F1=0.84).

## Application

Application Streamlit permettant de saisir les caractéristiques d'un prospect et d'obtenir une probabilité de succès, une zone de score, et une explication SHAP.

## Installation

conda create -n scoring-app python=3.10
conda activate scoring-app
pip install -r requirements.txt
streamlit run app.py
