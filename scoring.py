# scoring.py : calcul du score de prospects

from pathlib import Path
import json
import joblib
import pandas as pd
import shap
import streamlit as st

DOSSIER = Path(__file__).resolve().parent
MODEL_PATH = DOSSIER / "model_scoring_prospects.joblib"
MAPPINGS_PATH = DOSSIER / "mappings.json"
WIN_RATIO_PATH = DOSSIER / "win_ratio_par_entite.csv"

MARGE = 0.15
MIN_MISSIONS = 30

LABELS = {
    "duree_mission": "Duree de la mission",
    "priority_ordinal": "Priorite",
    "ecart_creation_demarrage": "Ecart creation / demarrage",
    "entityTrig_code": "Entite",
    "nb_consultants_filiale": "Taille de l'entite",
    "exp_moyenne_filiale": "Experience moyenne de l'entite",
    "nb_certifications_total": "Nombre total de certifications (entite)",
    "nb_certifications_actives": "Certifications actives (entite)",
    "nb_consultants_certifies": "Consultants certifies (entite)",
    "nb_types_certifications_distincts": "Diversite des certifications (entite)",
}


def verifier(chemin):
    # on verifie que le fichier existe avant de le charger
    if not chemin.exists():
        raise FileNotFoundError(f"Fichier introuvable : {chemin.name}")

@st.cache_resource(show_spinner="Chargement du modele...")
def charger_ressources():
    # on charge le modele et les infos utiles une seule fois
    verifier(MODEL_PATH)
    verifier(MAPPINGS_PATH)
    model = joblib.load(MODEL_PATH)
    with open(MAPPINGS_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
    explainer = shap.TreeExplainer(model.named_steps["clf"])
    return {
        "model": model,
        "priority_map": config["priority_map"],
        "entite_map": config["entityTrig_map"],
        "stats_entite": config["filiale_stats"],
        "features": config["features"],
        "seuil": config.get("seuil_optimal", 0.5),
        "explainer": explainer,
    }

@st.cache_data(show_spinner=False)
def charger_win_ratio(min_missions=MIN_MISSIONS):
    # on charge le taux de reussite par entite, on enleve les petites entites
    verifier(WIN_RATIO_PATH)
    df = pd.read_csv(WIN_RATIO_PATH)
    return df[df["nb_missions"] >= min_missions].copy()

def construire_ligne(ressources, priorite, entite, duree, delai):
    # on construit une ligne avec les memes colonnes que celles de l'entrainement
    code_entite = ressources["entite_map"][entite]
    stats = ressources["stats_entite"][str(code_entite)]
    ligne = pd.DataFrame([{
        "duree_mission": duree,
        "priority_ordinal": ressources["priority_map"][priorite],
        "ecart_creation_demarrage": abs(delai),
        "entityTrig_code": code_entite,
        "nb_consultants_filiale": stats["nb_consultants_filiale"],
        "exp_moyenne_filiale": stats["exp_moyenne_filiale"],
        "nb_certifications_total": stats["nb_certifications_total"],
        "nb_certifications_actives": stats["nb_certifications_actives"],
        "nb_consultants_certifies": stats["nb_consultants_certifies"],
        "nb_types_certifications_distincts": stats["nb_types_certifications_distincts"],
    }])
    return ligne[ressources["features"]]

def classer(proba, seuil):
    # on classe le prospect en 3 niveaux selon le seuil et la marge
    if proba >= seuil + MARGE:
        niveau = "Fort potentiel"
        conseil = "Ce prospect a de bonnes chances d'aboutir : à prioriser."
    elif proba >= seuil - MARGE:
        niveau = "À confirmer"
        conseil = "Ce prospect est dans une zone intermédiaire : à qualifier davantage."
    else:
        niveau = "Risque élevé"
        conseil = "Ce prospect semble moins prometteur en l'état : à surveiller."
    return {"proba": proba, "niveau": niveau, "conseil": conseil}

def scorer(ressources, priorite, entite, duree, delai, seuil):
    # on calcule la proba de succes puis on classe le prospect
    ligne = construire_ligne(ressources, priorite, entite, duree, delai)
    proba = float(ressources["model"].predict_proba(ligne)[0, 1])
    return classer(proba, seuil), ligne

def calculer_contrib(ressources, ligne):
    # on calcule les valeurs SHAP pour expliquer le score obtenu
    ligne_scaled = ressources["model"].named_steps["scaler"].transform(ligne)
    shap_vals = ressources["explainer"].shap_values(ligne_scaled)
    if isinstance(shap_vals, list):
        valeurs = shap_vals[1][0]
    else:
        valeurs = shap_vals[0, :, 1]
    df = pd.DataFrame({"feature": ressources["features"], "valeur": valeurs})
    df["label"] = df["feature"].map(lambda f: LABELS.get(f, f))
    df["valeur_abs"] = df["valeur"].abs()
    df["favorise"] = df["valeur"] > 0
    return df.sort_values("valeur_abs", ascending=False)
