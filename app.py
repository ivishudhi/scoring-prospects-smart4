import streamlit as st
import joblib
import json
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

#configuration de la page
st.set_page_config(page_title="Scoring de prospects", layout="centered")
#chargement du modele entraine et des infos necessaires pour preparer les donnees
model = joblib.load("model_scoring_prospects.joblib")
with open("mappings.json", "r", encoding="utf-8") as f:
    config = json.load(f)
priority_map = config["priority_map"]
entityTrig_map = config["entityTrig_map"]
entite_stats = config["filiale_stats"]
features = config["features"]
#seuil optimal calculé dans le notebook (Youden sur la courbe ROC)
#si le notebook n'a pas encore été rejoué avec la nouvelle cellule, on retombe sur 0.5
SEUIL_OPTIMAL = config.get("seuil_optimal", 0.5)
#labels lisibles pour afficher les facteurs explicatifs (SHAP) en francais
LABELS_FEATURES = {
    "duree_mission": "Duree de la mission",
    "priority_ordinal": "Priorite",
    "ecart_creation_demarrage": "Ecart creation/demarrage",
    "entityTrig_code": "Entite",
    "nb_consultants_filiale": "Taille de l'entite",
    "exp_moyenne_filiale": "Experience moyenne de l'entite",
    "nb_certifications_total": "Nombre total de certifications (entite)",
    "nb_certifications_actives": "Certifications actives (entite)",
    "nb_consultants_certifies": "Consultants certifies (entite)",
    "nb_types_certifications_distincts": "Diversite des certifications (entite)",
}
#l'explainer SHAP est mis en cache pour ne pas le recalculer à chaque interaction
@st.cache_resource
def get_explainer(_pipeline):
    return shap.TreeExplainer(_pipeline.named_steps["clf"])
explainer = get_explainer(model)
st.title("Scoring de prospects commerciaux")
st.write("Cet outil aide à savoir si un prospect a de bonnes chances d'aboutir, avant de passer du temps dessus.")
with st.sidebar:
    st.header("À propos")
    st.write("Renseignez les infos d'un prospect, l'outil calcule sa probabilité de succès et vous conseille.")
    st.divider()
    st.subheader("Réglage avancé")
    seuil = st.slider(
        "Seuil de décision",
        min_value=0.1, max_value=0.9, value=float(SEUIL_OPTIMAL), step=0.05,
        help="Seuil à partir duquel un prospect est considéré comme prometteur. "
             "Valeur calculée automatiquement à partir des données historiques (indice de Youden)."
    )
    st.caption(f"Seuil recommandé (calculé sur l'historique) : {SEUIL_OPTIMAL}")

if "historique" not in st.session_state:
    st.session_state.historique = []
st.subheader("Nouveau prospect")
col1, col2 = st.columns(2)
with col1:
    priority_label = st.selectbox("Priorité", list(priority_map.keys()), index=2)
    entite_label = st.selectbox("Entité", list(entityTrig_map.keys()))
with col2:
    duree = st.number_input("Durée de la mission (mois)", value=3.0)
    delai = st.number_input("Délai avant démarrage (mois)", value=1.0)
if st.button("Qualifier ce prospect"):
    entite_val = entityTrig_map[entite_label]
    stats = entite_stats[str(entite_val)]
    donnees = pd.DataFrame([{
        "duree_mission": duree,
        "priority_ordinal": priority_map[priority_label],
        "ecart_creation_demarrage": abs(delai),
        "entityTrig_code": entite_val,
        "nb_consultants_filiale": stats["nb_consultants_filiale"],
        "exp_moyenne_filiale": stats["exp_moyenne_filiale"],
        "nb_certifications_total": stats["nb_certifications_total"],
        "nb_certifications_actives": stats["nb_certifications_actives"],
        "nb_consultants_certifies": stats["nb_consultants_certifies"],
        "nb_types_certifications_distincts": stats["nb_types_certifications_distincts"],
    }])[features]
    proba = model.predict_proba(donnees)[0, 1]
    proba_pourcent = round(proba * 100, 1)
    marge = 0.15
    if proba >= seuil + marge:
        classification = "Fort potentiel"
        conseil = "Ce prospect a de bonnes chances d'aboutir : à prioriser."
        st.success(conseil)
    elif proba >= seuil - marge:
        classification = "À confirmer"
        conseil = "Ce prospect est dans une zone intermédiaire : à qualifier davantage avant de trancher."
        st.warning(conseil)
    else:
        classification = "Risque élevé"
        conseil = "Ce prospect semble moins prometteur en l'état : à surveiller."
        st.error(conseil)
    st.write(f"Probabilité de succès : {proba_pourcent} %")
    st.progress(proba)
    donnees_scaled = model.named_steps["scaler"].transform(donnees)
    shap_vals = explainer.shap_values(donnees_scaled)
    if isinstance(shap_vals, list):
        contributions = shap_vals[1][0]
    else:
        contributions = shap_vals[0, :, 1]
    contrib_df = pd.DataFrame({
        "feature": features,
        "contribution": contributions
    })
    contrib_df["abs_contribution"] = contrib_df["contribution"].abs()
    top3 = contrib_df.sort_values("abs_contribution", ascending=False).head(3)
    st.write("**Principaux facteurs pris en compte pour ce prospect :**")
    top5 = contrib_df.sort_values("abs_contribution", ascending=True).tail(5)
    labels_affiches = [LABELS_FEATURES.get(f, f) for f in top5["feature"]]
    couleurs = ["#2ecc71" if v > 0 else "#e74c3c" for v in top5["contribution"]]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(labels_affiches, top5["contribution"], color=couleurs)
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_xlabel("Impact sur le score de succès")
    ax.set_title("Facteurs d'influence pour ce prospect")
    fig.tight_layout()
    st.pyplot(fig)
    for _, ligne in top3.iterrows():
        label = LABELS_FEATURES.get(ligne["feature"], ligne["feature"])
        sens = "favorise le succès" if ligne["contribution"] > 0 else "joue en défaveur du succès"
        st.write(f"- {label} : {sens}")
    nouveau_prospect = {
        "Date": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "Priorité": priority_label,
        "Entité": entite_label,
        "Durée (mois)": duree,
        "Score (%)": proba_pourcent,
        "Résultat": classification,
    }
    st.session_state.historique.append(nouveau_prospect)
st.subheader("Historique de la session")
if len(st.session_state.historique) == 0:
    st.write("Aucun prospect qualifié pour le moment.")
else:
    df_historique = pd.DataFrame(st.session_state.historique)
    st.dataframe(df_historique)
    nb_fort = (df_historique["Résultat"] == "Fort potentiel").sum()
    nb_confirmer = (df_historique["Résultat"] == "À confirmer").sum()
    nb_risque = (df_historique["Résultat"] == "Risque élevé").sum()
    score_moyen = round(df_historique["Score (%)"].mean(), 1)
    st.write(
        f"Sur {len(df_historique)} prospects : {nb_fort} à fort potentiel, "
        f"{nb_confirmer} à confirmer, {nb_risque} à risque élevé. "
        f"Score moyen : {score_moyen} %."
    )
    csv = df_historique.to_csv(index=False).encode("utf-8")
    st.download_button("Télécharger l'historique en csv", data=csv, file_name="prospects_qualifies.csv")
    if st.button("Vider l'historique"):
        st.session_state.historique = []
        st.rerun()
