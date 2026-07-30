# app.py : interface Streamlit, toute la logique est dans scoring.py

from datetime import datetime
from pathlib import Path
import pandas as pd
import streamlit as st
from scoring import MIN_MISSIONS, charger_ressources, charger_win_ratio, scorer, calculer_contrib

st.set_page_config(page_title="Scoring de prospects", layout="centered")


def render(html):
    # on affiche du html propre, sans les espaces de debut de ligne
    lignes = "\n".join(l.strip() for l in html.strip().splitlines())
    st.markdown(lignes, unsafe_allow_html=True)


with open(Path(__file__).resolve().parent / "style.css", "r", encoding="utf-8") as f:
    render(f"<style>{f.read()}</style>")

try:
    ressources = charger_ressources()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

render("""
<div class="s4-header">
    <div class="s4-logo">S4</div>
    <div class="s4-header-texte">
        <p class="s4-header-titre">Scoring de prospects commerciaux</p>
        <p class="s4-header-sous-titre">Smart4 Engineering &middot; aide à la décision commerciale</p>
    </div>
</div>
""")

with st.sidebar:
    st.subheader("À propos")
    st.write("Renseignez les infos d'un prospect, l'outil calcule sa probabilité de succès et vous conseille.")
    st.divider()
    st.subheader("Accessibilité")
    grand_texte = st.toggle("Texte agrandi", value=False)
    st.divider()
    st.subheader("Réglage avancé")
    seuil = st.slider("Seuil de décision", 0.1, 0.9, float(ressources["seuil"]), 0.05)
    st.caption(f"Seuil recommandé : {ressources['seuil']}")

if grand_texte:
    render("<style>.block-container, .block-container p, .block-container span, .block-container label { font-size: 115% !important; }</style>")

if "historique" not in st.session_state:
    st.session_state.historique = []

render('<p class="s4-card-titre">Nouveau prospect</p>')
with st.container(border=True):
    col1, col2 = st.columns(2)
    with col1:
        priorite = st.selectbox("Priorité", list(ressources["priority_map"].keys()), index=2)
        entite = st.selectbox("Entité", list(ressources["entite_map"].keys()))
    with col2:
        duree = st.number_input("Durée de la mission (mois)", value=3.0, min_value=0.0)
        delai = st.number_input("Délai avant démarrage (mois)", value=1.0)
    qualifier = st.button("Qualifier ce prospect", type="primary", use_container_width=True)

if qualifier:
    # on calcule le score et on prepare les 3 facteurs les plus importants
    resultat, ligne = scorer(ressources, priorite, entite, duree, delai, seuil)
    proba_pct = round(resultat["proba"] * 100, 1)

    badges = {
        "Fort potentiel": ("s4-badge-fort", "&#10003;"),
        "À confirmer": ("s4-badge-confirmer", "&#8212;"),
        "Risque élevé": ("s4-badge-risque", "&#33;"),
    }
    badge_classe, badge_icone = badges[resultat["niveau"]]

    rayon = 226
    decalage = rayon - (rayon * resultat["proba"])

    contrib = calculer_contrib(ressources, ligne)
    facteurs_html = ""
    for _, ligne_f in contrib.head(3).iterrows():
        icone = "&#8599;" if ligne_f["favorise"] else "&#8600;"
        classe_mot = "s4-facteur-favorise" if ligne_f["favorise"] else "s4-facteur-defavorise"
        mot = "Favorise" if ligne_f["favorise"] else "Défavorise"
        facteurs_html += f"""
            <div class="s4-facteur">
                <span aria-hidden="true" style="font-weight:700;">{icone}</span>
                <span class="s4-facteur-libelle">{ligne_f['label']}</span>
                <span class="{classe_mot}">{mot}</span>
            </div>
        """

    render(f"""
    <div class="s4-card">
        <div style="display:flex;gap:20px;align-items:center;margin-bottom:16px;">
            <svg width="80" height="80" viewBox="0 0 84 84" role="img" aria-label="Score de {proba_pct} pourcent">
                <circle cx="42" cy="42" r="36" fill="none" stroke="#e4e7ec" stroke-width="9"/>
                <circle cx="42" cy="42" r="36" fill="none" stroke="var(--s4-orange-texte)" stroke-width="9"
                    stroke-dasharray="{rayon}" stroke-dashoffset="{decalage}"
                    stroke-linecap="round" transform="rotate(-90 42 42)"/>
                <text x="42" y="47" text-anchor="middle" class="s4-gauge-valeur">{proba_pct}%</text>
            </svg>
            <div>
                <div class="s4-badge {badge_classe}">
                    <span aria-hidden="true">{badge_icone}</span> {resultat['niveau']}
                </div>
                <p style="color:#3d3d3a;font-size:13px;margin:8px 0 0;">{resultat['conseil']}</p>
            </div>
        </div>
        <p class="s4-card-titre">Facteurs pris en compte</p>
        {facteurs_html}
    </div>
    """)

    st.session_state.historique.append({
        "Date": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "Priorité": priorite,
        "Entité": entite,
        "Durée (mois)": duree,
        "Score (%)": proba_pct,
        "Résultat": resultat["niveau"],
    })
    render('<p class="s4-card-titre">Historique de la session</p>')

if not st.session_state.historique:
    st.caption("Aucun prospect qualifié pour le moment.")
else:
    df_hist = pd.DataFrame(st.session_state.historique)
    nb_fort = int((df_hist["Résultat"] == "Fort potentiel").sum())
    nb_confirmer = int((df_hist["Résultat"] == "À confirmer").sum())
    score_moyen = round(df_hist["Score (%)"].mean(), 1)

    c1, c2, c3, c4 = st.columns(4)
    for col, label, valeur in zip(
        (c1, c2, c3, c4),
        ("Qualifiés", "Fort potentiel", "À confirmer", "Score moyen"),
        (len(df_hist), nb_fort, nb_confirmer, f"{score_moyen} %"),
    ):
        with col:
            render(f'<div class="s4-metric"><p class="s4-metric-label">{label}</p><p class="s4-metric-valeur">{valeur}</p></div>')

    st.dataframe(df_hist, use_container_width=True, hide_index=True)
    csv = df_hist.to_csv(index=False).encode("utf-8")
    col_dl, col_clear = st.columns(2)
    col_dl.download_button("Télécharger en CSV", data=csv, file_name="prospects_qualifies.csv", use_container_width=True)
    if col_clear.button("Vider l'historique", use_container_width=True):
        st.session_state.historique = []
        st.rerun()

st.divider()
render('<p class="s4-card-titre">Taux de réussite historique</p>')

try:
    win_ratio = charger_win_ratio(MIN_MISSIONS)
    win_ratio["entite"] = win_ratio["entityTrig_code"].map({v: k for k, v in ressources["entite_map"].items()})

    c1, c2 = st.columns(2)
    with c1:
        render('<div class="s4-metric"><p class="s4-metric-label">Win ratio global</p><p class="s4-metric-valeur">25,0 %</p></div>')
    with c2:
        render(f'<div class="s4-metric"><p class="s4-metric-label">Missions analysées</p><p class="s4-metric-valeur">{int(win_ratio["nb_missions"].sum())}</p></div>')
    st.bar_chart(win_ratio.set_index("entite")["win_ratio"], color="#e8792a")
except FileNotFoundError:
    st.caption("Fichier win_ratio_par_entite.csv non trouvé — section non affichée.")
