"""
ETAPE 3 - Dashboard de Monitoring avec Streamlit
Visualisation des prédictions, du drift et des performances
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from dotenv import load_dotenv
import os
from pymongo import MongoClient

# DB/collection par défaut (surchargées par variables d'env)
DEFAULT_MONGO_DB_NAME = 'perso'
DEFAULT_MONGO_COLLECTION_PREDICTIONS = 'predictions'

st.set_page_config(page_title="Monitoring API - Prêt à Dépenser", layout="wide")

# ============================================================================
# CONFIGURATION
# ============================================================================

TITLE = "📊 Dashboard de Monitoring - API de Scoring"

# ============================================================================
# FONCTIONS
# ============================================================================

@st.cache_data
def load_logs():
    """Placeholder kept for compatibility (not used)"""
    return None


# ============================================================================
# HEADER
# ============================================================================

st.markdown(f"# {TITLE}")
st.markdown("---")

# Choix de la source de données
load_dotenv()

@st.cache_data
def load_logs_from_db(limit=10000):
    """Charger les logs depuis MongoDB en se connectant directement via MONGO_URI"""
    try:
        MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017')
        db_name_env = os.getenv('MONGO_DB_NAME')
        collection_name = os.getenv('MONGO_COLLECTION_PREDICTIONS', DEFAULT_MONGO_COLLECTION_PREDICTIONS)
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        # ping
        client.admin.command('ping')

        # déterminer la db depuis la chaîne de connexion si possible
        db_name = None
        if '/' in MONGO_URI:
            try:
                db_name = MONGO_URI.rsplit('/', 1)[-1].split('?')[0]
                if db_name == '' or db_name.lower().startswith('mongodb'):
                    db_name = None
            except Exception:
                db_name = None

        # fallback vers MONGO_DB_NAME (prioritaire) sinon défaut
        if db_name_env:
            db_name = db_name_env
        if not db_name:
            db_name = DEFAULT_MONGO_DB_NAME

        db = client[db_name]
        coll = db[collection_name]

        docs = list(coll.find().limit(limit).sort('timestamp', -1))
        if not docs:
            return None

        df = pd.DataFrame(docs)
        if '_id' in df.columns:
            df = df.drop(columns=['_id'])

        # normaliser timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        else:
            # si présence d'un champ 'date'
            if 'date' in df.columns:
                df['timestamp'] = pd.to_datetime(df['date'], errors='coerce')

        # convertir certaines colonnes en numeric si besoin
        for col in ['probability', 'execution_time_ms', 'prediction']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df['date'] = df['timestamp'].dt.date
        df['hour'] = df['timestamp'].dt.hour
        df['risk_level'] = df['probability'].apply(
            lambda x: 'BAS' if x < 0.3 else 'MOYEN' if x < 0.5 else 'ÉLEVÉ'
        )

        return df
    except Exception as e:
        st.error(
            "❌ Erreur lors du chargement depuis MongoDB.\n\n"
            f"- MONGO_URI: {os.getenv('MONGO_URI', '(non défini)')}\n"
            f"- MONGO_DB_NAME: {os.getenv('MONGO_DB_NAME', DEFAULT_MONGO_DB_NAME)}\n"
            f"- MONGO_COLLECTION_PREDICTIONS: {os.getenv('MONGO_COLLECTION_PREDICTIONS', DEFAULT_MONGO_COLLECTION_PREDICTIONS)}\n\n"
            f"Détail: {e}"
        )
        return None


# Charger les données (MongoDB uniquement)
df = load_logs_from_db()

if df is None:
    st.warning("⚠️  Aucun log trouvé dans MongoDB. Exécutez l'API pour générer des prédictions ou vérifiez la connexion.")
    st.info("Commande: `python app.py`")
    st.stop()

if df is None:
    st.warning("⚠️  Aucun log trouvé. Exécutez l'API pour générer des prédictions.")
    st.info("Commande: `python app.py`")
    st.stop()

# ============================================================================
# STATISTIQUES GLOBALES
# ============================================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Prédictions", len(df))

with col2:
    st.metric("Avg. Probabilité", f"{df['probability'].mean():.2%}")

with col3:
    st.metric("Avg. Latence", f"{df['execution_time_ms'].mean():.2f}ms")

with col4:
    high_risk_count = (df['probability'] >= 0.5).sum()
    st.metric("Haut Risque", high_risk_count)

st.markdown("---")

# ============================================================================
# SECTION 1: DISTRIBUTION DES PROBABILITÉS
# ============================================================================

st.markdown("## 📊 1. Distribution des Prédictions")

col1, col2 = st.columns(2)

# Histogramme
with col1:
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=df['probability'],
        nbinsx=30,
        name='Probabilité',
        marker_color='rgb(55, 83, 109)'
    ))
    
    # Ajouter les seuils de risque
    fig_hist.add_vline(x=0.3, line_dash="dash", line_color="green", 
                       annotation_text="Bas/Moyen", annotation_position="top left")
    fig_hist.add_vline(x=0.5, line_dash="dash", line_color="red",
                       annotation_text="Moyen/Haut", annotation_position="top right")
    
    fig_hist.update_layout(
        title="Distribution des Probabilités",
        xaxis_title="Probabilité",
        yaxis_title="Fréquence",
        height=400
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# Pie chart des risques
with col2:
    risk_dist = df['risk_level'].value_counts()
    fig_pie = go.Figure(data=[go.Pie(
        labels=risk_dist.index,
        values=risk_dist.values,
        marker=dict(colors=['#2ecc71', '#f39c12', '#e74c3c'])
    )])
    fig_pie.update_layout(
        title="Distribution des Niveaux de Risque",
        height=400
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# ============================================================================
# SECTION 2: PERFORMANCES
# ============================================================================

st.markdown("## ⏱️  2. Performance de l'API")

col1, col2 = st.columns(2)

# Latence dans le temps
with col1:
    fig_latency = go.Figure()
    fig_latency.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['execution_time_ms'],
        mode='lines+markers',
        name='Latence',
        marker=dict(size=6, color=df['execution_time_ms'], 
                   colorscale='Viridis', showscale=True)
    ))
    
    fig_latency.update_layout(
        title="Latence de l'API dans le temps",
        xaxis_title="Temps",
        yaxis_title="Latence (ms)",
        height=400,
        hovermode='x unified'
    )
    st.plotly_chart(fig_latency, use_container_width=True)

# Box plot de latence par heure
with col2:
    fig_box = go.Figure()
    for hour in sorted(df['hour'].unique()):
        hour_data = df[df['hour'] == hour]['execution_time_ms']
        fig_box.add_trace(go.Box(
            y=hour_data,
            name=f"{hour:02d}:00",
            boxmean='sd'
        ))
    
    fig_box.update_layout(
        title="Distribution de latence par heure",
        yaxis_title="Latence (ms)",
        height=400,
        showlegend=False
    )
    st.plotly_chart(fig_box, use_container_width=True)

# ============================================================================
# SECTION 3: TENDANCES TEMPORELLES
# ============================================================================

st.markdown("## 📅 3. Tendances Temporelles")

col1, col2 = st.columns(2)

# Prédictions par jour
with col1:
    daily_counts = df.groupby('date').size()
    fig_daily = go.Figure()
    fig_daily.add_trace(go.Bar(
        x=daily_counts.index.astype(str),
        y=daily_counts.values,
        marker_color='rgb(158, 202, 225)'
    ))
    fig_daily.update_layout(
        title="Nombre de prédictions par jour",
        xaxis_title="Date",
        yaxis_title="Nombre",
        height=400
    )
    st.plotly_chart(fig_daily, use_container_width=True)

# Probabilité moyenne par heure
with col2:
    hourly_prob = df.groupby('hour')['probability'].mean()
    fig_hourly = go.Figure()
    fig_hourly.add_trace(go.Scatter(
        x=hourly_prob.index,
        y=hourly_prob.values,
        mode='lines+markers',
        name='Prob moyenne',
        marker=dict(size=10)
    ))
    fig_hourly.update_layout(
        title="Probabilité moyenne par heure",
        xaxis_title="Heure",
        yaxis_title="Probabilité",
        height=400
    )
    st.plotly_chart(fig_hourly, use_container_width=True)

# ============================================================================
# SECTION 4: ANALYSE DE DRIFT
# ============================================================================

st.markdown("## 🔍 4. Analyse du Data Drift")

if len(df) >= 4:
    mid_point = len(df) // 2
    early_prob = df.iloc[:mid_point]['probability']
    late_prob = df.iloc[mid_point:]['probability']
    
    mean_diff = abs(early_prob.mean() - late_prob.mean())
    drift_detected = mean_diff > 0.05
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Différence de moyenne", f"{mean_diff:.2%}", 
                 "⚠️  ALERTE" if drift_detected else "✅ OK")
    
    with col2:
        st.metric("Drift détecté?", "OUI" if drift_detected else "NON",
                 "🔴" if drift_detected else "🟢")
    
    # Distribution early vs late
    fig_drift = go.Figure()
    fig_drift.add_trace(go.Histogram(
        x=early_prob,
        name='Début (première moitié)',
        opacity=0.7,
        nbinsx=20
    ))
    fig_drift.add_trace(go.Histogram(
        x=late_prob,
        name='Fin (deuxième moitié)',
        opacity=0.7,
        nbinsx=20
    ))
    
    fig_drift.update_layout(
        title="Comparaison: Début vs Fin des prédictions",
        xaxis_title="Probabilité",
        yaxis_title="Fréquence",
        barmode='overlay',
        height=400
    )
    st.plotly_chart(fig_drift, use_container_width=True)
    
    # Statistiques
    st.write("### Statistiques Détaillées")
    stats_data = {
        'Métrique': ['Moyenne', 'Médiane', 'Écart-type', 'Min', 'Max', 'Q1', 'Q3'],
        'Début': [
            f"{early_prob.mean():.2%}",
            f"{early_prob.median():.2%}",
            f"{early_prob.std():.2%}",
            f"{early_prob.min():.2%}",
            f"{early_prob.max():.2%}",
            f"{early_prob.quantile(0.25):.2%}",
            f"{early_prob.quantile(0.75):.2%}"
        ],
        'Fin': [
            f"{late_prob.mean():.2%}",
            f"{late_prob.median():.2%}",
            f"{late_prob.std():.2%}",
            f"{late_prob.min():.2%}",
            f"{late_prob.max():.2%}",
            f"{late_prob.quantile(0.25):.2%}",
            f"{late_prob.quantile(0.75):.2%}"
        ]
    }
    st.dataframe(pd.DataFrame(stats_data), use_container_width=True)
else:
    st.info("ℹ️  Pas assez de données pour analyser le drift (minimum 4 prédictions)")

# ============================================================================
# SECTION 5: DONNÉES BRUTES
# ============================================================================

st.markdown("## 📋 5. Données Brutes")

st.write(f"Total: {len(df)} prédictions")

# Permettre le filtrage
col1, col2, col3 = st.columns(3)

with col1:
    date_filter = st.selectbox("Filtrer par date:", 
                               [None] + sorted(df['date'].unique()))

with col2:
    risk_filter = st.multiselect("Filtrer par risque:",
                                 df['risk_level'].unique(),
                                 default=df['risk_level'].unique())

df_filtered = df.copy()
if date_filter:
    df_filtered = df_filtered[df_filtered['date'] == date_filter]
if risk_filter:
    df_filtered = df_filtered[df_filtered['risk_level'].isin(risk_filter)]

# Afficher le tableau
display_cols = ['timestamp', 'probability', 'execution_time_ms', 'risk_level', 'prediction']
st.dataframe(df_filtered[display_cols].sort_values('timestamp', ascending=False),
            use_container_width=True, height=400)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
### 📝 Interprétation

- **Bas Risque (🟢)**: Probabilité < 30% → Crédit recommandé
- **Risque Moyen (🟡)**: Probabilité 30-50% → Vérification supplémentaire
- **Haut Risque (🔴)**: Probabilité > 50% → Crédit à refuser ou réduire

### ⚠️ Points de vigilance

- **Drift**: Si détecté, le modèle peut perdre en précision
- **Latence**: P99 acceptable < 100ms
- **Taux d'erreur**: Doit rester < 1%
""")

st.markdown(f"*Dernière mise à jour: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
