import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import pickle
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Survival Analysis - Maram Chebbi",
    page_icon="🧬",
    layout="wide"
)

@st.cache_resource
def load_models():
    model = keras.models.load_model('deepsurv_model.h5', compile=False)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    with open('metrics.pkl', 'rb') as f:
        metrics = pickle.load(f)
    return model, scaler, metadata, metrics

try:
    model, scaler, metadata, metrics = load_models()
    models_loaded = True
except:
    models_loaded = False

st.markdown("""
<style>
    .stButton>button {
        background: linear-gradient(135deg, #1A367E 0%, #4A8FE7 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 12px 30px;
        border-radius: 10px;
        width: 100%;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1A367E;
        margin: 10px 0;
    }
    .risk-low {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 5px solid #28a745;
    }
    .risk-medium {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border-left: 5px solid #ffc107;
    }
    .risk-high {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border-left: 5px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧬 Survival Analysis - DeepSurv")
st.markdown("### Prédiction de Risque pour Tarification d'Assurance-Vie")
st.markdown("**Développé par** : Maram Chebbi | ESPRIT & IRA Le Mans")
st.markdown("---")

if not models_loaded:
    st.error("⚠️ Modèles non chargés. Veuillez uploader les fichiers requis.")
    st.stop()

st.sidebar.header("📊 Performance du Modèle")
st.sidebar.metric("C-index (Test)", f"{metrics['test_c_index']:.3f}")
st.sidebar.metric("C-index (Train)", f"{metrics['train_c_index']:.3f}")
st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 Architecture")
st.sidebar.write("**Modèle**: DeepSurv Neural Network")
st.sidebar.write(f"**Features**: {metrics['n_features']}")
st.sidebar.write(f"**Dataset**: {metrics['dataset_size']} observations")
st.sidebar.markdown("---")
st.sidebar.info("**C-index > 0.7** = Excellent pouvoir prédictif")

st.subheader("📝 Informations sur l'Assuré")

feature_cols = metadata['feature_cols']

feature_labels = {
    'fin': '💰 Situation Financière (0=Mauvaise, 1=Bonne)',
    'age': '🎂 Âge (en années)',
    'race': '👤 Origine (0=Autre, 1=Noir)',
    'wexp': '💼 Expérience Professionnelle (0=Non, 1=Oui)',
    'mar': '💍 Situation Maritale (0=Célibataire, 1=Marié)',
    'paro': '👮 Libération Conditionnelle (0=Non, 1=Oui)',
    'prio': '📋 Nombre d\'Arrestations Antérieures',
    'week': '📅 Semaines depuis la Libération',
    'arrest': '🚨 Événement (0=Non, 1=Oui)'
}

feature_descriptions = {
    'fin': 'Stabilité financière de l\'assuré',
    'age': 'Âge de l\'assuré en années',
    'race': 'Catégorie démographique',
    'wexp': 'A une expérience de travail à temps plein',
    'mar': 'Statut marital',
    'paro': 'Libéré en conditionnelle',
    'prio': 'Nombre de condamnations antérieures'
}

feature_ranges = {
    'fin': (0, 1),
    'age': (18, 100),
    'race': (0, 1),
    'wexp': (0, 1),
    'mar': (0, 1),
    'paro': (0, 1),
    'prio': (0, 20)
}

n_cols = 3
cols = st.columns(n_cols)

inputs = {}
for idx, feature in enumerate(feature_cols):
    with cols[idx % n_cols]:
        label = feature_labels.get(feature, feature.replace('_', ' ').title())
        description = feature_descriptions.get(feature, f"Valeur pour {feature}")
        min_val, max_val = feature_ranges.get(feature, (0, 100))
        
        if max_val == 1:
            inputs[feature] = st.selectbox(
                label,
                options=[0, 1],
                index=0,
                key=feature,
                help=description
            )
        else:
            inputs[feature] = st.number_input(
                label,
                min_value=float(min_val),
                max_value=float(max_val),
                value=float(min_val + (max_val - min_val) // 2),
                step=1.0 if feature in ['age', 'prio'] else 0.1,
                key=feature,
                help=description
            )
if st.button("🧬 Analyser le Risque", use_container_width=True):
    with st.spinner("Analyse en cours..."):
        features_list = [inputs.get(feat, 0) for feat in metadata['feature_cols']]
        features_array = np.array(features_list).reshape(1, -1)
        features_scaled = scaler.transform(features_array)
        
        risk_score = float(model.predict(features_scaled, verbose=0)[0][0])
        
        risk_percentile = 50.0
        
        if risk_score < -0.5:
            risk_category = 'Faible'
            risk_class = 'risk-low'
            risk_color = 'green'
            risk_emoji = '✅'
        elif risk_score < 0.5:
            risk_category = 'Moyen'
            risk_class = 'risk-medium'
            risk_color = 'orange'
            risk_emoji = '⚠️'
        else:
            risk_category = 'Élevé'
            risk_class = 'risk-high'
            risk_color = 'red'
            risk_emoji = '🚨'
        
        st.markdown("---")
        st.subheader("📈 Résultat de l'Analyse de Risque")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f'<div class="metric-card {risk_class}">', unsafe_allow_html=True)
            st.metric("Catégorie de Risque", f"{risk_emoji} {risk_category}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Score de Risque", f"{risk_score:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            base_premium = 1000
            if risk_category == 'Faible':
                premium_multiplier = 0.8
            elif risk_category == 'Moyen':
                premium_multiplier = 1.0
            else:
                premium_multiplier = 1.5
            
            estimated_premium = base_premium * premium_multiplier
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Prime Estimée", f"${estimated_premium:.2f}/an")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Interprétation")
            
            if risk_category == 'Faible':
                st.success("""
                **Profil à Faible Risque** ✅
                - Score de risque négatif
                - Espérance de vie supérieure à la moyenne
                - Recommandation : Prime standard ou réduite
                """)
            elif risk_category == 'Moyen':
                st.warning("""
                **Profil à Risque Moyen** ⚠️
                - Score de risque modéré
                - Espérance de vie dans la moyenne
                - Recommandation : Prime standard
                """)
            else:
                st.error("""
                **Profil à Risque Élevé** 🚨
                - Score de risque élevé
                - Facteurs de risque identifiés
                - Recommandation : Prime majorée ou examen médical
                """)
            
            st.info(f"""
            **Base de calcul:**
            - Prime de base : ${base_premium}/an
            - Multiplicateur : {premium_multiplier}x
            - Prime finale : ${estimated_premium:.2f}/an
            """)
        
        with col2:
            st.markdown("### 📈 Visualisation du Risque")
            
            fig, ax = plt.subplots(figsize=(8, 6))
            
            risk_levels = ['Faible', 'Moyen', 'Élevé']
            risk_ranges = [-2, -0.5, 0.5, 2]
            colors = ['green', 'orange', 'red']
            
            for i in range(len(risk_levels)):
                ax.barh(risk_levels[i], risk_ranges[i+1] - risk_ranges[i], 
                       left=risk_ranges[i], color=colors[i], alpha=0.3)
            
            ax.axvline(x=risk_score, color='black', linewidth=3, linestyle='--', 
                      label=f'Votre score: {risk_score:.3f}')
            ax.axvline(x=0, color='gray', linewidth=1, linestyle='-', alpha=0.5)
            
            ax.set_xlabel('Score de Risque', fontsize=12)
            ax.set_title('Position sur l\'Échelle de Risque', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3, axis='x')
            
            st.pyplot(fig)

st.markdown("---")

with st.expander("📚 À propos du modèle DeepSurv"):
    st.markdown("""
    ### Méthodologie
    
    **DeepSurv - Deep Learning for Survival Analysis**
    - Architecture : Neural Network multi-couches
    - Loss Function : Cox Proportional Hazards
    - Optimisation : Adam optimizer
    
    **Métriques de Performance**
    - **C-index (Concordance Index)** : Mesure la capacité du modèle à classer correctement les paires d'observations
    - C-index > 0.7 : Excellent
    - C-index 0.5-0.7 : Bon
    - C-index < 0.5 : Faible
    
    **Applications**
    - Tarification d'assurance-vie
    - Évaluation de risque médical
    - Prédiction de survie
    - Segmentation de clientèle
    
    **Performance actuelle**
    - C-index Test : {:.3f}
    - Dataset : {} observations
    - Features : {}
    """.format(metrics['test_c_index'], metrics['dataset_size'], metrics['n_features']))

with st.expander("🔬 Exemples de Profils"):
    st.markdown("""
    ### Profil 1 : Faible Risque
    - Jeune âge
    - Bonne santé
    - Pas de facteurs de risque
    - **Prime** : ~$800/an
    
    ### Profil 2 : Risque Moyen
    - Âge moyen
    - Quelques facteurs de risque mineurs
    - **Prime** : ~$1,000/an
    
    ### Profil 3 : Risque Élevé
    - Âge avancé
    - Multiples facteurs de risque
    - **Prime** : ~$1,500/an
    """)

st.markdown("---")
st.markdown("**📧 Contact** : chebbimaram0@gmail.com | [LinkedIn](https://linkedin.com/in/maramchebbi) | [GitHub](https://github.com/maramchebbi)")
