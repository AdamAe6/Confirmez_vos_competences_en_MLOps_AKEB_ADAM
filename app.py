"""
API de scoring avec Gradio
Charge un modèle XGBoost et expose une interface pour faire des prédictions.
"""

import pickle
import pandas as pd
import numpy as np
import gradio as gr
import logging
from pathlib import Path
import json
from datetime import datetime

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Chemin du modèle
MODEL_PATH = Path("./exported_model/model/model.pkl")
LOGS_DIR = Path("./logs")
LOGS_DIR.mkdir(exist_ok=True)

# Charger le modèle une seule fois au démarrage (important!)
logger.info("Chargement du modèle...")
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    logger.info("✅ Modèle chargé avec succès")
    EXPECTED_FEATURES = list(model.feature_names_in_)
    NUM_FEATURES = len(EXPECTED_FEATURES)
except Exception as e:
    logger.error(f"❌ Erreur lors du chargement du modèle: {e}")
    model = None
    EXPECTED_FEATURES = []
    NUM_FEATURES = 0


def log_prediction(input_data: dict, prediction: int, probability: float, execution_time: float):
    """Log la prédiction et les métadonnées pour le monitoring"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "input_features_count": len(input_data),
        "prediction": int(prediction),
        "probability": float(probability),
        "execution_time_ms": execution_time,
        "status": "success"
    }
    
    # Sauvegarder dans un fichier log JSON
    log_file = LOGS_DIR / f"predictions_{datetime.now().strftime('%Y%m%d')}.jsonl"
    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')
    
    return log_entry


def predict_score(ext_source_3, ext_source_2, amt_req_credit_bureau_day,
                  name_education_type, code_gender, flag_document_3,
                  ext_source_1, name_income_type, flag_emp_phone,
                  amt_goods_price, prev_approved_ratio, pos_count,
                  days_birth, own_car_age, flag_own_car,
                  name_contract_type, days_employed, reg_city_not_live_city,
                  def_60_cnt_social_circle, amt_credit):
    """
    Fonction de prédiction pour Gradio.
    Prend les 20 features les plus importantes en entrée.
    Retourne le score de probabilité et le niveau de risque.
    """
    
    if model is None:
        return "❌ ERREUR: Modèle non disponible", "N/A", "N/A"
    
    try:
        import time
        start_time = time.time()
        
        # Créer un dictionnaire avec les 20 features les plus importantes
        input_dict = {
            'EXT_SOURCE_3': ext_source_3,
            'EXT_SOURCE_2': ext_source_2,
            'AMT_REQ_CREDIT_BUREAU_DAY': amt_req_credit_bureau_day,
            'NAME_EDUCATION_TYPE': name_education_type,
            'CODE_GENDER': code_gender,
            'FLAG_DOCUMENT_3': flag_document_3,
            'EXT_SOURCE_1': ext_source_1,
            'NAME_INCOME_TYPE': name_income_type,
            'FLAG_EMP_PHONE': flag_emp_phone,
            'AMT_GOODS_PRICE': amt_goods_price,
            'prev_approved_ratio': prev_approved_ratio,
            'pos_count': pos_count,
            'DAYS_BIRTH': days_birth,
            'OWN_CAR_AGE': own_car_age,
            'FLAG_OWN_CAR': flag_own_car,
            'NAME_CONTRACT_TYPE': name_contract_type,
            'DAYS_EMPLOYED': days_employed,
            'REG_CITY_NOT_LIVE_CITY': reg_city_not_live_city,
            'DEF_60_CNT_SOCIAL_CIRCLE': def_60_cnt_social_circle,
            'AMT_CREDIT': amt_credit
        }
        
        # Note importante: Le modèle a été entraîné avec 128 features
        # Cette fonction utilise les 20 features les plus importantes pour l'interface
        # En production, charger les données complètes et les transformer correctement
        
        # Créer un array avec les bonnes dimensions (128 features attendues)
        X_complete = np.zeros((1, 128))
        
        # Remplir les features disponibles (indices basés sur feature_names_in_)
        for feature_name, value in input_dict.items():
            if feature_name in EXPECTED_FEATURES:
                feature_idx = EXPECTED_FEATURES.index(feature_name)
                X_complete[0, feature_idx] = float(value)
        
        # Faire la prédiction
        prediction = model.predict(X_complete)[0]
        probability = model.predict_proba(X_complete)[0, 1]
        
        execution_time = (time.time() - start_time) * 1000  # en ms
        
        # Déterminer le niveau de risque
        if probability < 0.3:
            risk_level = "🟢 BAS RISQUE"
            risk_color = "green"
        elif probability < 0.5:
            risk_level = "🟡 RISQUE MOYEN"
            risk_color = "orange"
        else:
            risk_level = "🔴 RISQUE ÉLEVÉ"
            risk_color = "red"
        
        # Logger la prédiction
        log_prediction(input_dict, prediction, probability, execution_time)
        
        # Préparer la réponse
        result = f"Probabilité de défaut: {probability:.2%}\nLatence: {execution_time:.2f}ms"
        
        return result, risk_level, f"{probability:.4f}"
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la prédiction: {e}")
        return f"❌ Erreur: {str(e)}", "N/A", "N/A"


# Interface Gradio
with gr.Blocks(title="API de Scoring - Prêt à Dépenser") as demo:
    gr.Markdown("# 📊 API de Scoring de Crédit - Prêt à Dépenser")
    gr.Markdown("🎯 Prédiction du risque de défaut de paiement basée sur les 20 features les plus importantes")
    
    with gr.Row():
        gr.Markdown("### 📋 Entrez les informations du client:")
    
    # Les 20 features les plus importantes (TOP 20)
    # Groupées par 5 pour une meilleure UX
    
    with gr.Row():
        gr.Markdown("#### 🏆 Top 5 Features (40% du poids du modèle)")
    
    with gr.Row():
        ext_source_3 = gr.Number(label="1️⃣ EXT_SOURCE_3 (10.1%)", value=0.5, minimum=0, maximum=1)
        ext_source_2 = gr.Number(label="2️⃣ EXT_SOURCE_2 (8.7%)", value=0.5, minimum=0, maximum=1)
        amt_req_credit_bureau_day = gr.Number(label="3️⃣ AMT_REQ_CREDIT_BUREAU_DAY (5.0%)", value=0)
        name_education_type = gr.Number(label="4️⃣ NAME_EDUCATION_TYPE (3.2%)", value=0)
        code_gender = gr.Number(label="5️⃣ CODE_GENDER (3.1%)", value=0)
    
    with gr.Row():
        gr.Markdown("#### 🥈 Features 6-10 (10% du poids)")
    
    with gr.Row():
        flag_document_3 = gr.Number(label="6️⃣ FLAG_DOCUMENT_3 (2.9%)", value=0)
        ext_source_1 = gr.Number(label="7️⃣ EXT_SOURCE_1 (2.5%)", value=0.5, minimum=0, maximum=1)
        name_income_type = gr.Number(label="8️⃣ NAME_INCOME_TYPE (2.0%)", value=0)
        flag_emp_phone = gr.Number(label="9️⃣ FLAG_EMP_PHONE (2.0%)", value=0)
        amt_goods_price = gr.Number(label="🔟 AMT_GOODS_PRICE (2.0%)", value=0)
    
    with gr.Row():
        gr.Markdown("#### 🥉 Features 11-15 (9% du poids)")
    
    with gr.Row():
        prev_approved_ratio = gr.Number(label="1️⃣1️⃣ prev_approved_ratio (1.9%)", value=0, minimum=0, maximum=1)
        pos_count = gr.Number(label="1️⃣2️⃣ pos_count (1.8%)", value=0)
        days_birth = gr.Number(label="1️⃣3️⃣ DAYS_BIRTH (1.8%)", value=-20000)
        own_car_age = gr.Number(label="1️⃣4️⃣ OWN_CAR_AGE (1.7%)", value=0)
        flag_own_car = gr.Number(label="1️⃣5️⃣ FLAG_OWN_CAR (1.6%)", value=0)
    
    with gr.Row():
        gr.Markdown("#### 📊 Features 16-20 (7% du poids)")
    
    with gr.Row():
        name_contract_type = gr.Number(label="1️⃣6️⃣ NAME_CONTRACT_TYPE (1.5%)", value=0)
        days_employed = gr.Number(label="1️⃣7️⃣ DAYS_EMPLOYED (1.4%)", value=0)
        reg_city_not_live_city = gr.Number(label="1️⃣8️⃣ REG_CITY_NOT_LIVE_CITY (1.4%)", value=0)
        def_60_cnt_social_circle = gr.Number(label="1️⃣9️⃣ DEF_60_CNT_SOCIAL_CIRCLE (1.4%)", value=0)
        amt_credit = gr.Number(label="2️⃣0️⃣ AMT_CREDIT (1.4%)", value=0)
    
    # Bouton de prédiction
    with gr.Row():
        predict_btn = gr.Button("🎯 Prédire le Risque", variant="primary", scale=2)
        clear_btn = gr.Button("🔄 Réinitialiser", scale=1)
    
    # Outputs
    with gr.Row():
        output_score = gr.Textbox(label="📈 Résultat", interactive=False, lines=2)
    
    with gr.Row():
        output_risk = gr.Textbox(label="⚠️  Niveau de Risque", interactive=False)
        output_probability = gr.Textbox(label="📊 Probabilité (0-1)", interactive=False)
    
    # Connecter les inputs et outputs
    all_inputs = [
        ext_source_3, ext_source_2, amt_req_credit_bureau_day, name_education_type, code_gender,
        flag_document_3, ext_source_1, name_income_type, flag_emp_phone, amt_goods_price,
        prev_approved_ratio, pos_count, days_birth, own_car_age, flag_own_car,
        name_contract_type, days_employed, reg_city_not_live_city, def_60_cnt_social_circle, amt_credit
    ]
    
    all_outputs = [output_score, output_risk, output_probability]
    
    # Bouton prédire
    predict_btn.click(
        fn=predict_score,
        inputs=all_inputs,
        outputs=all_outputs
    )
    
    # Bouton réinitialiser
    clear_btn.click(
        fn=lambda: (["", "", ""] + [0] * 20),
        inputs=[],
        outputs=all_outputs + all_inputs
    )
    
    gr.Markdown("---")
    gr.Markdown("### � Informations")
    gr.Markdown("""
    - **Modèle**: XGBoost avec 128 features (entraîné sur données de crédit)
    - **Performance**: AUC = 0.7564, Accuracy = 69.63%
    - **Interface**: Les 20 features les plus importantes sont affichées
    - **Logging**: Chaque prédiction est enregistrée pour le monitoring
    - **Risque BAS**: Probabilité < 30%
    - **Risque MOYEN**: Probabilité 30-50%
    - **Risque ÉLEVÉ**: Probabilité > 50%
    """)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
