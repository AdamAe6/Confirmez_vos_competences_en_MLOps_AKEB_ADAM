"""
API de Scoring - Gradio + MongoDB Logging
Charge le modèle XGBoost et expose une interface pour les prédictions
Chaque prédiction est automatiquement pushée dans MongoDB Atlas
"""

import pickle
import pandas as pd
import numpy as np
import gradio as gr
import logging
from pathlib import Path
import json
from datetime import datetime
import time
import os
from pymongo import MongoClient
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Chemins
MODEL_PATH = Path("./exported_model/model/model.pkl")
LOGS_DIR = Path("./logs")
LOGS_DIR.mkdir(exist_ok=True)

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
mongo_client = None
mongo_db = None

def init_mongodb():
    """Initialiser la connexion MongoDB"""
    global mongo_client, mongo_db
    try:
        mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        mongo_client.admin.command('ping')
        mongo_db = mongo_client['perso']
        logger.info("✅ MongoDB connecté")
        return True
    except Exception as e:
        logger.warning(f"⚠️ MongoDB non disponible: {e}")
        return False

# Variables globales du modèle
model = None
EXPECTED_FEATURES = []
NUM_FEATURES = 0


def log_prediction(input_data: dict, prediction: int, probability: float, execution_time: float):
    """
    Push les prédictions dans MongoDB directement
    Simple et efficace!
    """
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "date": datetime.now().strftime("%Y-%m-%d"),
        "input_features": input_data,
        "prediction": int(prediction),
        "probability": float(probability),
        "execution_time_ms": float(execution_time),
        "model_version": "1.0",
        "status": "success"
    }
    
    # Push dans MongoDB
    # NOTE: pymongo Database objects do not implement truth value testing
    # Use explicit comparison with None to avoid TypeError
    if mongo_db is not None:
        try:
            mongo_db['predictions'].insert_one(log_entry)
            logger.info(f"✅ Log pushé dans MongoDB")
        except Exception as e:
            logger.error(f"❌ Erreur MongoDB: {e}")
    
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
        
        # Créer un array avec les bonnes dimensions (128 features attendues)
        X_complete = np.zeros((1, NUM_FEATURES))
        
        # Remplir les features disponibles
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
        elif probability < 0.5:
            risk_level = "🟡 RISQUE MOYEN"
        else:
            risk_level = "🔴 RISQUE ÉLEVÉ"
        
        # Logger la prédiction (ETAPE 3)
        log_prediction(input_dict, prediction, probability, execution_time)
        
        # Préparer la réponse
        result = f"Probabilité de défaut: {probability:.2%}\nLatence: {execution_time:.2f}ms"
        
        return result, risk_level, f"{probability:.4f}"
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la prédiction: {e}")
        return f"❌ Erreur: {str(e)}", "N/A", "N/A"


def compute_shap(ext_source_3, ext_source_2, amt_req_credit_bureau_day,
                 name_education_type, code_gender, flag_document_3,
                 ext_source_1, name_income_type, flag_emp_phone,
                 amt_goods_price, prev_approved_ratio, pos_count,
                 days_birth, own_car_age, flag_own_car,
                 name_contract_type, days_employed, reg_city_not_live_city,
                 def_60_cnt_social_circle, amt_credit):
    """
    Calcule et retourne les valeurs SHAP pour l'entrée fournie.
    Renvoie un texte listant les top features par importance SHAP.
    """
    if model is None:
        return "❌ ERREUR: Modèle non disponible"

    try:
        # Construire le dictionnaire d'entrée et la matrice complète
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

        X_complete = np.zeros((1, NUM_FEATURES))
        for feature_name, value in input_dict.items():
            if feature_name in EXPECTED_FEATURES:
                feature_idx = EXPECTED_FEATURES.index(feature_name)
                X_complete[0, feature_idx] = float(value)

        # Importer shap à la volée pour ne pas échouer au démarrage si non installé
        try:
            import shap
        except Exception:
            return "❌ La librairie 'shap' n'est pas installée. Installez-la avec: pip install shap"

        # Créer l'explainer selon le type de modèle
        shap_vals = None
        last_exc = None
        # 1) Essayer TreeExplainer directement
        try:
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_complete)
        except Exception as e_tree:
            logger.warning(f"TreeExplainer(model) a échoué: {e_tree}")
            last_exc = e_tree
            # 2) Si modèle XGBoost sklearn wrapper, essayer get_booster()
            try:
                if hasattr(model, 'get_booster'):
                    explainer = shap.TreeExplainer(model.get_booster())
                    shap_vals = explainer.shap_values(X_complete)
            except Exception as e_booster:
                logger.warning(f"TreeExplainer(model.get_booster()) a échoué: {e_booster}")
                last_exc = e_booster

        # 3) Fallback: utiliser shap.Explainer avec une fonction prédictive (predict_proba)
        if shap_vals is None:
            try:
                # Utiliser predict_proba pour expliquer la probabilité de la classe positive
                def predict_pos(X):
                    # s'assurer que X est un numpy array
                    arr = np.asarray(X)
                    # certains wrappers attendent 2D
                    return model.predict_proba(arr)[:, 1]

                # Créer un masker simple basé sur une ligne de zeros si disponible
                try:
                    masker = shap.maskers.Independent(np.zeros((1, NUM_FEATURES)))
                    explainer = shap.Explainer(predict_pos, masker)
                except Exception:
                    # Si shap.maskers indisponible ou usage différent, créer sans masker
                    explainer = shap.Explainer(predict_pos)

                shap_out = explainer(X_complete)
                shap_vals = shap_out.values
            except Exception as e_fallback:
                logger.error(f"Fallback shap.Explainer a échoué: {e_fallback}")
                return f"❌ Erreur SHAP (fallback): {str(e_fallback)}"

        # Pour les modèles de classification, shap_values peut être une liste (une per classe)
        if isinstance(shap_vals, list) or (hasattr(shap_vals, '__len__') and len(shap_vals) > 1 and isinstance(shap_vals[0], (list, tuple, np.ndarray))):
            # choisir la classe positive (1) si disponible, sinon la dernière
            try:
                shap_array = np.array(shap_vals[1]) if len(shap_vals) > 1 else np.array(shap_vals[-1])
            except Exception:
                shap_array = np.array(shap_vals[-1])
        else:
            shap_array = np.array(shap_vals)

        # shap_array devrait être de forme (1, n_features)
        shap_row = shap_array[0]

        feat_shap = list(zip(EXPECTED_FEATURES, shap_row))
        # Trier par importance absolue
        feat_shap_sorted = sorted(feat_shap, key=lambda x: abs(x[1]), reverse=True)

        # Construire le texte de sortie (top 10)
        top_k = 10
        lines = [f"Top {top_k} features par valeur SHAP:\n"]
        for i, (feat, val) in enumerate(feat_shap_sorted[:top_k], start=1):
            lines.append(f"{i}. {feat}: {val:.6f} (abs={abs(val):.6f})")

        return "\n".join(lines)

    except Exception as e:
        logger.error(f"❌ Erreur lors du calcul SHAP: {e}")
        return f"❌ Erreur SHAP: {str(e)}"


# Interface Gradio
with gr.Blocks(title="API de Scoring - Prêt à Dépenser") as demo:
    gr.Markdown("# 📊 API de Scoring de Crédit - Prêt à Dépenser")
    gr.Markdown("🎯 Prédiction du risque de défaut de paiement basée sur les 20 features les plus importantes")
    
    with gr.Row():
        gr.Markdown("### 📋 Entrez les informations du client:")
    
    # Les 20 features les plus importantes
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
    
    # Boutons d'action
    with gr.Row():
        predict_btn = gr.Button("🎯 Prédire le Risque", variant="primary", scale=2)
        clear_btn = gr.Button("🔄 Réinitialiser", scale=1)
    
    # Outputs
    with gr.Row():
        output_score = gr.Textbox(label="📈 Résultat", interactive=False, lines=2)
    
    with gr.Row():
        output_risk = gr.Textbox(label="⚠️  Niveau de Risque", interactive=False)
        output_probability = gr.Textbox(label="📊 Probabilité (0-1)", interactive=False)
        output_shap = gr.Textbox(label="🧭 SHAP - Top features", interactive=False, lines=12)
    
    # Connecter les inputs
    all_inputs = [
        ext_source_3, ext_source_2, amt_req_credit_bureau_day, name_education_type, code_gender,
        flag_document_3, ext_source_1, name_income_type, flag_emp_phone, amt_goods_price,
        prev_approved_ratio, pos_count, days_birth, own_car_age, flag_own_car,
        name_contract_type, days_employed, reg_city_not_live_city, def_60_cnt_social_circle, amt_credit
    ]
    
    all_outputs = [output_score, output_risk, output_probability]

    # Étendre la liste des outputs pour inclure SHAP
    all_outputs_with_shap = [output_score, output_risk, output_probability, output_shap]
    
    # Bouton prédire
    predict_btn.click(
        fn=predict_score,
        inputs=all_inputs,
        outputs=all_outputs
    )

    # Bouton pour calculer SHAP
    shap_btn = gr.Button("🧭 Obtenir SHAP", variant="secondary")
    shap_btn.click(
        fn=compute_shap,
        inputs=all_inputs,
        outputs=[output_shap]
    )
    
    # Bouton réinitialiser
    clear_btn.click(
        fn=lambda: (["", "", ""] + [""] + [0] * 20),
        inputs=[],
        outputs=all_outputs_with_shap + all_inputs
    )
    
    gr.Markdown("---")
    gr.Markdown("""
    ### 📝 Informations
    - **Modèle**: XGBoost avec 128 features
    - **Performance**: AUC = 0.7564, Accuracy = 69.63%
    - **Logging**: Chaque prédiction est pushée dans MongoDB Atlas
    - **Status**: MongoDB ✅ Connecté
    """)


if __name__ == "__main__":
    # Initialiser MongoDB au démarrage
    init_mongodb()
    
    # Charger le modèle
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
    
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
