"""
Exemple d'utilisation du modèle exporté
"""

import pickle
import pandas as pd

# Charger le modèle
with open('./model/model.pkl', 'rb') as f:
    model = pickle.load(f)

# Charger les données
X_new = pd.read_csv('new_data.csv')

# Vérifier que les colonnes correspondent
expected_features = {model.feature_names_in_}
actual_features = set(X_new.columns)

if expected_features != actual_features:
    print(f"⚠️  Colonnes manquantes: {expected_features - actual_features}")
    # Sélectionner les bonnes colonnes et dans le bon ordre
    X_new = X_new[model.feature_names_in_]

# Faire une prédiction
predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)[:, 1]

# Résultats
results = pd.DataFrame({
    'prediction': predictions,
    'probability': probabilities,
    'risk_level': ['BAS' if p < 0.3 else 'MOYEN' if p < 0.5 else 'ÉLEVÉ' for p in probabilities]
})

print(results)
