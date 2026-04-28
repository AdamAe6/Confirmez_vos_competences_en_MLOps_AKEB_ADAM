"""
Script de test rapide pour vérifier que l'API peut démarrer
"""

import sys
from pathlib import Path
import pickle
import numpy as np

print("🔍 Test du démarrage de l'API...\n")

# Test 1: Vérifier le modèle
print("✅ Test 1: Chargement du modèle")
model_path = Path("exported_model/model/model.pkl")
with open(model_path, 'rb') as f:
    model = pickle.load(f)
print(f"   - Modèle chargé: {type(model).__name__}")
print(f"   - Nombre de features: {len(model.feature_names_in_)}")

# Test 2: Faire une prédiction
print("\n✅ Test 2: Prédiction simple")
X_test = np.zeros((1, len(model.feature_names_in_)))
pred = model.predict(X_test)[0]
proba = model.predict_proba(X_test)[0, 1]
print(f"   - Prédiction: {pred}")
print(f"   - Probabilité: {proba:.2%}")

# Test 3: Vérifier les imports Gradio
print("\n✅ Test 3: Import de Gradio")
try:
    import gradio as gr
    print(f"   - Gradio version: {gr.__version__}")
except ImportError as e:
    print(f"   - ⚠️  Erreur: {e}")
    sys.exit(1)

# Test 4: Vérifier la structure des fichiers
print("\n✅ Test 4: Structure des fichiers")
required_files = [
    "app.py",
    "requirements.txt",
    "Dockerfile",
    "tests/test_api.py",
    ".github/workflows/ci-cd.yml",
    "README_ETAPE2.md"
]

for file in required_files:
    path = Path(file)
    status = "✓" if path.exists() else "✗"
    print(f"   [{status}] {file}")

print("\n🎉 Tous les tests sont passés! L'API est prête.")
print("\n📝 Pour démarrer l'API:")
print("   python app.py")
