# 📦 Modèle XGBoost Exporté

## 📊 Informations

- **Modèle**: XGBClassifier
- **Date d'export**: 2026-04-27T18:15:42.502047
- **Nombre de features**: 128

## 📈 Performance

- **AUC**: 0.7564
- **Accuracy**: 0.6963
- **Cost**: 32622

Note: XGBoost baseline model - AUC: 0.7564

## 🏆 Top 10 Features

| Rang | Feature | Importance |
|------|---------|-----------|
| 1 | EXT_SOURCE_3 | 0.100976 |
| 2 | EXT_SOURCE_2 | 0.086858 |
| 3 | AMT_REQ_CREDIT_BUREAU_DAY | 0.049698 |
| 4 | NAME_EDUCATION_TYPE | 0.032165 |
| 5 | CODE_GENDER | 0.030557 |
| 6 | FLAG_DOCUMENT_3 | 0.028984 |
| 7 | EXT_SOURCE_1 | 0.024832 |
| 8 | NAME_INCOME_TYPE | 0.020197 |
| 9 | FLAG_EMP_PHONE | 0.019736 |
| 10 | AMT_GOODS_PRICE | 0.019563 |


## 🚀 Utilisation Rapide

### 1. Charger le modèle

```python
import pickle

with open('./model/model.pkl', 'rb') as f:
    model = pickle.load(f)
```

### 2. Faire une prédiction

```python
import pandas as pd

X_new = pd.read_csv('new_data.csv')
predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)[:, 1]
```

### 3. Exemple complet

Voir `example_usage.py`

## ⚠️ Points importants

1. **Features doivent être identiques**
   - Même nom
   - Même ordre
   - Aucune colonne supplémentaire

2. **Versions des librairies**
   ```bash
   pip install -r model/requirements.txt
   ```

3. **Pas de preprocessing** - Le modèle attend les données brutes

## 📂 Structure

```
exported_model/
├── model/
│   ├── model.pkl          # Modèle pickle
│   ├── MLmodel            # Métadonnées MLflow
│   ├── requirements.txt    # Dépendances
│   └── conda.yaml         # Environnement Conda
├── config.json            # Configuration
├── example_usage.py       # Exemple d'utilisation
└── README.md             # Ce fichier
```

## 🔍 Vérifications

```python
# Vérifier le modèle
import pickle

with open('./model/model.pkl', 'rb') as f:
    model = pickle.load(f)

print(f"Type: <class 'xgboost.sklearn.XGBClassifier'>")
print(f"Features: 128")
print(f"Top feature: NAME_CONTRACT_TYPE")
```

## 📞 Troubleshooting

### Feature mismatch
```python
# Solution: Vérifier les colonnes
print("Attendu:", model.feature_names_in_)
print("Obtenu:", X_new.columns)
```

### Version incompatible
```bash
# Solution: Installer les bonnes versions
pip install --upgrade -r model/requirements.txt
```

---

Export du modèle XGBoost (AUC: 0.7564)
