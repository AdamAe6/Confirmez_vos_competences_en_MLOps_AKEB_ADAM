# API de Scoring - Prêt à Dépenser

Un projet MLOps complet mettant en production un modèle de scoring XGBoost. Le projet inclut une API Gradio, du monitoring en temps réel, des tests, de la conteneurisation Docker et un pipeline CI/CD.

## Vue d'ensemble

L'objectif du projet est de mettre en production un modèle de scoring pour l'évaluation des demandes de crédit. On part du modèle versionnée avec MLflow, puis on le déploie via une API, on surveille ses performances et on l'optimise si possible.

Le projet couvre 4 étapes principales:

- **ÉTAPE 1**: Versionning Git et structure du projet
- **ÉTAPE 2**: API fonctionnelle, tests et Docker
- **ÉTAPE 3**: Monitoring et analyse du drift des données
- **ÉTAPE 4**: Benchmarking et optimisation des performances

## Structure du projet

```
.
├── app.py                           # API Gradio (point d'entrée principal)
├── dashboard_monitoring.py          # Dashboard Streamlit pour le monitoring
├── analyze_monitoring.py            # Analyse du drift et des performances
├── bench_inference.py               # Benchmark de latence
├── profile_inference.py             # Profiling cProfile
├── eval_after_opt.py                # Comparaison modèle original vs optimisé
│
├── exported_model/                  # Artefacts du modèle MLflow
│   └── model/
│       ├── model.pkl               # Le modèle XGBoost
│       ├── MLmodel                 # Métadonnées MLflow
│       └── conda.yaml
│
├── tests/                           # Tests unitaires
│   └── test_api.py
│
├── logs/                            # Logs et données de production
│   ├── predictions_*.csv
│   └── predictions_*.jsonl
│
├── reports/                         # Rapports et benchmarks
│   ├── bench_baseline.json
│   └── infer.prof
│
├── Dockerfile                       # Conteneurisation
├── requirements.txt                 # Dépendances Python
├── .env.example                     # Variables d'environnement (exemple)
└── README.md                        # Cette documentation
```

## Installation et configuration

### Prérequis

- **Python 3.11+**
- **pip** ou **conda**
- **Docker** (optionnel, pour le déploiement)
- **Git**
- **MongoDB Atlas** (optionnel, pour le logging en prod)

### 1. Cloner le projet

```bash
git clone https://github.com/AdamAe6/Confirmez_vos_competences_en_MLOps_AKEB_ADAM.git
cd Confirmez_vos_competences_en_MLOps_AKEB_ADAM
```

### 2. Créer un environnement virtuel

```bash
python3 -m venv venv
source venv/bin/activate  # Sur macOS/Linux
# ou
venv\Scripts\activate     # Sur Windows
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Configurer les variables d'environnement

Créer un fichier `.env` à la racine du projet (optionnel mais recommandé):

```env
# Si vous utilisez MongoDB Atlas pour logger les prédictions
MONGO_URI=mongodb+srv://username:password@cluster.mongodb.net/?retryWrites=true&w=majority

# Sinon, MongoDB local
MONGO_URI=mongodb://localhost:27017
```

Pour un test rapide sans MongoDB, l'API fonctionnera mais les logs ne seront pas persistés en base.

## Démarrage rapide

### Lancer l'API

```bash
python app.py
```

L'API sera accessible sur `http://localhost:7860`

L'interface Gradio affiche:

- Un formulaire pour saisir les features du client
- Le score de crédit prédis (0-1)
- La classe prédite (Refusé/Accordé)
- Le temps de réponse

### Lancer le dashboard de monitoring

Dans un autre terminal:

```bash
streamlit run dashboard_monitoring.py
```

Le dashboard sera accessible sur `http://localhost:8501`

On peut y voir:

- Historique des prédictions
- Distribution des scores
- Latence de l'API
- Détection du data drift

## Tests

### Tester le démarrage rapide

```bash
python test_startup.py
```

Cela vérifie que:

- Le modèle se charge correctement
- Les predictions fonctionnent
- Les dépendances sont installées
- La structure des fichiers est OK

### Tester l'API complètement

```bash
pytest tests/ -v
```

Les tests vérifient:

- Chargement du modèle
- Prédictions valides
- Format des données attendues
- Intégration MongoDB (si disponible)

Avec couverture de code:

```bash
pytest tests/ --cov=. --cov-report=html
```

## ÉTAPE 1: Versionning Git

Le projet est versionné avec Git et poussé sur GitHub.

Commits majeurs:

- Structure initiale du projet
- Intégration du modèle XGBoost
- API Gradio avec tests
- Containerisation Docker
- Dashboard Streamlit
- Pipeline CI/CD
- Optimisations et benchmarks

Voir l'historique complet:

```bash
git log --oneline
```

Ou directement sur [GitHub](https://github.com/AdamAe6/Confirmez_vos_competences_en_MLOps_AKEB_ADAM/commits/main).

## ÉTAPE 2: API et Déploiement

### L'API Gradio

L'API est implémentée avec Gradio dans `app.py`. Elle:

- Charge le modèle XGBoost au démarrage
- Expose une interface web pour les prédictions
- Loggue chaque prédiction dans un CSV local + MongoDB (si disponible)
- Mesure le temps de réponse
- Gère les erreurs gracieusement

### Tests unitaires

Les tests dans `tests/test_api.py` vérifient:

- Existence et chargement du modèle
- Validité des prédictions
- Format des données
- Présence des features attendues

### Containerisation Docker

Build et run l'image Docker:

```bash
# Build
docker build -t api-scoring:v1 .

# Run
docker run -p 7860:7860 api-scoring:v1

# Avec variables d'environnement
docker run -p 7860:7860 \
  -e MONGO_URI="mongodb://mongo:27017" \
  api-scoring:v1
```

## ÉTAPE 3: Monitoring et Data Drift

### Dashboard Streamlit

`dashboard_monitoring.py` fournit un dashboard complet avec:

- **Vue des prédictions**: Affiche l'historique des scores et clients
- **Distribution des scores**: Histogramme et statistiques
- **Latence de l'API**: Évolution du temps de réponse
- **Data Drift**: Comparaison distribution features entrantes vs données d'entraînement
- **Alertes**: Détecte les anomalies

### Chargement des données

Le dashboard charge les prédictions depuis:

1. **MongoDB** (si disponible): Collection `predictions` avec toutes les prédictions en prod
2. **Fichiers CSV locaux** (fallback): `logs/predictions_*.csv`
3. **Fichiers JSONL** (fallback): `logs/predictions_*.jsonl`

### Analyse du drift

Le script `analyze_monitoring.py` analyse:

- Distribution des features
- Similarité avec données d'entraînement
- Statistiques descriptives
- Alertes sur anomalies

```bash
python analyze_monitoring.py
```

## ÉTAPE 4: Benchmarking et Optimisation

### Benchmark simple

Mesurer la latence moyenne de prédiction sur 200 exemples:

```bash
python bench_inference.py \
  --model exported_model/model/model.pkl \
  --n 200 \
  --out reports/bench_baseline.json
```

Résultat dans `reports/bench_baseline.json`:

```json
{
  "model_path": "exported_model/model/model.pkl",
  "n_predictions": 200,
  "latency_mean_ms": 2.5,
  "latency_std_ms": 0.8,
  "latency_min_ms": 1.2,
  "latency_max_ms": 5.1,
  "throughput_predictions_per_sec": 400
}
```

### Profiling avec cProfile

Générer un profil détaillé pour identifier les goulots d'étranglement:

```bash
python -m cProfile -o reports/infer.prof profile_inference.py \
  --model exported_model/model/model.pkl \
  --n 100
```

Analyser le profil:

```bash
python -m pstats reports/infer.prof
# Dans le shell: sort cumulative, stats 10
```

### Comparaison modèle original vs optimisé

Si vous avez optimisé le modèle (ONNX, quantization, etc.):

```bash
python eval_after_opt.py \
  --orig exported_model/model/model.pkl \
  --opt exported_model/model/model_optimized.pkl \
  --testdata tests/test_labels.csv \
  --out reports/eval_compare.json
```

Le rapport compare:

- Latence moyenne
- Accuracité (perte de performance)
- Gain en temps de calcul

## Pipeline CI/CD

Le projet inclut un workflow GitHub Actions (`.github/workflows/ci-cd.yml`) qui:

- Lance les tests à chaque push
- Valide le format du code
- Build l'image Docker
- (Optionnel) Déploie en production

Voir le statut des builds: [Actions](https://github.com/AdamAe6/Confirmez_vos_competences_en_MLOps_AKEB_ADAM/actions)

## Déploiement sur Railway

Pour le déploiement en production, on utilise **Railway**. C'est une plateforme simple et efficace pour déployer rapidement un POC (Proof of Concept) sans avoir besoin de gérer l'infrastructure complètement.

### Pourquoi Railway?

- **Déploiement automatique**: Git connected - déploie automatiquement à chaque push/merge sur `main`
- **Simple et rapide**: Pas de configuration complexe d'infrastructure
- **Build automatique**: Lit le Dockerfile, build et lance le conteneur directement
- **Environnement variable intégré**: Gère facilement `.env` et `MONGO_URI`
- **Gratuit/Payant flexible**: Bon pour les POC, scaling facile après

### Comment ça fonctionne

#### 1. Se connecter avec GitHub

- Aller sur [railway.app](https://railway.app)
- S'inscrire avec GitHub
- Autoriser Railway à accéder au dépôt

#### 2. Créer un nouveau projet

- Cliquer "New Project"
- Sélectionner "Deploy from GitHub"
- Choisir le repo `Confirmez_vos_competences_en_MLOps_AKEB_ADAM`
- Railway détecte automatiquement le Dockerfile

#### 3. Configuration des variables d'environnement

Dans le dashboard Railway:

- Aller dans "Variables"
- Ajouter `MONGO_URI` (si vous utilisez MongoDB Atlas)
- Ajouter d'autres variables si nécessaire

Example:

```
MONGO_URI=mongodb+srv://user:pass@cluster.mongodb.net/?retryWrites=true&w=majority
```

#### 4. Deploy automatique

À partir de là, chaque push et/ou merge sur la branche `main` va:

1. Déclencher un build automatique
2. Exécuter le Dockerfile
3. Exposer l'API sur `https://<your-app>.railway.app`

Pas besoin de faire quoi que ce soit - c'est complètement automatisé!

### Monitoring du déploiement

- **Logs en direct**: Dashboard Railway montre les logs en temps réel
- **URL de l'app**: `https://<your-app-name>.railway.app`
- **Rebuild**: Cliquer "Deploy" pour relancer manuellement si besoin
- **Rollback**: Revenir à une version précédente en un clic

### Exemple d'URL de l'API

Après déploiement:

```
https://api-scoring-prod.railway.app
```

L'interface Gradio sera accessible à cette URL automatiquement.

### Variables d'environnement en production

Railway injecte les variables automatiquement au démarrage. L'app.py charge via:

```python
from dotenv import load_dotenv
import os

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
```

### Avantages pour ce projet

- **Validation rapide**: Déployer en 2 min après avoir push
- **POC efficace**: Montrer à Chloé (la Lead Data) que c'est production-ready
- **Scaling facile**: Si ça marche bien, migrer vers un vrai k8s après
- **Git workflow naturel**: Le déploiement se fait en faisant un simple push

### Troubleshooting Railway

**Build échoue?**

- Vérifier les logs en live dans le dashboard
- Vérifier que `requirements.txt` est à jour
- Vérifier que `Dockerfile` existe et est correct

**App crash au démarrage?**

- Vérifier les logs (section "Logs" du dashboard)
- Généralement c'est un problème de `MONGO_URI` - vérifier la variable d'env
- Sinon, vérifier que le modèle existe dans l'image Docker

**Besoin d'update rapidement?**

- Push sur main → build auto en ~2 min
- Pas besoin d'aller en production manuellement

## Variables d'environnement

Créer `.env` à la racine avec:

```env
# MongoDB Atlas
MONGO_URI=mongodb+srv://user:pass@cluster.mongodb.net/?retryWrites=true&w=majority

# Ou MongoDB local
MONGO_URI=mongodb://localhost:27017

# Mode développement
DEBUG=True
LOG_LEVEL=INFO
```

L'API fonctionne sans `.env` (valeurs par défaut), mais le logging MongoDB ne sera pas disponible.

## Dépendances

Principales dépendances installées:

- **gradio** (≥4.0.0): Interface API web
- **xgboost** (≥2.0.0): Modèle ML
- **scikit-learn** (≥1.3.0): Utilitaires ML
- **pandas** (≥2.0.0): Manipulation de données
- **numpy** (≥1.24.0): Calculs numériques
- **streamlit** (≥1.20.0): Dashboard
- **plotly** (≥5.10.0): Graphiques interactifs
- **pymongo** (≥4.0.0): Connexion MongoDB
- **python-dotenv** (≥1.0.0): Variables d'environnement
- **pytest** (≥7.0.0): Tests unitaires
- **shap** (≥0.42.1): Explainability

Voir `requirements.txt` pour la liste complète.

## Résumé des validations par étape

### ÉTAPE 1 ✓

- [x] Dépôt Git public sur GitHub
- [x] Structure claire du projet
- [x] Commits explicites
- [x] .gitignore configuré
- [x] README initial

### ÉTAPE 2 ✓

- [x] API Gradio fonctionnelle
- [x] Tests unitaires (pytest)
- [x] Dockerfile
- [x] Logs prédictions en CSV/JSONL
- [x] Gestion d'erreurs
- [x] Mesure de latence

### ÉTAPE 3 ✓

- [x] Dashboard Streamlit
- [x] Chargement MongoDB/CSV
- [x] Distribution des scores
- [x] Analyse du drift
- [x] Histogramme latences
- [x] Alertes anomalies

### ÉTAPE 4 ✓

- [x] Script benchmark
- [x] Profiling cProfile
- [x] Comparison modèle optimisé
- [x] Rapports JSON
- [x] Documentation détaillée

## Troubleshooting

### "ModuleNotFoundError: No module named 'gradio'"

```bash
pip install gradio
```

### "Cannot connect to MongoDB"

Si vous ne voulez pas MongoDB, l'API fonctionne quand même (pas de logging en base). Pour MongoDB local:

```bash
# Sur macOS
brew services start mongodb-community

# Ou lancer MongoDB en Docker
docker run -d -p 27017:27017 mongo
```

### "Port 7860 already in use"

Gradio utilise le port 7860 par défaut. Changer le port:

```python
# Dans app.py, ajouter avant .launch():
iface.launch(server_name="127.0.0.1", server_port=8000)
```

### Les tests échouent

Vérifier que le modèle existe:

```bash
ls -la exported_model/model/model.pkl
```

Si absent, vous devez d'abord exécuter le projet précédent (Initiez-vous au MLOps) pour générer le modèle.

## Notes de développement

- Le projet utilise **Python 3.11** (testé sur macOS)
- Gradio fonctionne mieux sur macOS/Linux; sur Windows il faut peut-être ajouter `share=True` pour le lancer correctement
- Les logs sont stockés localement en CSV par défaut; utiliser MongoDB Atlas pour la persistance en prod
- Le dashboard scrape les données MongoDB en temps réel; il y a une latence si beaucoup de données
