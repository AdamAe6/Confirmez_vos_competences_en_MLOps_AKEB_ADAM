"""
Tests unitaires pour l'API de scoring
"""

import pytest
import sys
from pathlib import Path
import pickle
import numpy as np

# Ajouter le répertoire parent au chemin
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestModelLoading:
    """Tests du chargement du modèle"""
    
    def test_model_file_exists(self):
        """Vérifier que le fichier du modèle existe"""
        model_path = Path("exported_model/model/model.pkl")
        assert model_path.exists(), "Le fichier du modèle n'existe pas"
    
    def test_model_can_be_loaded(self):
        """Vérifier que le modèle peut être chargé"""
        model_path = Path("exported_model/model/model.pkl")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        assert model is not None, "Le modèle n'a pas pu être chargé"
        assert hasattr(model, 'predict'), "Le modèle n'a pas de méthode predict"
        assert hasattr(model, 'predict_proba'), "Le modèle n'a pas de méthode predict_proba"
    
    def test_model_has_feature_names(self):
        """Vérifier que le modèle a les noms de features"""
        model_path = Path("exported_model/model/model.pkl")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        assert hasattr(model, 'feature_names_in_'), "Le modèle n'a pas d'attribut feature_names_in_"
        assert len(model.feature_names_in_) > 0, "Le modèle n'a pas de features"


class TestModelPrediction:
    """Tests des prédictions du modèle"""
    
    @pytest.fixture
    def model(self):
        """Charger le modèle pour les tests"""
        model_path = Path("exported_model/model/model.pkl")
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    
    def test_model_predict_returns_valid_output(self, model):
        """Vérifier que les prédictions retournent des valeurs valides"""
        # Créer des données de test (128 features d'après le README)
        num_features = len(model.feature_names_in_)
        X_test = np.zeros((1, num_features))
        
        predictions = model.predict(X_test)
        assert predictions is not None, "Les prédictions sont None"
        assert len(predictions) == 1, "Le nombre de prédictions est incorrect"
        assert predictions[0] in [0, 1], "La prédiction doit être 0 ou 1"
    
    def test_model_predict_proba_returns_valid_output(self, model):
        """Vérifier que predict_proba retourne des probabilités valides"""
        num_features = len(model.feature_names_in_)
        X_test = np.zeros((1, num_features))
        
        probabilities = model.predict_proba(X_test)
        assert probabilities is not None, "Les probabilités sont None"
        assert probabilities.shape[1] == 2, "Doit retourner 2 probabilités"
        assert all(0 <= p <= 1 for p in probabilities[0]), "Les probabilités doivent être entre 0 et 1"
        assert abs(sum(probabilities[0]) - 1.0) < 0.001, "La somme des probabilités doit être 1"
    
    def test_model_handles_multiple_samples(self, model):
        """Vérifier que le modèle peut traiter plusieurs échantillons"""
        num_features = len(model.feature_names_in_)
        X_test = np.zeros((5, num_features))
        
        predictions = model.predict(X_test)
        assert len(predictions) == 5, "Le modèle doit traiter 5 échantillons"
    
    def test_model_prediction_consistency(self, model):
        """Vérifier que le modèle est déterministe"""
        num_features = len(model.feature_names_in_)
        X_test = np.ones((1, num_features))
        
        pred1 = model.predict(X_test)[0]
        pred2 = model.predict(X_test)[0]
        
        assert pred1 == pred2, "Le modèle doit être déterministe"


class TestInputValidation:
    """Tests de validation des entrées"""
    
    def test_invalid_feature_count(self):
        """Vérifier que les données avec un nombre de features invalide sont rejetées"""
        model_path = Path("exported_model/model/model.pkl")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        num_features = len(model.feature_names_in_)
        X_test = np.zeros((1, num_features - 1))  # Une feature en moins
        
        with pytest.raises(Exception):
            model.predict(X_test)
    
    def test_negative_values_handled(self):
        """Vérifier que le modèle gère les valeurs négatives"""
        model_path = Path("exported_model/model/model.pkl")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        num_features = len(model.feature_names_in_)
        X_test = np.full((1, num_features), -100)  # Valeurs négatives
        
        # Le modèle doit pouvoir les traiter (pas de levée d'exception)
        predictions = model.predict(X_test)
        assert predictions is not None
    
    def test_large_values_handled(self):
        """Vérifier que le modèle gère les valeurs très grandes"""
        model_path = Path("exported_model/model/model.pkl")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        num_features = len(model.feature_names_in_)
        X_test = np.full((1, num_features), 1e6)  # Valeurs très grandes
        
        predictions = model.predict(X_test)
        assert predictions is not None
        assert predictions[0] in [0, 1]


class TestAPIStructure:
    """Tests de la structure de l'API"""
    
    def test_app_file_exists(self):
        """Vérifier que le fichier app.py existe"""
        app_path = Path("app.py")
        assert app_path.exists(), "Le fichier app.py n'existe pas"
    
    def test_requirements_file_exists(self):
        """Vérifier que requirements.txt existe"""
        req_path = Path("requirements.txt")
        assert req_path.exists(), "Le fichier requirements.txt n'existe pas"
    
    def test_dockerfile_exists(self):
        """Vérifier que le Dockerfile existe"""
        docker_path = Path("Dockerfile")
        assert docker_path.exists(), "Le Dockerfile n'existe pas"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
