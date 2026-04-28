# Dockerfile pour l'API de scoring Gradio
# Utilise Python 3.11 comme base

FROM python:3.11-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers requis
COPY requirements.txt .
COPY app.py .
COPY exported_model/ ./exported_model/

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port Gradio (7860)
EXPOSE 7860

# Commande de démarrage
CMD ["python", "app.py"]
