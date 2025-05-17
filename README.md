# 🧠 Entraînement d’un modèle de traduction automatique avec OpenNMT-tf

Ce guide explique étape par étape comment entraîner un modèle de traduction automatique Anglais → Français avec OpenNMT-tf à partir de fichiers de données simples.

## 📁 Structure attendue des fichiers

project/
├── data/
│   ├── train.en         # Corpus source (anglais)
│   ├── train.fr         # Corpus cible (français)
│   ├── dev.en           # Corpus source de validation
│   ├── dev.fr           # Corpus cible de validation
│   ├── src-vocab.txt    # Vocabulaire source (généré)
│   └── tgt-vocab.txt    # Vocabulaire cible (généré)
├── config.yml           # Fichier de configuration OpenNMT
└── README.md            # Ce fichier

## ⚙️ Prérequis

✅ Créer un environnement virtuel compatible
OpenNMT-tf nécessite TensorFlow ≥ 2.6 et < 2.14, donc Python 3.8 à 3.10 est recommandé.

Voici les étapes pour créer un environnement Python propre :

## Crée un environnement virtuel avec Python 3.10

python3.10 -m venv onmt-env

### Active-le

source onmt-env/bin/activate      # Linux/macOS

### Mets pip à jour

pip install --upgrade pip

### Installe OpenNMT-tf avec TensorFlow compatible

pip install "tensorflow>=2.10,<2.14" opennmt-tf

### 📊 Préparer les données

Créer les fichiers de données
Place dans data/ les fichiers suivants :

train.en : une phrase anglaise par ligne (ex : Hello how are you ?)

train.fr : la traduction française correspondante (ex : Bonjour comment ça va ?)

dev.en et dev.fr : mêmes formats, pour la validation

⚠️ Les lignes doivent se correspondre 1 à 1 entre source et cible.

### 🧠  Générer les vocabulaires

```bash
onmt-build-vocab --size 5000 --save_vocab data/src-vocab.txt data/train.en
onmt-build-vocab --size 5000 --save_vocab data/tgt-vocab.txt data/train.fr
```
