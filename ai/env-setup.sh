#!/bin/bash

echo "Creating virtual environment..."
python3 -m venv venv

echo "Activating environment..."
source venv/bin/activate

echo "Upgrading pip..."
python -m pip install --upgrade pip

echo "Installing Python libraries..."
pip install -r requirements.txt

echo "Downloading NLTK data..."
python -m nltk.downloader punkt stopwords wordnet averaged_perceptron_tagger

echo "Setup complete!"
