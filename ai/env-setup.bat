@echo off

echo Creating virtual environment...
python -m venv venv

echo Activating environment...
call venv\Scripts\activate

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing Python libraries...
pip install -r requirements.txt

echo Downloading NLTK data...
python -m nltk.downloader punkt stopwords wordnet averaged_perceptron_tagger

echo Setup complete!
pause