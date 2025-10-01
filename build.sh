#!/bin/bash

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Download NLTK data
python -c "
import nltk
import os

# Set NLTK data path for Render
nltk_data_path = '/opt/render/nltk_data'
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

# Download required NLTK packages
print('Downloading NLTK packages...')
nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('punkt_tab', download_dir=nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)
nltk.download('wordnet', download_dir=nltk_data_path)
nltk.download('omw-1.4', download_dir=nltk_data_path)
print('NLTK packages downloaded successfully!')
"

echo "Build completed successfully!"